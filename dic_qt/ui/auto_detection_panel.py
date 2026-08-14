from __future__ import annotations

import json
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, QObject, QThread, Qt, Slot
from PySide6.QtCore import QPointF, QRectF, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from dic_qt.core.auto_pipeline import (
    AutoPipelineParams,
    default_display_range,
    display_image,
    image_size,
    image_value_summary,
    load_region,
    run_boundary_cut_events,
    run_mask_event_detection,
    run_hough_seed_detection,
    run_preprocessing,
    trace_hough_events,
)
from dic_qt.ui.alignment_analysis_panel import AlignmentAnalysisPanel


MAX_CROP_PREVIEW_DIM = 1800
MAX_OVERLAY_PREVIEW_DIM = 2400
DISPLAY_ADJUSTMENT_SCALE = 2.0
AUTO_PARAMETER_DEFAULTS_PATH = Path(__file__).resolve().parents[2] / ".dic_qt_auto_parameter_defaults.json"


def display_crop_rgb_fast(
    arr: np.ndarray,
    display_min: float,
    display_max: float,
    brightness_adjust: float = 0.0,
) -> np.ndarray:
    values = np.nan_to_num(np.squeeze(arr).astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim == 3 and values.shape[2] >= 3:
        rgb = values[..., :3]
        lo = float(rgb.min())
        hi = float(rgb.max())
        if hi <= lo:
            scaled = np.zeros(rgb.shape, dtype=np.uint8)
        else:
            scaled = (np.clip((rgb - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
        return scaled
    if display_max <= display_min:
        gray = np.zeros(values.shape, dtype=np.uint8)
    else:
        gray = (np.clip((values - display_min) / (display_max - display_min), 0.0, 1.0) * 255).astype(np.uint8)
    if abs(float(brightness_adjust)) > 1e-9:
        gray = np.clip(
            gray.astype(np.float32, copy=False) + float(brightness_adjust) / 100.0 * 127.5,
            0.0,
            255.0,
        ).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


@lru_cache(maxsize=4)
def load_downsampled_preview_array(
    path: str,
    max_dim: int = MAX_CROP_PREVIEW_DIM,
) -> tuple[np.ndarray, float, float]:
    with Image.open(path) as img:
        original_width, original_height = img.size
        scale = min(1.0, max_dim / max(original_width, original_height))
        preview_width = max(1, int(round(original_width * scale)))
        preview_height = max(1, int(round(original_height * scale)))
        preview = img.resize((preview_width, preview_height), Image.Resampling.BILINEAR)
        arr = np.asarray(preview).copy()
    scale_x = original_width / preview_width
    scale_y = original_height / preview_height
    return arr, scale_x, scale_y


def load_downsampled_preview(
    path: str,
    display_min: float,
    display_max: float,
    brightness_adjust: float = 0.0,
    max_dim: int = MAX_CROP_PREVIEW_DIM,
) -> tuple[np.ndarray, float, float]:
    arr, scale_x, scale_y = load_downsampled_preview_array(path, max_dim)
    rgb = display_crop_rgb_fast(arr, display_min, display_max, brightness_adjust)
    return rgb, scale_x, scale_y


def downsample_rgb_for_view(
    image: np.ndarray,
    max_dim: int = MAX_OVERLAY_PREVIEW_DIM,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> np.ndarray:
    if max(image.shape[:2]) <= max_dim:
        return image
    pil_image = Image.fromarray(display_image(image))
    pil_image.thumbnail((max_dim, max_dim), resample)
    return np.asarray(pil_image).copy()


def sanitize_display_range(value: tuple[float, float], raw_min: float, raw_max: float) -> tuple[float, float]:
    display_min, display_max = value
    display_min = max(raw_min, min(float(display_min), raw_max))
    display_max = max(raw_min, min(float(display_max), raw_max))
    if display_max <= display_min:
        display_max = min(raw_max, display_min + max((raw_max - raw_min) / 2000.0, 0.0001))
        if display_max <= display_min:
            display_min = raw_min
            display_max = raw_max
    return display_min, display_max


def display_range_from_adjustments(
    base_range: tuple[float, float],
    contrast: float,
    raw_min: float,
    raw_max: float,
) -> tuple[float, float]:
    raw_min = float(raw_min)
    raw_max = float(raw_max)
    span = raw_max - raw_min
    if span <= 0:
        return raw_min, raw_max

    min_width = max(span / 2000.0, 0.0001)
    base_min, base_max = sanitize_display_range(base_range, raw_min, raw_max)
    base_center = (base_min + base_max) / 2.0
    base_width = max(min_width, min(base_max - base_min, span))
    contrast = float(np.clip(contrast, -100.0, 100.0))

    center = base_center
    if contrast >= 0:
        width = base_width * (1.0 - 0.95 * (contrast / 100.0))
    else:
        width = base_width + (span - base_width) * (abs(contrast) / 100.0)

    width = max(min_width, min(width, span))
    center = max(raw_min, min(center, raw_max))
    display_min = center - width / 2.0
    display_max = center + width / 2.0
    if display_min < raw_min:
        display_max += raw_min - display_min
        display_min = raw_min
    if display_max > raw_max:
        display_min -= display_max - raw_max
        display_max = raw_max
    return sanitize_display_range((display_min, display_max), raw_min, raw_max)


def load_downsampled_processing_region(
    params: AutoPipelineParams,
    max_dim: int = MAX_CROP_PREVIEW_DIM,
) -> dict:
    with Image.open(params.image_path) as img:
        image_width, image_height = img.size
        if params.use_full_image:
            x0, y0, crop_width, crop_height = 0, 0, image_width, image_height
        else:
            crop_width = max(1, min(int(params.crop_width), image_width))
            crop_height = max(1, min(int(params.crop_height), image_height))
            x0 = max(0, min(int(params.crop_x), image_width - crop_width))
            y0 = max(0, min(int(params.crop_y), image_height - crop_height))
        crop = img.crop((x0, y0, x0 + crop_width, y0 + crop_height))
        scale = min(1.0, max_dim / max(crop_width, crop_height))
        preview_width = max(1, int(round(crop_width * scale)))
        preview_height = max(1, int(round(crop_height * scale)))
        if scale < 1.0:
            crop = crop.resize((preview_width, preview_height), Image.Resampling.BILINEAR)
        arr = np.asarray(crop).copy()
    display_rgb = display_crop_rgb_fast(arr, params.display_min, params.display_max, params.display_brightness)
    return {
        "display_rgb": display_rgb,
        "detection_rgb": display_rgb,
        "origin": (x0, y0),
        "preview_scale": 1.0 / max(scale, 1e-9),
    }


def run_preprocessing_preview(params: AutoPipelineParams) -> dict:
    from scipy import ndimage as ndi
    from skimage import measure, morphology
    from skimage.exposure import equalize_adapthist
    from skimage.filters import meijering, threshold_otsu

    crop = load_downsampled_processing_region(params)
    gray = crop["detection_rgb"][..., 2].astype(np.float32, copy=False) / 255.0
    enhanced = equalize_adapthist(gray, clip_limit=params.clahe_clip_limit).astype(np.float32)
    ridges = meijering(
        enhanced,
        sigmas=tuple(range(1, max(1, params.ridge_sigma_max) + 1)),
        black_ridges=False,
    ).astype(np.float32)
    percentile_threshold = float(np.percentile(ridges, params.ridge_percentile))
    otsu_threshold = float(threshold_otsu(ridges) * params.threshold_multiplier)
    ridge_threshold = max(percentile_threshold, otsu_threshold)
    candidate_mask = ridges > ridge_threshold
    labels = measure.label(candidate_mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= max(1, int(params.min_object_size))
    if keep.size:
        keep[0] = False
    candidate_clean = keep[labels]
    if params.closing_radius > 0:
        candidate_clean = morphology.closing(candidate_clean, morphology.disk(params.closing_radius))
    detection_mask = morphology.skeletonize(candidate_clean) if params.use_skeletonize else candidate_clean
    display_mask = (
        ndi.binary_dilation(detection_mask, iterations=2)
        if params.use_skeletonize
        else detection_mask
    )
    return {
        "params": params,
        "crop": crop,
        "enhanced": enhanced,
        "ridges": ridges,
        "candidate_mask": candidate_mask,
        "candidate_clean": candidate_clean,
        "display_mask": display_mask,
    }


@contextmanager
def busy_cursor():
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    QApplication.processEvents()
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()
        QApplication.processEvents()


class ImagePreview(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(260, 220)
        self.setStyleSheet("QLabel { background: #151515; color: #dddddd; border: 1px solid #333333; }")
        self._pixmap: QPixmap | None = None

    def set_array(self, image: np.ndarray | None) -> None:
        if image is None:
            self._pixmap = None
            self.clear()
            return
        arr = np.ascontiguousarray(display_image(image))
        height, width, channels = arr.shape
        bytes_per_line = channels * width
        qimage = QImage(arr.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self._fit_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_pixmap()

    def _fit_pixmap(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class CropCanvas(QGraphicsView):
    crop_changed = Signal(int, int, int, int)

    def __init__(self) -> None:
        super().__init__()
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_bytes: bytes | None = None
        self._image_shape: tuple[int, int] | None = None
        self._preview_scale_x = 1.0
        self._preview_scale_y = 1.0
        self._crop_rect: QRectF | None = None
        self._select_start: QPointF | None = None
        self._select_current: QPointF | None = None
        self._zoom_level = 0
        self._zoom_factor = 1.0
        self._space_pan_active = False
        self._panning = False

    def set_image(
        self,
        image_rgb: np.ndarray,
        preview_scale_x: float = 1.0,
        preview_scale_y: float = 1.0,
        preserve_view: bool = False,
    ) -> None:
        transform = self.transform()
        h_scroll = self.horizontalScrollBar().value()
        v_scroll = self.verticalScrollBar().value()
        zoom_level = self._zoom_level
        zoom_factor = self._zoom_factor
        contiguous = np.ascontiguousarray(image_rgb)
        self._image_bytes = contiguous.tobytes()
        h, w, _ = contiguous.shape
        self._preview_scale_x = max(float(preview_scale_x), 1e-9)
        self._preview_scale_y = max(float(preview_scale_y), 1e-9)
        qimage = QImage(self._image_bytes, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.scene().clear()
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self._image_shape = (h, w)
        self.scene().setSceneRect(0, 0, w, h)
        if preserve_view and zoom_level > 0:
            self.setTransform(transform)
            self.horizontalScrollBar().setValue(h_scroll)
            self.verticalScrollBar().setValue(v_scroll)
            self._zoom_level = zoom_level
            self._zoom_factor = zoom_factor
        else:
            self.fit_to_window()

    def clear_image(self) -> None:
        self.scene().clear()
        self._pixmap_item = None
        self._image_bytes = None
        self._image_shape = None
        self._crop_rect = None
        self._select_start = None
        self._select_current = None
        self.resetTransform()

    def set_crop_rect(self, x: int, y: int, width: int, height: int) -> None:
        self._crop_rect = QRectF(
            float(x) / self._preview_scale_x,
            float(y) / self._preview_scale_y,
            float(width) / self._preview_scale_x,
            float(height) / self._preview_scale_y,
        )
        self.viewport().update()

    def fit_to_window(self) -> None:
        if self._image_shape is None:
            return
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = 0
        self._zoom_factor = 1.0

    def zoom_in(self) -> None:
        self._apply_zoom(1)

    def zoom_out(self) -> None:
        self._apply_zoom(-1)

    def drawForeground(self, painter: QPainter, rect) -> None:
        super().drawForeground(painter, rect)
        crop_rect = self._current_rect()
        if crop_rect is None:
            return
        view_scale = max(abs(painter.transform().m11()), 1e-6)
        painter.setPen(QPen(QColor(255, 230, 0), 3.0 / view_scale))
        painter.setBrush(QBrush(QColor(255, 230, 0, 35)))
        painter.drawRect(crop_rect)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if (
            event.button() == Qt.MouseButton.MiddleButton
            or event.button() == Qt.MouseButton.RightButton
            or (event.button() == Qt.MouseButton.LeftButton and self._space_pan_active)
        ):
            self._panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._image_shape is not None:
            self._select_start = self.mapToScene(event.position().toPoint())
            self._select_current = self._select_start
            self.viewport().update()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._select_start is not None:
            self._select_current = self.mapToScene(event.position().toPoint())
            self.viewport().update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            super().mouseReleaseEvent(event)
            self._panning = False
            if not self._space_pan_active:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._select_start is not None:
            self._select_current = self.mapToScene(event.position().toPoint())
            crop_rect = self._current_rect()
            self._select_start = None
            self._select_current = None
            if crop_rect is not None and crop_rect.width() >= 2 and crop_rect.height() >= 2:
                x = int(round(crop_rect.x()))
                y = int(round(crop_rect.y()))
                width = int(round(crop_rect.width()))
                height = int(round(crop_rect.height()))
                self._crop_rect = QRectF(x, y, width, height)
                self.crop_changed.emit(
                    int(round(x * self._preview_scale_x)),
                    int(round(y * self._preview_scale_y)),
                    int(round(width * self._preview_scale_x)),
                    int(round(height * self._preview_scale_y)),
                )
            self.viewport().update()
            return
        super().mouseReleaseEvent(event)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._apply_zoom(1)
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self._apply_zoom(-1)
            event.accept()
            return
        if event.key() == Qt.Key.Key_0:
            self.fit_to_window()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan_active = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan_active = False
            if not self._panning:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._image_shape is None:
            return
        if event.angleDelta().y() > 0:
            self._apply_zoom(1)
        elif event.angleDelta().y() < 0:
            self._apply_zoom(-1)

    def _apply_zoom(self, direction: int) -> None:
        if self._image_shape is None:
            return
        if direction > 0 and self._zoom_level < 20:
            self._zoom_level = min(20, self._zoom_level + 1)
            factor = 1.25
        elif direction < 0 and self._zoom_level > 0:
            self._zoom_level = max(0, self._zoom_level - 1)
            factor = 0.8
        else:
            return
        if self._zoom_level == 0:
            self.fit_to_window()
            return
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scale(factor, factor)
        self._zoom_factor *= factor

    def _current_rect(self) -> QRectF | None:
        if self._select_start is not None and self._select_current is not None:
            rect = QRectF(self._select_start, self._select_current).normalized()
        else:
            rect = self._crop_rect
        if rect is None or self._image_shape is None:
            return rect
        h, w = self._image_shape
        x = max(0.0, min(rect.x(), float(w - 1)))
        y = max(0.0, min(rect.y(), float(h - 1)))
        right = max(x + 1.0, min(rect.right(), float(w)))
        bottom = max(y + 1.0, min(rect.bottom(), float(h)))
        return QRectF(x, y, right - x, bottom - y)


class ZoomImageView(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._image_bytes: bytes | None = None
        self._image_shape: tuple[int, int] | None = None
        self._zoom_level = 0
        self._zoom_factor = 1.0
        self._space_pan_active = False
        self._panning = False

    def set_array(self, image: np.ndarray | None) -> None:
        self.scene().clear()
        if image is None:
            self._image_bytes = None
            self._image_shape = None
            return
        arr = np.ascontiguousarray(display_image(image))
        h, w, _ = arr.shape
        self._image_bytes = arr.tobytes()
        qimage = QImage(self._image_bytes, w, h, w * 3, QImage.Format.Format_RGB888)
        self.scene().addPixmap(QPixmap.fromImage(qimage))
        self._image_shape = (h, w)
        self.scene().setSceneRect(0, 0, w, h)
        self.fit_to_window()

    def fit_to_window(self) -> None:
        if self._image_shape is None:
            return
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = 0
        self._zoom_factor = 1.0

    def zoom_in(self) -> None:
        self._apply_zoom(1)

    def zoom_out(self) -> None:
        self._apply_zoom(-1)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if (
            event.button() == Qt.MouseButton.MiddleButton
            or event.button() == Qt.MouseButton.RightButton
            or (event.button() == Qt.MouseButton.LeftButton and self._space_pan_active)
        ):
            self._panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            super().mouseReleaseEvent(event)
            self._panning = False
            if not self._space_pan_active:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
            return
        super().mouseReleaseEvent(event)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.text() in ("+", "="):
            self.zoom_in()
            event.accept()
            return
        if event.text() in ("-", "_"):
            self.zoom_out()
            event.accept()
            return
        if event.text() == "0":
            self.fit_to_window()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan_active = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan_active = False
            if not self._panning:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._image_shape is None:
            return
        if event.angleDelta().y() > 0:
            self.zoom_in()
        elif event.angleDelta().y() < 0:
            self.zoom_out()

    def _apply_zoom(self, direction: int) -> None:
        if self._image_shape is None:
            return
        if direction > 0 and self._zoom_level < 20:
            self._zoom_level = min(20, self._zoom_level + 1)
            factor = 1.25
        elif direction < 0 and self._zoom_level > 0:
            self._zoom_level = max(0, self._zoom_level - 1)
            factor = 0.8
        else:
            return
        if self._zoom_level == 0:
            self.fit_to_window()
            return
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scale(factor, factor)
        self._zoom_factor *= factor


class TraceEventsWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object, float)
    failed = Signal(str)

    def __init__(self, preprocess: dict, hough: dict, params: AutoPipelineParams) -> None:
        super().__init__()
        self._preprocess = preprocess
        self._hough = hough
        self._params = params

    @Slot()
    def run(self) -> None:
        started = time.perf_counter()
        try:
            total_seeds = len(self._hough.get("seeds", []))
            self.progress.emit(35, f"Seed events 0/{total_seeds}")

            def report_seed_progress(done: int, total: int) -> None:
                if total <= 0:
                    self.progress.emit(85, "Seed events complete")
                    return
                percent = 35 + int(round(50.0 * done / total))
                self.progress.emit(percent, f"Seed events {done}/{total}")

            result = trace_hough_events(
                self._preprocess,
                self._hough,
                self._params,
                progress_callback=report_seed_progress,
            )
            self.progress.emit(90, "Rendering overlay...")
            result = dict(result)
            result["overlay"] = downsample_rgb_for_view(result["overlay"])
        except Exception as exc:  # pragma: no cover - UI safety path
            self.failed.emit(str(exc))
            return
        self.finished.emit(result, time.perf_counter() - started)


class AutoDetectionPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._project_files: dict[str, str] = {}
        self._image_path: str | None = None
        self._summary: dict[str, float] | None = None
        self._region: dict | None = None
        self._preprocess: dict | None = None
        self._preprocess_preview: dict | None = None
        self._hough: dict | None = None
        self._mask: dict | None = None
        self._trace: dict | None = None
        self._boundary_cut: dict | None = None
        self._trace_thread: QThread | None = None
        self._trace_worker: TraceEventsWorker | None = None
        self._range_controls_updating = False
        self._display_base_range: tuple[float, float] = (0.0, 1.0)
        self.alignment_analysis_panel = AlignmentAnalysisPanel()

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Load a BLN image in Project Files, choose a region, then tune preprocessing.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #374151;")
        self.status_label.setToolTip("Shows the current auto-detection workflow status and any processing messages.")
        layout.addWidget(self.status_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self._build_crop_tab()
        self._build_display_tab()
        self._build_hough_tab()
        self._build_boundary_cut_tab()
        self.tabs.addTab(self.alignment_analysis_panel, "Alignment")
        self._load_parameter_defaults()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
    def _build_crop_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setMaximumWidth(380)
        sidebar = QWidget()
        sidebar_scroll.setWidget(sidebar)
        side_layout = QVBoxLayout(sidebar)
        self.dimension_label = QLabel("Image: not loaded")
        self.dimension_label.setWordWrap(True)
        self.dimension_label.setToolTip("Original full-resolution image dimensions in pixels.")
        side_layout.addWidget(self.dimension_label)

        crop_box = QGroupBox("Crop")
        crop_box.setToolTip("Select the full image or a rectangular crop. Crop coordinates are stored in original image pixels.")
        crop_layout = QVBoxLayout(crop_box)
        self.use_full_image = QCheckBox("Use full image")
        self.use_full_image.setToolTip("Process the entire image instead of a selected crop. Full-image processing can be slower.")
        self.use_full_image.toggled.connect(self._update_crop_enabled)
        self.use_full_image.toggled.connect(self._refresh_from_region_change)
        crop_layout.addWidget(self.use_full_image)

        form = QFormLayout()
        self.crop_x = self._spin(0, 1_000_000, 1, 0)
        self.crop_y = self._spin(0, 1_000_000, 1, 0)
        self.crop_width = self._spin(1, 1_000_000, 10, 500)
        self.crop_height = self._spin(1, 1_000_000, 10, 500)
        self.crop_x.setReadOnly(True)
        self.crop_y.setReadOnly(True)
        self.crop_x.setEnabled(False)
        self.crop_y.setEnabled(False)
        self.crop_x.setToolTip("Left coordinate of the selected crop in original image pixels. Set by dragging on the image.")
        self.crop_y.setToolTip("Top coordinate of the selected crop in original image pixels. Set by dragging on the image.")
        self.crop_width.setToolTip("Width of the selected crop in original image pixels. You can edit this after selecting a crop.")
        self.crop_height.setToolTip("Height of the selected crop in original image pixels. You can edit this after selecting a crop.")
        form.addRow("Crop x", self.crop_x)
        form.addRow("Crop y", self.crop_y)
        form.addRow("Width", self.crop_width)
        form.addRow("Height", self.crop_height)
        crop_layout.addLayout(form)

        side_layout.addWidget(crop_box)

        crop_preview_label = QLabel("Selected Crop Preview")
        crop_preview_label.setToolTip("Downsampled preview of the selected crop using the current brightness/contrast settings.")
        crop_preview_label.setStyleSheet("font-weight: 700;")
        side_layout.addWidget(crop_preview_label)
        self.crop_preview = ZoomImageView()
        self.crop_preview.setMinimumSize(320, 260)
        self.crop_preview.setToolTip("Preview of the selected crop. Use mouse wheel to zoom, 0 to fit, and space-drag to pan.")
        side_layout.addWidget(self.crop_preview)
        side_layout.addStretch(1)

        self.crop_canvas = CropCanvas()
        self.crop_canvas.setMinimumSize(620, 420)
        self.crop_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.crop_canvas.crop_changed.connect(self._set_crop_from_canvas)
        right_side = QWidget()
        right_layout = QVBoxLayout(right_side)
        right_layout.setContentsMargins(0, 0, 0, 0)
        original_label = QLabel("Original Image")
        original_label.setToolTip("Downsampled display preview of the original image. Crop coordinates still map to the full-resolution image.")
        original_label.setStyleSheet("font-weight: 700;")
        right_layout.addWidget(original_label)
        right_layout.addWidget(self.crop_canvas, 1)
        self.crop_canvas.setToolTip("Drag to select a crop. Use =/+ and - to zoom, 0 to fit, mouse wheel to zoom, and space-drag to pan.")
        layout.addWidget(sidebar_scroll)
        layout.addWidget(right_side, 1)

        for widget in (self.crop_x, self.crop_y, self.crop_width, self.crop_height):
            widget.valueChanged.connect(self._crop_spin_changed)
        self.tabs.addTab(tab, "Crop")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(tab),
            "Select the full image or a crop. Display and preprocessing controls are in Common Steps.",
        )

    def _build_display_controls(self) -> QGroupBox:
        display_box = QGroupBox("Brightness / Contrast")
        display_box.setToolTip("Controls how the BLN values are mapped into the 0-255 image used by later preprocessing steps.")
        display_layout = QVBoxLayout(display_box)
        self.display_stats_label = QLabel("Raw range: -")
        self.display_stats_label.setWordWrap(True)
        self.display_stats_label.setToolTip("Raw BLN value summary for the loaded image.")
        display_layout.addWidget(self.display_stats_label)

        self.display_brightness_label = QLabel("Brightness: +0")
        self.display_brightness_label.setToolTip("Positive values brighten the mapped image; negative values darken it.")
        self.display_brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.display_brightness_slider.setRange(-200, 200)
        self.display_brightness_slider.setValue(0)
        self.display_brightness_slider.setSingleStep(1)
        self.display_brightness_slider.setPageStep(20)
        self.display_brightness_slider.setToolTip("Positive values brighten the mapped image; negative values darken it.")
        display_layout.addWidget(self.display_brightness_label)
        brightness_controls = QHBoxLayout()
        brightness_down = QPushButton("-")
        brightness_down.setToolTip("Decrease brightness by 0.5.")
        brightness_down.setAutoRepeat(True)
        brightness_down.setAutoRepeatInterval(60)
        brightness_down.clicked.connect(lambda checked=False: self._nudge_display_adjustment(self.display_brightness_slider, -0.5))
        brightness_up = QPushButton("+")
        brightness_up.setToolTip("Increase brightness by 0.5.")
        brightness_up.setAutoRepeat(True)
        brightness_up.setAutoRepeatInterval(60)
        brightness_up.clicked.connect(lambda checked=False: self._nudge_display_adjustment(self.display_brightness_slider, 0.5))
        brightness_controls.addWidget(brightness_down)
        brightness_controls.addWidget(self.display_brightness_slider, 1)
        brightness_controls.addWidget(brightness_up)
        display_layout.addLayout(brightness_controls)

        self.display_contrast_label = QLabel("Contrast: +0")
        self.display_contrast_label.setToolTip("Positive values increase contrast; negative values reduce contrast.")
        self.display_contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.display_contrast_slider.setRange(-200, 200)
        self.display_contrast_slider.setValue(0)
        self.display_contrast_slider.setSingleStep(1)
        self.display_contrast_slider.setPageStep(20)
        self.display_contrast_slider.setToolTip("Positive values increase contrast; negative values reduce contrast.")
        display_layout.addWidget(self.display_contrast_label)
        contrast_controls = QHBoxLayout()
        contrast_down = QPushButton("-")
        contrast_down.setToolTip("Decrease contrast by 0.5.")
        contrast_down.setAutoRepeat(True)
        contrast_down.setAutoRepeatInterval(60)
        contrast_down.clicked.connect(lambda checked=False: self._nudge_display_adjustment(self.display_contrast_slider, -0.5))
        contrast_up = QPushButton("+")
        contrast_up.setToolTip("Increase contrast by 0.5.")
        contrast_up.setAutoRepeat(True)
        contrast_up.setAutoRepeatInterval(60)
        contrast_up.clicked.connect(lambda checked=False: self._nudge_display_adjustment(self.display_contrast_slider, 0.5))
        contrast_controls.addWidget(contrast_down)
        contrast_controls.addWidget(self.display_contrast_slider, 1)
        contrast_controls.addWidget(contrast_up)
        display_layout.addLayout(contrast_controls)

        self.display_range_label = QLabel("Display window: -")
        self.display_range_label.setWordWrap(True)
        self.display_range_label.setToolTip("Computed raw-value window. Values below this map to black; above this map to white.")
        display_layout.addWidget(self.display_range_label)
        default_button = QPushButton("Set As Default")
        default_button.setToolTip("Save current brightness/contrast controls as the default for future app sessions.")
        default_button.clicked.connect(self._save_parameter_defaults)
        display_layout.addWidget(default_button)

        self.display_brightness_slider.valueChanged.connect(self._display_adjustment_changed)
        self.display_contrast_slider.valueChanged.connect(self._display_adjustment_changed)
        return display_box

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if (
            event.type() == QEvent.Type.KeyPress
            and self.tabs.currentWidget() is not None
            and self.tabs.tabText(self.tabs.currentIndex()) in {"Crop", "Common Steps", "Detection", "Boundary Cuts"}
            and self.isVisible()
        ):
            active_tab = self.tabs.tabText(self.tabs.currentIndex())
            if active_tab == "Crop":
                target = self.crop_preview if self.crop_preview.hasFocus() else self.crop_canvas
            elif active_tab == "Detection":
                if hasattr(self, "detection_tabs") and self.detection_tabs.tabText(self.detection_tabs.currentIndex()).startswith("Hough"):
                    target = self.trace_preview if self.hough_result_tabs.currentIndex() == 1 else self.hough_preview
                else:
                    target = self.mask_preview
            elif active_tab == "Boundary Cuts":
                target = self.boundary_cut_preview
            else:
                target = self.display_preview
            text = event.text()
            if text in ("+", "="):
                target.zoom_in()
                return True
            if text in ("-", "_"):
                target.zoom_out()
                return True
            if text == "0":
                target.fit_to_window()
                return True
        return super().eventFilter(watched, event)

    def _build_display_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)

        controls = QGroupBox("Common Steps")
        controls.setMaximumWidth(340)
        controls.setToolTip("Shared preprocessing for Hough and Mask detection: brightness/contrast, CLAHE, threshold, and clean/close.")
        controls.setStyleSheet(
            """
            QGroupBox {
                font-weight: 700;
            }
            QGroupBox::title {
                font-size: 15px;
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QPushButton[stageButton="true"] {
                text-align: left;
                padding: 8px 10px;
                border: 1px solid #b8c0cc;
                border-radius: 6px;
                background: #f6f7f9;
                color: #1f2328;
                font-weight: 600;
            }
            QPushButton[stageButton="true"]:checked {
                background: #dbeafe;
                border: 2px solid #2563eb;
                color: #0f172a;
            }
            QPushButton[stageButton="true"]:hover {
                background: #edf2ff;
            }
            """
        )
        control_layout = QVBoxLayout(controls)
        self.display_original_button = QPushButton("Brightness / Contrast")
        self.display_clahe_button = QPushButton("CLAHE")
        self.display_ridge_button = QPushButton("Threshold")
        self.display_clean_button = QPushButton("Clean / Close")
        self.display_original_button.setToolTip("Shows the selected crop after display range mapping.")
        self.display_clahe_button.setToolTip("Applies CLAHE to the original selected crop.")
        self.display_ridge_button.setToolTip("Enhances line-like features and thresholds the result into a candidate mask.")
        self.display_clean_button.setToolTip("Removes small candidate objects and applies closing to bridge small gaps.")
        for button in (
            self.display_original_button,
            self.display_clahe_button,
            self.display_ridge_button,
            self.display_clean_button,
        ):
            button.setCheckable(True)
            button.setProperty("stageButton", True)
            button.setMinimumHeight(36)
            control_layout.addWidget(button)
        self.display_original_button.clicked.connect(lambda: self._set_display_stage("original"))
        self.display_clahe_button.clicked.connect(lambda: self._set_display_stage("clahe"))
        self.display_ridge_button.clicked.connect(lambda: self._set_display_stage("ridge"))
        self.display_clean_button.clicked.connect(lambda: self._set_display_stage("clean"))
        self._display_stage = "original"
        control_layout.addSpacing(14)

        auto_process_button = QPushButton("Auto Process Common Steps")
        auto_process_button.setToolTip(
            "Run brightness/contrast, CLAHE, thresholding, and clean/close using the current/default parameters."
        )
        auto_process_button.clicked.connect(self._auto_process_common_steps)
        control_layout.addWidget(auto_process_button)

        self.option_stack = QStackedWidget()
        self.option_stack.setMinimumHeight(280)
        self.option_stack.setToolTip("Parameters for the selected common processing step.")
        self.brightness_options = self._build_display_controls()
        self.option_stack.addWidget(self.brightness_options)

        self.clahe_options = QGroupBox("CLAHE Options")
        self.clahe_options.setToolTip("Contrast-limited adaptive histogram equalization. It improves local contrast before ridge detection.")
        form = QFormLayout(self.clahe_options)
        self.clahe_clip = self._double_spin(0.001, 1.0, 0.001, 0.300, decimals=3)
        self.clahe_clip.setToolTip("Limits local contrast amplification in CLAHE.")
        self.clahe_clip_slider = QSlider(Qt.Orientation.Horizontal)
        self.clahe_clip_slider.setRange(1, 1000)
        self.clahe_clip_slider.setValue(300)
        self.clahe_clip_slider.setToolTip("Limits local contrast amplification in CLAHE.")
        form.addRow("Clip limit", self.clahe_clip)
        form.addRow("Clip slider", self.clahe_clip_slider)
        clahe_default = QPushButton("Set As Default")
        clahe_default.setToolTip("Save current CLAHE parameters as the default.")
        clahe_default.clicked.connect(self._save_parameter_defaults)
        form.addRow("", clahe_default)
        self.option_stack.addWidget(self.clahe_options)

        self.ridge_options = QGroupBox("Threshold Options")
        self.ridge_options.setToolTip("Ridge filtering and thresholding create the candidate mask.")
        form = QFormLayout(self.ridge_options)
        self.ridge_sigma_max = self._spin(1, 12, 1, 3)
        self.ridge_sigma_max.setToolTip("Largest ridge-filter scale in pixels.")
        self.ridge_percentile = self._double_spin(80.0, 99.9, 0.1, 80.0, decimals=1)
        self.ridge_percentile.setToolTip("Percentile cutoff for the ridge response. Higher values keep fewer, stronger ridge pixels.")
        self.threshold_multiplier = self._double_spin(0.1, 2.0, 0.05, 0.80, decimals=2)
        self.threshold_multiplier.setToolTip("Multiplier applied to the Otsu ridge threshold. Lower values include weaker ridges; higher values are stricter.")
        form.addRow("Ridge sigma max", self.ridge_sigma_max)
        form.addRow("Ridge percentile", self.ridge_percentile)
        form.addRow("Otsu multiplier", self.threshold_multiplier)
        threshold_default = QPushButton("Set As Default")
        threshold_default.setToolTip("Save current threshold parameters as the default.")
        threshold_default.clicked.connect(self._save_parameter_defaults)
        form.addRow("", threshold_default)
        self.option_stack.addWidget(self.ridge_options)

        self.clean_options = QGroupBox("Clean / Close Options")
        self.clean_options.setToolTip("Thresholds and cleans the ridge response to remove small noise and bridge small gaps.")
        form = QFormLayout(self.clean_options)
        self.min_object_size = self._spin(1, 5000, 5, 40)
        self.min_object_size.setToolTip("Removes connected ridge-mask objects smaller than this many pixels.")
        self.closing_radius = self._spin(0, 20, 1, 1)
        self.closing_radius.setToolTip("Morphological closing radius. Higher values bridge small gaps but can merge nearby ridges.")
        form.addRow("Min object pixels", self.min_object_size)
        form.addRow("Closing radius", self.closing_radius)
        clean_default = QPushButton("Set As Default")
        clean_default.setToolTip("Save current clean/close parameters as the default.")
        clean_default.clicked.connect(self._save_parameter_defaults)
        form.addRow("", clean_default)
        self.option_stack.addWidget(self.clean_options)

        control_layout.addWidget(self.option_stack)

        self.preprocess_stats = QLabel("No preprocessing result yet.")
        self.preprocess_stats.setWordWrap(True)
        self.preprocess_stats.setToolTip("Reserved preprocessing status area. Detailed pixel counts are hidden to keep the interface compact.")
        self.preprocess_stats.hide()
        control_layout.addWidget(self.preprocess_stats)
        control_layout.addStretch(1)

        self.display_preview = ZoomImageView()
        self.display_preview.setMinimumSize(520, 420)
        self.display_preview.setToolTip("Preview of the selected preprocessing stage. Use =/+ and - to zoom, 0 to fit, mouse wheel to zoom, and space-drag to pan.")
        layout.addWidget(controls)
        layout.addWidget(self.display_preview, 1)

        self.clahe_clip.valueChanged.connect(self._clahe_spin_changed)
        self.clahe_clip_slider.valueChanged.connect(self._clahe_slider_changed)
        for widget in (
            self.ridge_sigma_max,
            self.ridge_percentile,
            self.threshold_multiplier,
            self.min_object_size,
            self.closing_radius,
        ):
            widget.valueChanged.connect(self._refresh_display)
        self._update_stage_options()
        self.tabs.addTab(tab, "Common Steps")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(tab),
            "Shared preprocessing stages used by Hough and Mask detection.",
        )

    def _build_hough_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.detection_tabs = QTabWidget()
        layout.addWidget(self.detection_tabs, 1)

        hough_tab = QWidget()
        hough_layout = QHBoxLayout(hough_tab)
        hough_controls = QWidget()
        hough_controls.setMaximumWidth(380)
        hough_controls_layout = QVBoxLayout(hough_controls)
        hough_controls_layout.setContentsMargins(8, 8, 8, 8)

        self.hough_detection_group = QGroupBox("Step 1: Seed Detection")
        seed_layout = QVBoxLayout(self.hough_detection_group)
        params_box = QGroupBox("Hough Parameters")
        params_box.setToolTip("Parameters for probabilistic Hough line detection and Hough-based seed selection.")
        form = QFormLayout(params_box)
        self.hough_threshold = self._spin(1, 200, 1, 20)
        self.hough_threshold.setToolTip("Minimum accumulator votes needed to accept a Hough line segment. Higher values keep fewer, stronger lines.")
        self.hough_line_length = self._spin(1, 1000, 5, 30)
        self.hough_line_length.setToolTip("Minimum length in pixels for an accepted Hough line segment.")
        self.hough_line_gap = self._spin(0, 200, 1, 10)
        self.hough_line_gap.setToolTip("Maximum gap in pixels that can be bridged while forming one Hough line segment.")
        self.hough_seed_spacing = self._spin(1, 300, 1, 20)
        self.hough_seed_spacing.setToolTip("Minimum spacing in pixels between selected seed points. Larger values reduce nearby duplicate seeds.")
        self.hough_use_all_seeds = QCheckBox("Use all Hough seeds")
        self.hough_use_all_seeds.setChecked(True)
        self.hough_use_all_seeds.setToolTip("When enabled, all spaced Hough seed candidates are used. Disable to cap the seed count.")
        self.hough_use_all_seeds.toggled.connect(self._sync_max_seed_enabled)
        self.hough_max_seeds = self._spin(1, 10000, 10, 200)
        self.hough_max_seeds.setToolTip("Maximum selected Hough seeds to use when 'Use all Hough seeds' is disabled.")
        form.addRow("Hough threshold", self.hough_threshold)
        form.addRow("Line length", self.hough_line_length)
        form.addRow("Line gap", self.hough_line_gap)
        form.addRow("Seed spacing", self.hough_seed_spacing)
        form.addRow("", self.hough_use_all_seeds)
        form.addRow("Max seeds", self.hough_max_seeds)
        hough_default = QPushButton("Set As Default")
        hough_default.setToolTip("Save current Hough seed parameters as the default.")
        hough_default.clicked.connect(self._save_parameter_defaults)
        form.addRow("", hough_default)
        seed_layout.addWidget(params_box)

        run_button = QPushButton("Detect Hough Lines And Seeds")
        run_button.setToolTip("Runs full-resolution preprocessing, detects Hough line segments, then selects seed points from the strongest pixel on each line.")
        run_button.clicked.connect(self._run_hough)
        seed_layout.addWidget(run_button)

        self.hough_progress = QLabel("Ready.")
        self.hough_progress.setWordWrap(True)
        self.hough_progress.setToolTip("Shows whether Hough seed detection is idle, running, failed, or completed with elapsed time.")
        seed_layout.addWidget(self.hough_progress)
        self.hough_progress_bar = QProgressBar()
        self.hough_progress_bar.setRange(0, 1)
        self.hough_progress_bar.setValue(0)
        self.hough_progress_bar.setTextVisible(True)
        self.hough_progress_bar.setFormat("Ready")
        self.hough_progress_bar.setToolTip("Shows progress for Hough seed detection. An animated bar means processing is active.")
        seed_layout.addWidget(self.hough_progress_bar)

        self.hough_stats = QLabel("No Hough result yet.")
        self.hough_stats.setWordWrap(True)
        self.hough_stats.setToolTip("After processing, shows detected Hough line count, seed candidate count, and selected seed count.")
        seed_layout.addWidget(self.hough_stats)

        self.hough_fill_group = QGroupBox("Step 2: Fill Algorithm")
        self.hough_fill_group.setToolTip("Grow DIC events from Hough seeds and optionally merge compatible event fragments.")
        fill_layout = QVBoxLayout(self.hough_fill_group)
        fill_params_box = QGroupBox("Fill Parameters")
        fill_params_box.setToolTip("Parameters for Java-style event growing and event merging.")
        form = QFormLayout(fill_params_box)
        self.intensity_tolerance = self._spin(0, 255, 1, 75)
        self.intensity_tolerance.setToolTip("Allowed blue-channel intensity drop while growing from each seed. Larger values let events grow through weaker pixels.")
        self.bfl_tolerance = self._double_spin(0.1, 100.0, 0.1, 7.0, decimals=1)
        self.bfl_tolerance.setToolTip("Best-fit-line vertical tolerance used by the Java-style seed growth. Larger values allow more curved or scattered growth.")
        self.min_intensity = self._spin(0, 255, 1, 120)
        self.min_intensity.setToolTip("Minimum blue-channel intensity accepted for grown event pixels.")
        self.min_points = self._spin(1, 10000, 1, 10)
        self.min_points.setToolTip("Reject grown events smaller than this point count.")
        self.duplicate_overlap = self._double_spin(0.0, 1.0, 0.05, 0.50, decimals=2)
        self.duplicate_overlap.setToolTip("Merge events when their shared-pixel fraction reaches this value.")
        self.merge_distance = self._double_spin(0.0, 100.0, 0.5, 5.0, decimals=1)
        self.merge_distance.setToolTip("Merge nearby same-angle events when their closest pixels are within this distance.")
        self.merge_angle = self._double_spin(0.0, 90.0, 0.5, 12.0, decimals=1)
        self.merge_angle.setToolTip("Maximum PCA/SVD angle difference, in degrees, for distance-based merging.")
        form.addRow("Intensity difference tolerance", self.intensity_tolerance)
        form.addRow("Best fit line tolerance", self.bfl_tolerance)
        form.addRow("Minimum intensity", self.min_intensity)
        form.addRow("Minimum points", self.min_points)
        form.addRow("Duplicate overlap", self.duplicate_overlap)
        form.addRow("Merge distance", self.merge_distance)
        form.addRow("Merge angle", self.merge_angle)
        fill_default = QPushButton("Set As Default")
        fill_default.setToolTip("Save current Hough fill parameters as the default.")
        fill_default.clicked.connect(self._save_parameter_defaults)
        form.addRow("", fill_default)
        fill_layout.addWidget(fill_params_box)

        self.trace_button = QPushButton("Fill Events From Hough Seeds")
        self.trace_button.setToolTip("Grows events from the selected Hough seeds using the Java-style algorithm, then merges duplicate or nearby compatible events.")
        self.trace_button.clicked.connect(self._run_trace)
        fill_layout.addWidget(self.trace_button)

        self.trace_progress = QLabel("Ready.")
        self.trace_progress.setWordWrap(True)
        self.trace_progress.setToolTip("Shows whether event tracing is idle, running, failed, or completed with elapsed time.")
        fill_layout.addWidget(self.trace_progress)
        self.trace_progress_bar = QProgressBar()
        self.trace_progress_bar.setRange(0, 1)
        self.trace_progress_bar.setValue(0)
        self.trace_progress_bar.setTextVisible(True)
        self.trace_progress_bar.setFormat("Ready")
        self.trace_progress_bar.setToolTip("Shows progress for the Hough fill algorithm. An animated bar means processing is active.")
        fill_layout.addWidget(self.trace_progress_bar)

        self.trace_stats = QLabel("No filled events yet.")
        self.trace_stats.setWordWrap(True)
        self.trace_stats.setToolTip("After tracing, shows detected event count, skipped covered seeds, rejected seeds, merged groups, and event pixel count.")
        fill_layout.addWidget(self.trace_stats)

        self.hough_controls_stack = QStackedWidget()
        self.hough_controls_stack.addWidget(self.hough_detection_group)
        self.hough_controls_stack.addWidget(self.hough_fill_group)
        hough_controls_layout.addWidget(self.hough_controls_stack)
        hough_controls_layout.addStretch(1)

        hough_image_panel = QWidget()
        hough_image_layout = QVBoxLayout(hough_image_panel)
        hough_image_layout.setContentsMargins(0, 0, 0, 0)
        hough_image_label = QLabel("Hough Detection Results")
        hough_image_label.setToolTip("Switch between Hough seed detection and filled event overlays.")
        hough_image_label.setStyleSheet("font-weight: 700;")
        hough_image_layout.addWidget(hough_image_label)
        self.hough_result_tabs = QTabWidget()
        self.hough_preview = ZoomImageView()
        self.hough_preview.setMinimumSize(520, 420)
        self.hough_preview.setToolTip("Red pixels show detected Hough lines. Yellow points show selected seeds. Use =/+ and - to zoom, 0 to fit, or mouse wheel to zoom.")

        self.trace_preview = ZoomImageView()
        self.trace_preview.setMinimumSize(520, 420)
        self.trace_preview.setToolTip("Detected filled event pixels are shown as an overlay. Use =/+ and - to zoom, 0 to fit, or mouse wheel to zoom.")

        self.hough_result_tabs.addTab(self.hough_preview, "Step 1 Seeds")
        self.hough_result_tabs.addTab(self.trace_preview, "Step 2 Filled Events")
        self.hough_result_tabs.currentChanged.connect(self._sync_hough_step_controls)
        hough_image_layout.addWidget(self.hough_result_tabs, 1)

        hough_layout.addWidget(hough_controls)
        hough_layout.addWidget(hough_image_panel, 1)

        mask_tab = QWidget()
        mask_layout_outer = QHBoxLayout(mask_tab)
        mask_controls = QGroupBox("Mask Detection")
        mask_controls.setMaximumWidth(360)
        mask_controls.setToolTip("Uses Common Steps output directly. Connected mask components become finite-width events.")
        mask_layout = QVBoxLayout(mask_controls)
        description = QLabel(
            "Uses Common Steps output directly: thresholded, cleaned, and closed mask components become events."
        )
        description.setWordWrap(True)
        description.setToolTip("This branch does not use Hough seeds or seed growing. It keeps finite-width mask bands as event pixels.")
        mask_layout.addWidget(description)

        mask_run_button = QPushButton("Detect Mask Events")
        mask_run_button.setToolTip("Runs full-resolution Common Steps and labels connected mask components as events.")
        mask_run_button.clicked.connect(self._run_mask_events)
        mask_layout.addWidget(mask_run_button)

        self.mask_progress = QLabel("Ready.")
        self.mask_progress.setWordWrap(True)
        self.mask_progress.setToolTip("Shows whether mask-event detection is idle, running, failed, or completed with elapsed time.")
        mask_layout.addWidget(self.mask_progress)
        self.mask_progress_bar = QProgressBar()
        self.mask_progress_bar.setRange(0, 1)
        self.mask_progress_bar.setValue(0)
        self.mask_progress_bar.setTextVisible(True)
        self.mask_progress_bar.setFormat("Ready")
        self.mask_progress_bar.setToolTip("Shows progress for mask event detection. An animated bar means processing is active.")
        mask_layout.addWidget(self.mask_progress_bar)

        self.mask_stats = QLabel("No mask-event result yet.")
        self.mask_stats.setWordWrap(True)
        self.mask_stats.setToolTip("After processing, shows connected component count, accepted event count, and event pixels.")
        mask_layout.addWidget(self.mask_stats)
        mask_layout.addStretch(1)
        mask_image_panel = QWidget()
        image_layout = QVBoxLayout(mask_image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        mask_image_label = QLabel("Mask Events")
        mask_image_label.setToolTip("Overlay image for mask connected component detection.")
        mask_image_label.setStyleSheet("font-weight: 700;")
        image_layout.addWidget(mask_image_label)
        self.mask_preview = ZoomImageView()
        self.mask_preview.setMinimumSize(520, 420)
        self.mask_preview.setToolTip("Red pixels show connected mask events. Use =/+ and - to zoom, 0 to fit, or mouse wheel to zoom.")
        image_layout.addWidget(self.mask_preview, 1)

        mask_layout_outer.addWidget(mask_controls)
        mask_layout_outer.addWidget(mask_image_panel, 1)

        self.detection_tabs.addTab(mask_tab, "Mask Detection")
        self.detection_tabs.addTab(hough_tab, "Hough Detection")
        self.detection_tabs.setTabToolTip(
            self.detection_tabs.indexOf(hough_tab),
            "Step 1 detects Hough seeds; Step 2 fills/grows events from those seeds.",
        )
        self.detection_tabs.setTabToolTip(
            self.detection_tabs.indexOf(mask_tab),
            "Treat cleaned mask components as detected events directly.",
        )
        self.tabs.addTab(tab, "Detection")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(tab),
            "Run either Hough-based detection or direct mask-component detection.",
        )
        self._sync_max_seed_enabled()

    def _build_boundary_cut_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)

        controls = QGroupBox("Boundary Cut Events")
        controls.setMaximumWidth(380)
        controls.setToolTip("Cut detected events wherever they cross dark EBSD grain-boundary pixels.")
        control_layout = QVBoxLayout(controls)

        self.boundary_path_label = QLabel("Boundary image: not loaded")
        self.boundary_path_label.setWordWrap(True)
        self.boundary_path_label.setToolTip("EBSD boundary image path comes from the Project Files tab.")
        control_layout.addWidget(self.boundary_path_label)

        form = QFormLayout()
        self.boundary_event_source = QComboBox()
        self.boundary_event_source.addItems(["Mask events", "Hough filled events"])
        self.boundary_event_source.setToolTip("Choose which current detected event set should be cut by the EBSD boundary image.")
        self.boundary_black_threshold = self._double_spin(0.0, 1.0, 0.01, 0.60, decimals=2)
        self.boundary_black_threshold.setToolTip("Boundary image pixels darker than this normalized value are treated as grain-boundary pixels.")
        self.boundary_dilation_radius = self._spin(0, 20, 1, 0)
        self.boundary_dilation_radius.setToolTip("Expands boundary pixels before cutting. Higher values cut more aggressively near boundaries.")
        self.boundary_min_segment_pixels = self._spin(1, 10000, 1, 40)
        self.boundary_min_segment_pixels.setToolTip("Discard cut event fragments smaller than this many pixels.")
        self.boundary_connectivity = QComboBox()
        self.boundary_connectivity.addItems(["1 - edge connected", "2 - edge/corner connected"])
        self.boundary_connectivity.setCurrentIndex(1)
        self.boundary_connectivity.setToolTip("Connectivity used to label event fragments after boundary pixels are removed.")
        form.addRow("Event source", self.boundary_event_source)
        form.addRow("Black threshold", self.boundary_black_threshold)
        form.addRow("Boundary dilation", self.boundary_dilation_radius)
        form.addRow("Min segment pixels", self.boundary_min_segment_pixels)
        form.addRow("Connectivity", self.boundary_connectivity)
        boundary_default = QPushButton("Set As Default")
        boundary_default.setToolTip("Save current boundary-cut parameters as the default.")
        boundary_default.clicked.connect(self._save_parameter_defaults)
        form.addRow("", boundary_default)
        control_layout.addLayout(form)

        run_button = QPushButton("Run Boundary Cut")
        run_button.setToolTip("Cut the selected current event set by the loaded EBSD boundary image.")
        run_button.clicked.connect(self._run_boundary_cut)
        control_layout.addWidget(run_button)

        self.boundary_cut_progress = QLabel("Ready.")
        self.boundary_cut_progress.setWordWrap(True)
        self.boundary_cut_progress.setToolTip("Shows whether boundary cutting is idle, running, failed, or completed.")
        self.boundary_cut_progress.setStyleSheet(
            "padding: 8px; border-radius: 6px; background: #f9fafb; color: #374151;"
        )
        control_layout.addWidget(self.boundary_cut_progress)
        self.boundary_cut_progress_bar = QProgressBar()
        self.boundary_cut_progress_bar.setRange(0, 1)
        self.boundary_cut_progress_bar.setValue(0)
        self.boundary_cut_progress_bar.setTextVisible(True)
        self.boundary_cut_progress_bar.setFormat("Ready")
        control_layout.addWidget(self.boundary_cut_progress_bar)

        self.boundary_cut_stats = QLabel("No boundary-cut result yet.")
        self.boundary_cut_stats.setWordWrap(True)
        self.boundary_cut_stats.setToolTip("After cutting, shows event counts, removed pixels, splits, and discarded small fragments.")
        control_layout.addWidget(self.boundary_cut_stats)
        legend = QLabel("Overlay: red = kept event pixels, blue = EBSD boundary, green = pixels removed by boundary cut.")
        legend.setWordWrap(True)
        legend.setToolTip("Colors used in the boundary-cut result overlay.")
        control_layout.addWidget(legend)
        control_layout.addStretch(1)

        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_label = QLabel("Boundary Cut Overlay")
        image_label.setStyleSheet("font-weight: 700;")
        image_layout.addWidget(image_label)
        self.boundary_cut_preview = ZoomImageView()
        self.boundary_cut_preview.setMinimumSize(520, 420)
        self.boundary_cut_preview.setToolTip("Boundary-cut overlay. Use =/+ and - to zoom, 0 to fit, mouse wheel to zoom, and space-drag to pan.")
        image_layout.addWidget(self.boundary_cut_preview, 1)

        layout.addWidget(controls)
        layout.addWidget(image_panel, 1)
        self.tabs.addTab(tab, "Boundary Cuts")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(tab),
            "Cut current detected events using the EBSD boundary image from Project Files.",
        )

    def _scroll_tab(self) -> tuple[QScrollArea, QWidget]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        return scroll, content

    def set_project_files(self, files: dict[str, str]) -> None:
        self._project_files = dict(files)
        self.alignment_analysis_panel.set_project_files(self._project_files)
        self._refresh_boundary_file_summary()
        bln_path = self._project_files.get("bln_image", "").strip()
        if not bln_path:
            self._image_path = None
            self._summary = None
            self.dimension_label.setText("Image: not loaded")
            self._set_status_neutral("Load a BLN image in Project Files before using Auto Detection.")
            self._clear_downstream()
            self.crop_canvas.clear_image()
            self.crop_preview.set_array(None)
            return

        expanded = str(Path(bln_path).expanduser())
        if expanded == self._image_path:
            return
        if not Path(expanded).exists():
            self._set_status_error(f"Project Files BLN image not found: {expanded}")
            return
        self._load_image_metadata(expanded)

    def _set_boundary_cut_result(self, result: dict | None) -> None:
        self._boundary_cut = result
        self.alignment_analysis_panel.set_boundary_cut_result(result)

    def _refresh_boundary_file_summary(self) -> None:
        if not hasattr(self, "boundary_path_label"):
            return
        boundary_path = self._project_files.get("ebsd_boundary_image", "").strip()
        if not boundary_path:
            self.boundary_path_label.setText("Boundary image: not loaded")
            self.boundary_path_label.setStyleSheet("color: #374151;")
            return
        path = Path(boundary_path).expanduser()
        if not path.exists():
            self.boundary_path_label.setText(f"Boundary image not found:\n{path}")
            self.boundary_path_label.setStyleSheet("color: #b91c1c; font-weight: 700;")
            return
        try:
            width, height = image_size(str(path))
        except Exception:
            self.boundary_path_label.setText(f"Boundary image selected:\n{path.name}")
            self.boundary_path_label.setStyleSheet("color: #374151;")
            return
        self.boundary_path_label.setStyleSheet("color: #374151;")
        self.boundary_path_label.setText(f"Boundary image: {path.name}\nDimension: {width} x {height} pixels")

    def _load_image_metadata(self, path: str) -> None:
        with busy_cursor():
            self._set_status_neutral("Loading image...")
            try:
                width, height = image_size(path)
                self._summary = image_value_summary(path)
            except Exception as exc:  # pragma: no cover - UI safety path
                self._set_status_error(f"Could not load image: {exc}")
                return

        self._image_path = path
        self.dimension_label.setText(f"Original dimension: {width} x {height} pixels")
        self._set_status_neutral(f"Loaded image: {width} x {height}.")
        display_min, display_max = default_display_range(self._summary)
        self._display_base_range = (display_min, display_max)
        self._range_controls_updating = True
        self._apply_display_adjustment_defaults()
        self._range_controls_updating = False
        self._update_display_adjustment_labels()
        crop_size = min(500, width, height)
        self.crop_x.setMaximum(max(0, width - 1))
        self.crop_y.setMaximum(max(0, height - 1))
        self.crop_width.setMaximum(width)
        self.crop_height.setMaximum(height)
        self.crop_width.setValue(crop_size)
        self.crop_height.setValue(crop_size)
        self.crop_x.setValue(max(0, (width - crop_size) // 2))
        self.crop_y.setValue(max(0, (height - crop_size) // 2))
        self._clear_downstream()
        self._refresh_crop_canvas()
        self._refresh_crop_preview()
        self._run_region()

    def _params(self) -> AutoPipelineParams:
        if not self._image_path:
            raise ValueError("No BLN image is loaded.")
        display_min, display_max = self._current_display_range()
        return AutoPipelineParams(
            image_path=self._image_path,
            display_min=float(display_min),
            display_max=float(display_max),
            display_brightness=self._display_slider_value(self.display_brightness_slider),
            use_full_image=bool(self.use_full_image.isChecked()),
            crop_x=int(self.crop_x.value()),
            crop_y=int(self.crop_y.value()),
            crop_width=int(self.crop_width.value()),
            crop_height=int(self.crop_height.value()),
            clahe_clip_limit=float(self.clahe_clip.value()),
            ridge_sigma_max=int(self.ridge_sigma_max.value()),
            ridge_percentile=float(self.ridge_percentile.value()),
            threshold_multiplier=float(self.threshold_multiplier.value()),
            min_object_size=int(self.min_object_size.value()),
            closing_radius=int(self.closing_radius.value()),
            use_skeletonize=True,
            hough_threshold=int(self.hough_threshold.value()),
            hough_line_length=int(self.hough_line_length.value()),
            hough_line_gap=int(self.hough_line_gap.value()),
            hough_seed_spacing=int(self.hough_seed_spacing.value()),
            hough_max_seeds=int(self.hough_max_seeds.value()),
            hough_use_all_seeds=bool(self.hough_use_all_seeds.isChecked()),
            intensity_difference_tolerance=int(self.intensity_tolerance.value()),
            best_fit_line_tolerance=float(self.bfl_tolerance.value()),
            min_intensity=int(self.min_intensity.value()),
            min_points_threshold=int(self.min_points.value()),
            duplicate_overlap=float(self.duplicate_overlap.value()),
            merge_distance_tolerance=float(self.merge_distance.value()),
            merge_angle_tolerance=float(self.merge_angle.value()),
            connect_merged_event_gaps=False,
        )

    def _run_region(self) -> None:
        try:
            self._region = load_region(self._params())
        except Exception as exc:  # pragma: no cover - UI safety path
            self._set_status_error(f"Region load failed: {exc}")
            return
        origin = self._region["origin"]
        height, width = self._region["display_rgb"].shape[:2]
        self._set_status_neutral(f"Loaded region at x={origin[0]}, y={origin[1]}, size {width} x {height}.")
        self._preprocess = None
        self._preprocess_preview = None
        self._hough = None
        self._mask = None
        self._trace = None
        self._set_boundary_cut_result(None)
        self._show_display_stage()

    def _run_preprocessing(self) -> None:
        with busy_cursor():
            try:
                self._preprocess_preview = run_preprocessing_preview(self._params())
            except Exception as exc:  # pragma: no cover - UI safety path
                self._set_status_error(f"Preprocessing failed: {exc}")
                return
        self._set_status_neutral("Preprocessing preview updated.")
        self._preprocess = None
        self._hough = None
        self._mask = None
        self._trace = None
        self._set_boundary_cut_result(None)
        self._show_display_stage()

    def _run_hough(self, switch_to_seed_view: bool = True) -> bool:
        started = time.perf_counter()
        self._set_label_neutral(self.hough_progress, "Processing Hough lines and seeds...")
        self._set_progress_value(self.hough_progress_bar, 5, "Starting...")
        self.hough_stats.setText("")
        self._set_status_neutral("Processing Hough lines and seeds...")
        QApplication.processEvents()
        with busy_cursor():
            try:
                self._preprocess = run_preprocessing(self._params())
                self._set_progress_value(self.hough_progress_bar, 45, "Preprocessing done")
            except Exception as exc:  # pragma: no cover - UI safety path
                self._set_status_error(f"Full-resolution preprocessing failed: {exc}")
                self._set_label_error(self.hough_progress, "Processing failed during full-resolution preprocessing.")
                self._set_progress_failed(self.hough_progress_bar)
                return False
            try:
                self._set_progress_value(self.hough_progress_bar, 55, "Detecting seeds...")
                self._hough = run_hough_seed_detection(self._preprocess, self._params())
                self._set_progress_value(self.hough_progress_bar, 90, "Rendering overlay...")
            except Exception as exc:  # pragma: no cover - UI safety path
                self._set_status_error(f"Hough seed detection failed: {exc}")
                self._set_label_error(self.hough_progress, "Processing failed during Hough seed detection.")
                self._set_progress_failed(self.hough_progress_bar)
                return False
        elapsed = time.perf_counter() - started
        if self.hough_use_all_seeds.isChecked():
            self.hough_max_seeds.setMaximum(max(1, self._hough["candidate_seed_count"]))
            self.hough_max_seeds.setValue(max(1, self._hough["candidate_seed_count"]))
        self.hough_preview.set_array(self._hough["overlay"])
        if switch_to_seed_view:
            self.hough_result_tabs.setCurrentIndex(0)
        self.hough_stats.setText(
            f"Detected Hough lines: {self._hough['raw_count']:,}\n"
            f"Hough seed candidates: {self._hough['candidate_seed_count']:,}\n"
            f"Selected seeds: {len(self._hough['seeds']):,}"
        )
        self._set_label_neutral(self.hough_progress, f"Processed in {elapsed:.2f} seconds.")
        self._set_progress_complete(self.hough_progress_bar)
        self._set_status_neutral(f"Hough line and seed detection complete in {elapsed:.2f} seconds.")
        self._trace = None
        self._set_boundary_cut_result(None)
        return True

    def _run_mask_events(self) -> None:
        started = time.perf_counter()
        self._set_label_neutral(self.mask_progress, "Processing mask events...")
        self._set_progress_value(self.mask_progress_bar, 5, "Starting...")
        self.mask_stats.setText("")
        self._set_status_neutral("Processing mask events...")
        QApplication.processEvents()
        with busy_cursor():
            try:
                self._preprocess = run_preprocessing(self._params())
                self._set_progress_value(self.mask_progress_bar, 55, "Preprocessing done")
            except Exception as exc:  # pragma: no cover - UI safety path
                self._set_status_error(f"Full-resolution common preprocessing failed: {exc}")
                self._set_label_error(self.mask_progress, "Processing failed during full-resolution common preprocessing.")
                self._set_progress_failed(self.mask_progress_bar)
                return
            try:
                self._set_progress_value(self.mask_progress_bar, 70, "Labeling mask events...")
                self._mask = run_mask_event_detection(self._preprocess, self._params())
                self._set_progress_value(self.mask_progress_bar, 90, "Rendering overlay...")
            except Exception as exc:  # pragma: no cover - UI safety path
                self._set_status_error(f"Mask event detection failed: {exc}")
                self._set_label_error(self.mask_progress, "Processing failed during mask event detection.")
                self._set_progress_failed(self.mask_progress_bar)
                return
        elapsed = time.perf_counter() - started
        self.detection_tabs.setCurrentIndex(0)
        self.mask_preview.set_array(downsample_rgb_for_view(self._mask["overlay"]))
        total_pixels = sum(line.size for line in self._mask["accepted"])
        self.mask_stats.setText(
            f"Connected mask components: {self._mask['component_count']:,}\n"
            f"Accepted mask events: {self._mask['accepted_count']:,}\n"
            f"Event pixels: {total_pixels:,}"
        )
        self._set_label_neutral(self.mask_progress, f"Processed in {elapsed:.2f} seconds.")
        self._set_progress_complete(self.mask_progress_bar)
        self._set_status_neutral(f"Mask event detection complete in {elapsed:.2f} seconds.")
        self._set_boundary_cut_result(None)

    def _run_boundary_cut(self) -> None:
        boundary_path = self._project_files.get("ebsd_boundary_image", "").strip()
        if not boundary_path:
            self._set_notice_error(self.boundary_cut_progress, "Load an EBSD boundary image in Project Files first.")
            self._set_progress_failed(self.boundary_cut_progress_bar)
            return
        boundary_path = str(Path(boundary_path).expanduser())
        if not Path(boundary_path).exists():
            self._set_notice_error(self.boundary_cut_progress, f"Boundary image not found: {boundary_path}")
            self._set_progress_failed(self.boundary_cut_progress_bar)
            return
        if self._preprocess is None:
            self._set_notice_error(self.boundary_cut_progress, "Run Detection before boundary cutting.")
            self._set_progress_failed(self.boundary_cut_progress_bar)
            return

        source_is_mask = self.boundary_event_source.currentText().startswith("Mask")
        event_result = self._mask if source_is_mask else self._trace
        source_label = "Mask events" if source_is_mask else "Hough filled events"
        if event_result is None:
            self._set_notice_error(self.boundary_cut_progress, f"{source_label} are not available yet.")
            self._set_progress_failed(self.boundary_cut_progress_bar)
            return

        started = time.perf_counter()
        self._set_notice_neutral(self.boundary_cut_progress, f"Cutting {source_label.lower()} by EBSD boundary...")
        self._set_boundary_cut_result(None)
        self.boundary_cut_stats.setText("")
        self._set_status_neutral("Running boundary cut...")
        self._set_progress_value(self.boundary_cut_progress_bar, 10, "Loading boundary...")
        QApplication.processEvents()
        with busy_cursor():
            try:
                self._set_progress_value(self.boundary_cut_progress_bar, 35, "Cutting events...")
                self._set_boundary_cut_result(run_boundary_cut_events(
                    self._preprocess,
                    event_result,
                    boundary_path,
                    black_threshold=float(self.boundary_black_threshold.value()),
                    boundary_dilation_radius=int(self.boundary_dilation_radius.value()),
                    min_segment_pixels=int(self.boundary_min_segment_pixels.value()),
                    connectivity=1 if self.boundary_connectivity.currentIndex() == 0 else 2,
                ))
                self._set_progress_value(self.boundary_cut_progress_bar, 90, "Rendering overlay...")
            except Exception as exc:  # pragma: no cover - UI safety path
                self._set_status_error(f"Boundary cut failed: {exc}")
                self._set_notice_error(self.boundary_cut_progress, f"Boundary cut failed: {exc}")
                self._set_progress_failed(self.boundary_cut_progress_bar)
                return

        elapsed = time.perf_counter() - started
        self.boundary_cut_preview.set_array(
            downsample_rgb_for_view(
                self._boundary_cut["overlay"],
                resample=Image.Resampling.NEAREST,
            )
        )
        self.boundary_cut_stats.setText(
            f"Source events: {self._boundary_cut['source_count']:,}\n"
            f"Cut events: {self._boundary_cut['accepted_count']:,}\n"
            f"Source pixels: {self._boundary_cut['source_pixel_count']:,}\n"
            f"Kept pixels: {self._boundary_cut['kept_pixel_count']:,}\n"
            f"Removed boundary pixels: {self._boundary_cut['removed_pixel_count']:,}\n"
            f"Events touching boundary: {self._boundary_cut['events_touching_boundary']:,}\n"
            f"Events split: {self._boundary_cut['events_split']:,}\n"
            f"Discarded small segments: {self._boundary_cut['discarded_small_segments']:,}\n"
            f"Boundary pixels in crop: {self._boundary_cut['boundary_pixel_count']:,}"
        )
        if self._boundary_cut["removed_pixel_count"] == 0:
            if self._boundary_cut["boundary_pixel_count"] == 0:
                self._set_notice_neutral(
                    self.boundary_cut_progress,
                    f"Processed in {elapsed:.2f} seconds. No boundary pixels were found in the selected crop."
                )
            else:
                self._set_notice_neutral(
                    self.boundary_cut_progress,
                    f"Processed in {elapsed:.2f} seconds. Boundary pixels did not overlap event pixels."
                )
        else:
            self._set_notice_success(self.boundary_cut_progress, f"Processed in {elapsed:.2f} seconds.")
        self._set_progress_complete(self.boundary_cut_progress_bar)
        self._set_status_neutral(f"Boundary cut complete in {elapsed:.2f} seconds.")

    def _run_trace(self) -> None:
        if self._trace_thread is not None:
            self._set_label_neutral(self.trace_progress, "Step 2 is already running.")
            return
        self._set_label_neutral(self.trace_progress, "Tracing events from Hough seeds...")
        self._set_progress_value(self.trace_progress_bar, 5, "Starting...")
        self.trace_stats.setText("")
        self._set_status_neutral("Tracing events from Hough seeds...")
        self.detection_tabs.setCurrentIndex(1)
        self.hough_result_tabs.setCurrentIndex(1)
        self._sync_hough_step_controls(1)
        QApplication.processEvents()

        if self._hough is None:
            self._set_label_neutral(self.trace_progress, "Hough seeds missing. Running Hough Step 1 first...")
            self._set_progress_value(self.trace_progress_bar, 10, "Running Step 1 first...")
            QApplication.processEvents()
            if not self._run_hough(switch_to_seed_view=False):
                self._set_label_error(self.trace_progress, "Trace could not start because Hough Step 1 failed.")
                self._set_progress_failed(self.trace_progress_bar)
                return
            self.detection_tabs.setCurrentIndex(1)
            self.hough_result_tabs.setCurrentIndex(1)
            self._sync_hough_step_controls(1)
            self._set_progress_value(self.trace_progress_bar, 30, "Step 1 ready")

        if self._preprocess is None or self._hough is None:
            self._set_label_error(self.trace_progress, "Trace could not start because Hough seeds are unavailable.")
            self._set_progress_failed(self.trace_progress_bar)
            return

        self.trace_button.setEnabled(False)
        self._trace_thread = QThread(self)
        self._trace_worker = TraceEventsWorker(self._preprocess, self._hough, self._params())
        self._trace_worker.moveToThread(self._trace_thread)
        self._trace_thread.started.connect(self._trace_worker.run)
        self._trace_worker.progress.connect(self._trace_progress_from_worker)
        self._trace_worker.finished.connect(self._trace_finished)
        self._trace_worker.failed.connect(self._trace_failed)
        self._trace_worker.finished.connect(self._trace_worker.deleteLater)
        self._trace_worker.failed.connect(self._trace_worker.deleteLater)
        self._trace_worker.finished.connect(self._trace_thread.quit)
        self._trace_worker.failed.connect(self._trace_thread.quit)
        self._trace_thread.finished.connect(self._trace_thread_finished)
        self._trace_thread.start()

    @Slot(int, str)
    def _trace_progress_from_worker(self, value: int, text: str) -> None:
        if value < 0:
            self._set_progress_busy(self.trace_progress_bar, text)
        else:
            self._set_progress_value(self.trace_progress_bar, value, text)
        self._set_label_neutral(self.trace_progress, text)
        self._set_status_neutral(text)

    @Slot(object, float)
    def _trace_finished(self, result: object, elapsed: float) -> None:
        self._trace = result
        self._set_boundary_cut_result(None)
        self.trace_preview.set_array(self._trace["overlay"])
        total_pixels = sum(line.size for line in self._trace["accepted"])
        self.trace_stats.setText(
            f"Detected events: {self._trace['accepted_count']:,}\n"
            f"Skipped covered seeds: {self._trace.get('skipped_covered', 0):,}\n"
            f"Rejected seeds: {self._trace['rejected']:,}\n"
            f"Merged groups: {self._trace['merged']:,}\n"
            f"Event pixels: {total_pixels:,}"
        )
        self._set_label_neutral(self.trace_progress, f"Processed in {elapsed:.2f} seconds.")
        self._set_progress_complete(self.trace_progress_bar)
        self._set_status_neutral(f"Event tracing complete in {elapsed:.2f} seconds.")

    @Slot(str)
    def _trace_failed(self, message: str) -> None:
        self._set_status_error(f"Event tracing failed: {message}")
        self._set_label_error(self.trace_progress, "Event tracing failed.")
        self._set_progress_failed(self.trace_progress_bar)

    @Slot()
    def _trace_thread_finished(self) -> None:
        if self._trace_thread is not None:
            self._trace_thread.deleteLater()
        self._trace_worker = None
        self._trace_thread = None
        self.trace_button.setEnabled(True)

    def _reset_display_adjustments(self) -> None:
        self._range_controls_updating = True
        self._apply_display_adjustment_defaults()
        self._range_controls_updating = False
        self._update_display_adjustment_labels()
        self._refresh_crop_display()

    def _apply_display_adjustment_defaults(self) -> None:
        brightness = 0
        contrast = 0
        if AUTO_PARAMETER_DEFAULTS_PATH.exists():
            try:
                with AUTO_PARAMETER_DEFAULTS_PATH.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                brightness = int(data.get("display_brightness", 0))
                contrast = int(data.get("display_contrast", 0))
            except Exception:
                brightness = 0
                contrast = 0
        self.display_brightness_slider.blockSignals(True)
        self.display_contrast_slider.blockSignals(True)
        self.display_brightness_slider.setValue(max(self.display_brightness_slider.minimum(), min(self.display_brightness_slider.maximum(), brightness)))
        self.display_contrast_slider.setValue(max(self.display_contrast_slider.minimum(), min(self.display_contrast_slider.maximum(), contrast)))
        self.display_brightness_slider.blockSignals(False)
        self.display_contrast_slider.blockSignals(False)

    def _display_adjustment_changed(self) -> None:
        if self._range_controls_updating:
            return
        self._update_display_adjustment_labels()
        self._refresh_crop_display()

    def _current_display_range(self) -> tuple[float, float]:
        if not self._summary:
            return 0.0, 1.0
        return display_range_from_adjustments(
            self._display_base_range,
            self._display_slider_value(self.display_contrast_slider),
            self._summary["min"],
            self._summary["max"],
        )

    def _display_slider_value(self, slider: QSlider) -> float:
        return float(slider.value()) / DISPLAY_ADJUSTMENT_SCALE

    def _nudge_display_adjustment(self, slider: QSlider, delta: float) -> None:
        step = int(round(float(delta) * DISPLAY_ADJUSTMENT_SCALE))
        next_value = max(slider.minimum(), min(slider.maximum(), slider.value() + step))
        slider.setValue(next_value)

    def _update_display_adjustment_labels(self) -> None:
        if not self._summary:
            self.display_stats_label.setText("Raw range: -")
            self.display_brightness_label.setText("Brightness: +0")
            self.display_contrast_label.setText("Contrast: +0")
            self.display_range_label.setText("Display window: -")
            return
        brightness = self._display_slider_value(self.display_brightness_slider)
        contrast = self._display_slider_value(self.display_contrast_slider)
        display_min, display_max = self._current_display_range()
        self.display_stats_label.setText(
            f"Raw min/max: {self._summary['min']:.4g} / {self._summary['max']:.4g}\n"
            f"Auto low/high: {self._summary['p0_5']:.4g} / {self._summary['p99_5']:.4g}"
        )
        self.display_brightness_label.setText(f"Brightness: {brightness:+.1f}")
        self.display_contrast_label.setText(f"Contrast: {contrast:+.1f}")
        self.display_range_label.setText(
            f"Display window: {display_min:.4g} to {display_max:.4g}"
        )

    def _update_crop_enabled(self) -> None:
        enabled = not self.use_full_image.isChecked()
        self.crop_x.setEnabled(False)
        self.crop_y.setEnabled(False)
        for widget in (self.crop_width, self.crop_height):
            widget.setEnabled(enabled)
        if self.use_full_image.isChecked() and self._image_path:
            try:
                width, height = image_size(self._image_path)
            except Exception:
                return
            self.crop_canvas.set_crop_rect(0, 0, width, height)

    def _sync_max_seed_enabled(self) -> None:
        self.hough_max_seeds.setEnabled(not self.hough_use_all_seeds.isChecked())

    def _sync_hough_step_controls(self, index: int) -> None:
        if hasattr(self, "hough_controls_stack"):
            self.hough_controls_stack.setCurrentIndex(1 if index == 1 else 0)

    def _set_progress_value(self, progress: QProgressBar, value: int, text: str) -> None:
        progress.setRange(0, 100)
        progress.setValue(max(0, min(100, int(value))))
        progress.setFormat(f"{text} %p%")
        QApplication.processEvents()

    def _set_progress_busy(self, progress: QProgressBar, text: str) -> None:
        progress.setRange(0, 0)
        progress.setFormat(text)
        QApplication.processEvents()

    def _set_progress_complete(self, progress: QProgressBar) -> None:
        progress.setRange(0, 100)
        progress.setValue(100)
        progress.setFormat("Complete")
        QApplication.processEvents()

    def _set_progress_failed(self, progress: QProgressBar) -> None:
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setFormat("Failed")
        QApplication.processEvents()

    def _set_notice_neutral(self, label: QLabel, text: str) -> None:
        label.setText(text)
        label.setStyleSheet(
            "padding: 8px; border-radius: 6px; background: #f9fafb; color: #374151;"
        )

    def _set_notice_success(self, label: QLabel, text: str) -> None:
        label.setText(text)
        label.setStyleSheet(
            "padding: 8px; border-radius: 6px; background: #dcfce7; color: #14532d;"
        )

    def _set_notice_error(self, label: QLabel, text: str) -> None:
        label.setText(text)
        label.setStyleSheet(
            "padding: 8px; border-radius: 6px; background: #fee2e2; color: #7f1d1d; font-weight: 700;"
        )

    def _set_status_error(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet("color: #b91c1c; font-weight: 700;")

    def _set_status_neutral(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet("color: #374151;")

    @staticmethod
    def _set_label_error(label: QLabel, text: str) -> None:
        label.setText(text)
        label.setStyleSheet("color: #b91c1c; font-weight: 700;")

    @staticmethod
    def _set_label_neutral(label: QLabel, text: str) -> None:
        label.setText(text)
        label.setStyleSheet("color: #374151;")

    def _current_parameter_defaults(self) -> dict:
        return {
            "display_brightness": int(self.display_brightness_slider.value()),
            "display_contrast": int(self.display_contrast_slider.value()),
            "clahe_clip": float(self.clahe_clip.value()),
            "ridge_sigma_max": int(self.ridge_sigma_max.value()),
            "ridge_percentile": float(self.ridge_percentile.value()),
            "threshold_multiplier": float(self.threshold_multiplier.value()),
            "min_object_size": int(self.min_object_size.value()),
            "closing_radius": int(self.closing_radius.value()),
            "hough_threshold": int(self.hough_threshold.value()),
            "hough_line_length": int(self.hough_line_length.value()),
            "hough_line_gap": int(self.hough_line_gap.value()),
            "hough_seed_spacing": int(self.hough_seed_spacing.value()),
            "hough_use_all_seeds": bool(self.hough_use_all_seeds.isChecked()),
            "hough_max_seeds": int(self.hough_max_seeds.value()),
            "intensity_tolerance": int(self.intensity_tolerance.value()),
            "bfl_tolerance": float(self.bfl_tolerance.value()),
            "min_intensity": int(self.min_intensity.value()),
            "min_points": int(self.min_points.value()),
            "duplicate_overlap": float(self.duplicate_overlap.value()),
            "merge_distance": float(self.merge_distance.value()),
            "merge_angle": float(self.merge_angle.value()),
            "boundary_event_source": self.boundary_event_source.currentText(),
            "boundary_black_threshold": float(self.boundary_black_threshold.value()),
            "boundary_dilation_radius": int(self.boundary_dilation_radius.value()),
            "boundary_min_segment_pixels": int(self.boundary_min_segment_pixels.value()),
            "boundary_connectivity": self.boundary_connectivity.currentIndex(),
        }

    def _save_parameter_defaults(self) -> None:
        try:
            with AUTO_PARAMETER_DEFAULTS_PATH.open("w", encoding="utf-8") as handle:
                json.dump(self._current_parameter_defaults(), handle, indent=2)
        except Exception as exc:
            self._set_status_error(f"Could not save auto-detection defaults: {exc}")
            return
        self._set_status_neutral("Saved auto-detection parameter defaults.")

    def _load_parameter_defaults(self) -> None:
        if not AUTO_PARAMETER_DEFAULTS_PATH.exists():
            return
        try:
            with AUTO_PARAMETER_DEFAULTS_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return
        self._range_controls_updating = True
        for widget_name, key in [
            ("display_brightness_slider", "display_brightness"),
            ("display_contrast_slider", "display_contrast"),
            ("clahe_clip", "clahe_clip"),
            ("ridge_sigma_max", "ridge_sigma_max"),
            ("ridge_percentile", "ridge_percentile"),
            ("threshold_multiplier", "threshold_multiplier"),
            ("min_object_size", "min_object_size"),
            ("closing_radius", "closing_radius"),
            ("hough_threshold", "hough_threshold"),
            ("hough_line_length", "hough_line_length"),
            ("hough_line_gap", "hough_line_gap"),
            ("hough_seed_spacing", "hough_seed_spacing"),
            ("hough_max_seeds", "hough_max_seeds"),
            ("intensity_tolerance", "intensity_tolerance"),
            ("bfl_tolerance", "bfl_tolerance"),
            ("min_intensity", "min_intensity"),
            ("min_points", "min_points"),
            ("duplicate_overlap", "duplicate_overlap"),
            ("merge_distance", "merge_distance"),
            ("merge_angle", "merge_angle"),
            ("boundary_black_threshold", "boundary_black_threshold"),
            ("boundary_dilation_radius", "boundary_dilation_radius"),
            ("boundary_min_segment_pixels", "boundary_min_segment_pixels"),
        ]:
            if key in data:
                getattr(self, widget_name).setValue(data[key])
        self._range_controls_updating = False
        if "hough_use_all_seeds" in data:
            self.hough_use_all_seeds.setChecked(bool(data["hough_use_all_seeds"]))
        source = data.get("boundary_event_source")
        if source:
            index = self.boundary_event_source.findText(str(source))
            if index >= 0:
                self.boundary_event_source.setCurrentIndex(index)
        if "boundary_connectivity" in data:
            self.boundary_connectivity.setCurrentIndex(int(data["boundary_connectivity"]))
        self._sync_max_seed_enabled()
        self._update_display_adjustment_labels()
        self._clahe_spin_changed()

    def _clear_downstream(self) -> None:
        self._region = None
        self._preprocess = None
        self._preprocess_preview = None
        self._hough = None
        self._mask = None
        self._trace = None
        self._set_boundary_cut_result(None)
        for preview in (self.display_preview, self.hough_preview, self.mask_preview, self.trace_preview, self.boundary_cut_preview):
            preview.set_array(None)
        if hasattr(self, "crop_preview"):
            self.crop_preview.set_array(None)

    def _refresh_crop_canvas(self) -> None:
        if not self._image_path:
            return
        with busy_cursor():
            try:
                display_min, display_max = self._current_display_range()
                preview_rgb, scale_x, scale_y = load_downsampled_preview(
                    self._image_path,
                    float(display_min),
                    float(display_max),
                    self._display_slider_value(self.display_brightness_slider),
                )
            except Exception as exc:
                self._set_status_error(f"Could not show crop image: {exc}")
                return
        self.crop_canvas.set_image(
            preview_rgb,
            scale_x,
            scale_y,
            preserve_view=self.crop_canvas._image_shape is not None,
        )
        self.crop_canvas.set_crop_rect(
            int(self.crop_x.value()),
            int(self.crop_y.value()),
            int(self.crop_width.value()),
            int(self.crop_height.value()),
        )

    def _refresh_crop_preview(self) -> None:
        if not self._image_path or not hasattr(self, "crop_preview"):
            return
        with busy_cursor():
            try:
                preview_region = load_downsampled_processing_region(self._params(), max_dim=1400)
            except Exception as exc:
                self._set_status_error(f"Could not show selected crop preview: {exc}")
                self.crop_preview.set_array(None)
                return
        self.crop_preview.set_array(preview_region["display_rgb"])

    def _set_crop_from_canvas(self, x: int, y: int, width: int, height: int) -> None:
        if self._image_path:
            try:
                image_width, image_height = image_size(self._image_path)
            except Exception:
                image_width, image_height = x + width, y + height
            width = max(1, min(int(width), image_width))
            height = max(1, min(int(height), image_height))
            x = max(0, min(int(x), image_width - width))
            y = max(0, min(int(y), image_height - height))
        self.crop_x.blockSignals(True)
        self.crop_y.blockSignals(True)
        self.crop_width.blockSignals(True)
        self.crop_height.blockSignals(True)
        self.crop_x.setValue(x)
        self.crop_y.setValue(y)
        self.crop_width.setValue(width)
        self.crop_height.setValue(height)
        self.crop_x.blockSignals(False)
        self.crop_y.blockSignals(False)
        self.crop_width.blockSignals(False)
        self.crop_height.blockSignals(False)
        self.use_full_image.setChecked(False)
        self._refresh_from_region_change()

    def _crop_spin_changed(self) -> None:
        self.crop_canvas.set_crop_rect(
            int(self.crop_x.value()),
            int(self.crop_y.value()),
            int(self.crop_width.value()),
            int(self.crop_height.value()),
        )
        self._refresh_from_region_change()

    def _refresh_from_region_change(self) -> None:
        if self._image_path:
            self._refresh_crop_preview()
            self._run_region()

    def _refresh_crop_display(self) -> None:
        if not self._image_path:
            return
        self._refresh_crop_canvas()
        self._refresh_crop_preview()
        self._region = None
        self._preprocess = None
        self._preprocess_preview = None
        self._hough = None
        self._mask = None
        self._trace = None
        self._set_boundary_cut_result(None)
        for preview in (self.hough_preview, self.mask_preview, self.trace_preview, self.boundary_cut_preview):
            preview.set_array(None)
        if self._display_stage == "original":
            try:
                preview_region = load_downsampled_processing_region(self._params())
            except Exception as exc:
                self._set_status_error(f"Could not update brightness/contrast preview: {exc}")
                self.display_preview.set_array(None)
            else:
                self.display_preview.set_array(preview_region["display_rgb"])

    def _set_display_stage(self, stage: str) -> None:
        self._display_stage = stage
        self._update_stage_buttons()
        self._update_stage_options()
        self._refresh_display()

    def _update_stage_options(self) -> None:
        if not hasattr(self, "option_stack"):
            return
        self._update_stage_buttons()
        stage_indices = {
            "original": 0,
            "clahe": 1,
            "ridge": 2,
            "clean": 3,
        }
        self.option_stack.setCurrentIndex(stage_indices.get(self._display_stage, 0))

    def _update_stage_buttons(self) -> None:
        if not hasattr(self, "display_original_button"):
            return
        buttons = {
            "original": self.display_original_button,
            "clahe": self.display_clahe_button,
            "ridge": self.display_ridge_button,
            "clean": self.display_clean_button,
        }
        for stage, button in buttons.items():
            button.setChecked(stage == self._display_stage)

    def _auto_process_common_steps(self) -> None:
        if not self._image_path:
            self._set_status_neutral("Load a BLN image in Project Files before running Common Steps.")
            return
        self._display_stage = "clean"
        self._update_stage_buttons()
        self._run_preprocessing()

    def _clahe_spin_changed(self) -> None:
        value = int(round(self.clahe_clip.value() * 1000.0))
        self.clahe_clip_slider.blockSignals(True)
        self.clahe_clip_slider.setValue(max(1, min(1000, value)))
        self.clahe_clip_slider.blockSignals(False)
        self._refresh_display()

    def _clahe_slider_changed(self, value: int) -> None:
        self.clahe_clip.blockSignals(True)
        self.clahe_clip.setValue(max(1, int(value)) / 1000.0)
        self.clahe_clip.blockSignals(False)
        self._refresh_display()

    def _refresh_display(self) -> None:
        if not self._image_path:
            return
        if self._display_stage == "original":
            self._show_display_stage()
            return
        self._run_preprocessing()

    def _show_display_stage(self) -> None:
        if self._display_stage == "original":
            try:
                preview_region = load_downsampled_processing_region(self._params())
            except Exception:
                self.display_preview.set_array(None)
            else:
                self.display_preview.set_array(preview_region["display_rgb"])
            return
        if self._preprocess_preview is None:
            self.display_preview.set_array(None)
            return
        if self._display_stage == "clahe":
            self.display_preview.set_array(self._preprocess_preview["enhanced"])
        elif self._display_stage == "ridge":
            self.display_preview.set_array(self._preprocess_preview["candidate_mask"])
        elif self._display_stage == "clean":
            self.display_preview.set_array(self._preprocess_preview["candidate_clean"])

    @staticmethod
    def _spin(minimum: int, maximum: int, step: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        decimals: int = 3,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin
