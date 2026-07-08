from __future__ import annotations

import csv
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, Qt
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
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
    run_hough_seed_detection,
    run_preprocessing,
    trace_hough_events,
)


MAX_CROP_PREVIEW_DIM = 1800
MAX_OVERLAY_PREVIEW_DIM = 2400
DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[2]
    / "z_share_DIC_data_for_hv_mvu_pk"
    / "Ti_Cryo"
    / "Fused_BlN_step3.tif"
)


def display_crop_rgb_fast(arr: np.ndarray, display_min: float, display_max: float) -> np.ndarray:
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
    return np.repeat(gray[..., None], 3, axis=2)


def load_downsampled_preview(
    path: str,
    display_min: float,
    display_max: float,
    max_dim: int = MAX_CROP_PREVIEW_DIM,
) -> tuple[np.ndarray, float, float]:
    with Image.open(path) as img:
        original_width, original_height = img.size
        scale = min(1.0, max_dim / max(original_width, original_height))
        preview_width = max(1, int(round(original_width * scale)))
        preview_height = max(1, int(round(original_height * scale)))
        preview = img.resize((preview_width, preview_height), Image.Resampling.BILINEAR)
        arr = np.asarray(preview).copy()
    rgb = display_crop_rgb_fast(arr, display_min, display_max)
    scale_x = original_width / preview_width
    scale_y = original_height / preview_height
    return rgb, scale_x, scale_y


def downsample_rgb_for_view(image: np.ndarray, max_dim: int = MAX_OVERLAY_PREVIEW_DIM) -> np.ndarray:
    if max(image.shape[:2]) <= max_dim:
        return image
    pil_image = Image.fromarray(display_image(image))
    pil_image.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)
    return np.asarray(pil_image).copy()


def sanitize_file_prefix(prefix: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix.strip())
    return safe.strip("._-")


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
    display_rgb = display_crop_rgb_fast(arr, params.display_min, params.display_max)
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


class AutoDetectionPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._image_path: str | None = str(DEFAULT_IMAGE) if DEFAULT_IMAGE.exists() else None
        self._summary: dict[str, float] | None = None
        self._region: dict | None = None
        self._preprocess: dict | None = None
        self._preprocess_preview: dict | None = None
        self._hough: dict | None = None
        self._trace: dict | None = None
        self._range_controls_updating = False

        layout = QVBoxLayout(self)
        header = QLabel("Auto event detection")
        header.setStyleSheet("font-weight: 700; font-size: 16px;")
        layout.addWidget(header)

        self.status_label = QLabel("Load a BLN image in Crop, choose a region, then tune preprocessing.")
        self.status_label.setWordWrap(True)
        self.status_label.setToolTip("Shows the current auto-detection workflow status and any processing messages.")
        layout.addWidget(self.status_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self._build_crop_tab()
        self._build_display_tab()
        self._build_hough_tab()
        self._build_trace_tab()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        if self._image_path:
            self._load_image_metadata(self._image_path)

    def _build_crop_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)

        sidebar = QWidget()
        sidebar.setMaximumWidth(340)
        side_layout = QVBoxLayout(sidebar)
        load_button = QPushButton("Load Image")
        load_button.setToolTip("Choose the BLN/DIC image used for crop selection, preprocessing, Hough seeds, and tracing.")
        load_button.clicked.connect(self._choose_image)
        side_layout.addWidget(load_button)
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

        display_box = QGroupBox("Display")
        display_box.setToolTip("Controls only how the image is displayed. Values below the minimum are black; values above the maximum are white.")
        display_layout = QVBoxLayout(display_box)
        range_form = QFormLayout()
        self.display_min = self._double_spin(-1_000_000.0, 1_000_000.0, 0.001, 0.0, decimals=6)
        self.display_max = self._double_spin(-1_000_000.0, 1_000_000.0, 0.001, 1.0, decimals=6)
        self.display_min.setToolTip("Lower display range. Raw values at or below this value display as black.")
        self.display_max.setToolTip("Upper display range. Raw values at or above this value display as white.")
        range_form.addRow("Range min", self.display_min)
        range_form.addRow("Range max", self.display_max)
        display_layout.addLayout(range_form)
        self.display_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.display_max_slider = QSlider(Qt.Orientation.Horizontal)
        for slider in (self.display_min_slider, self.display_max_slider):
            slider.setRange(0, 1000)
            slider.setSingleStep(1)
            slider.setPageStep(25)
        self.display_min_slider.setToolTip("Coarse lower display range. Drag to quickly darken or brighten the image.")
        self.display_max_slider.setToolTip("Coarse upper display range. Drag to quickly change contrast.")
        display_layout.addWidget(QLabel("Range min slider"))
        display_layout.addWidget(self.display_min_slider)
        display_layout.addWidget(QLabel("Range max slider"))
        display_layout.addWidget(self.display_max_slider)

        range_buttons = QHBoxLayout()
        auto_range = QPushButton("Auto Range")
        auto_range.setToolTip("Use the 0.5th and 99.5th percentile values.")
        auto_range.clicked.connect(self._set_auto_range)
        reset_range = QPushButton("Reset Range")
        reset_range.setToolTip("Use the true raw minimum and maximum values.")
        reset_range.clicked.connect(self._set_reset_range)
        range_buttons.addWidget(auto_range)
        range_buttons.addWidget(reset_range)
        display_layout.addLayout(range_buttons)
        side_layout.addWidget(display_box)
        side_layout.addStretch(1)

        self.crop_canvas = CropCanvas()
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
        layout.addWidget(sidebar)
        layout.addWidget(right_side, 1)

        for widget in (self.crop_x, self.crop_y, self.crop_width, self.crop_height):
            widget.valueChanged.connect(self._crop_spin_changed)
        self.display_min.valueChanged.connect(self._display_spin_changed)
        self.display_max.valueChanged.connect(self._display_spin_changed)
        self.display_min_slider.valueChanged.connect(self._display_slider_changed)
        self.display_max_slider.valueChanged.connect(self._display_slider_changed)
        self.tabs.addTab(tab, "Crop")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(tab),
            "Load the BLN image, adjust display range, and select the full image or a crop.",
        )

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if (
            event.type() == QEvent.Type.KeyPress
            and self.tabs.currentWidget() is not None
            and self.tabs.tabText(self.tabs.currentIndex()) in {"Crop", "Preprocessing", "Hough Seeds", "Trace Events"}
            and self.isVisible()
        ):
            active_tab = self.tabs.tabText(self.tabs.currentIndex())
            if active_tab == "Crop":
                target = self.crop_canvas
            elif active_tab == "Hough Seeds":
                target = self.hough_preview
            elif active_tab == "Trace Events":
                target = self.trace_preview
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

        controls = QGroupBox("Preprocessing")
        controls.setMaximumWidth(340)
        controls.setToolTip("Tune each preprocessing stage on a downsampled preview. Full-resolution processing is run later for Hough detection.")
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
        self.display_original_button = QPushButton("Original")
        self.display_clahe_button = QPushButton("CLAHE")
        self.display_ridge_button = QPushButton("Ridge")
        self.display_clean_button = QPushButton("Clean Mask")
        self.display_seed_button = QPushButton("Seed Mask")
        self.display_original_button.setToolTip("Shows the selected crop after display range mapping.")
        self.display_clahe_button.setToolTip("Applies CLAHE to the original selected crop.")
        self.display_ridge_button.setToolTip("Applies the ridge filter to the CLAHE output.")
        self.display_clean_button.setToolTip("Thresholds and cleans the ridge-filter output.")
        self.display_seed_button.setToolTip("Creates the seed-detection mask from the cleaned mask.")
        for button in (
            self.display_original_button,
            self.display_clahe_button,
            self.display_ridge_button,
            self.display_clean_button,
            self.display_seed_button,
        ):
            button.setCheckable(True)
            button.setProperty("stageButton", True)
            button.setMinimumHeight(36)
            control_layout.addWidget(button)
        self.display_original_button.clicked.connect(lambda: self._set_display_stage("original"))
        self.display_clahe_button.clicked.connect(lambda: self._set_display_stage("clahe"))
        self.display_ridge_button.clicked.connect(lambda: self._set_display_stage("ridge"))
        self.display_clean_button.clicked.connect(lambda: self._set_display_stage("clean"))
        self.display_seed_button.clicked.connect(lambda: self._set_display_stage("seed"))
        self._display_stage = "original"
        control_layout.addSpacing(14)

        self.option_stack = QStackedWidget()
        self.option_stack.setMinimumHeight(210)
        self.option_stack.setToolTip("Options for the currently selected preprocessing stage.")

        self.original_options = QGroupBox("Original")
        self.original_options.setToolTip("No parameters are applied. This shows the selected crop after display range mapping.")
        original_layout = QVBoxLayout(self.original_options)
        original_layout.addStretch(1)
        self.option_stack.addWidget(self.original_options)

        self.clahe_options = QGroupBox("CLAHE Options")
        self.clahe_options.setToolTip("Contrast-limited adaptive histogram equalization. It improves local contrast before ridge detection.")
        form = QFormLayout(self.clahe_options)
        self.clahe_clip = self._double_spin(0.001, 1.0, 0.001, 0.100, decimals=3)
        self.clahe_clip.setToolTip("Limits local contrast amplification in CLAHE.")
        self.clahe_clip_slider = QSlider(Qt.Orientation.Horizontal)
        self.clahe_clip_slider.setRange(1, 1000)
        self.clahe_clip_slider.setValue(100)
        self.clahe_clip_slider.setToolTip("Limits local contrast amplification in CLAHE.")
        form.addRow("Clip limit", self.clahe_clip)
        form.addRow("Clip slider", self.clahe_clip_slider)
        self.option_stack.addWidget(self.clahe_options)

        self.ridge_options = QGroupBox("Ridge Options")
        self.ridge_options.setToolTip("Ridge filtering enhances line-like features after CLAHE.")
        form = QFormLayout(self.ridge_options)
        self.ridge_sigma_max = self._spin(1, 12, 1, 4)
        self.ridge_sigma_max.setToolTip("Largest ridge-filter scale in pixels.")
        form.addRow("Ridge sigma max", self.ridge_sigma_max)
        self.option_stack.addWidget(self.ridge_options)

        self.clean_options = QGroupBox("Clean Mask Options")
        self.clean_options.setToolTip("Thresholds and cleans the ridge response to remove small noise and bridge small gaps.")
        form = QFormLayout(self.clean_options)
        self.ridge_percentile = self._double_spin(80.0, 99.9, 0.1, 88.0, decimals=1)
        self.ridge_percentile.setToolTip("Percentile cutoff for the ridge response. Higher values keep fewer, stronger ridge pixels.")
        self.threshold_multiplier = self._double_spin(0.1, 2.0, 0.05, 0.40, decimals=2)
        self.threshold_multiplier.setToolTip("Multiplier applied to the Otsu ridge threshold. Lower values include weaker ridges; higher values are stricter.")
        self.min_object_size = self._spin(1, 5000, 5, 231)
        self.min_object_size.setToolTip("Removes connected ridge-mask objects smaller than this many pixels.")
        self.closing_radius = self._spin(0, 20, 1, 3)
        self.closing_radius.setToolTip("Morphological closing radius. Higher values bridge small gaps but can merge nearby ridges.")
        form.addRow("Ridge percentile", self.ridge_percentile)
        form.addRow("Otsu multiplier", self.threshold_multiplier)
        form.addRow("Min object pixels", self.min_object_size)
        form.addRow("Closing radius", self.closing_radius)
        self.option_stack.addWidget(self.clean_options)

        self.seed_options = QGroupBox("Seed Mask Options")
        self.seed_options.setToolTip("Creates the final mask used by the Hough seed detector.")
        form = QFormLayout(self.seed_options)
        self.use_skeletonize = QCheckBox("Use skeletonized mask for seed detection")
        self.use_skeletonize.setChecked(True)
        self.use_skeletonize.setToolTip("Thin the cleaned ridge mask to centerlines before Hough detection.")
        form.addRow("", self.use_skeletonize)
        self.option_stack.addWidget(self.seed_options)
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
        self.use_skeletonize.toggled.connect(self._refresh_display)
        self._update_stage_options()
        self.tabs.addTab(tab, "Preprocessing")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(tab),
            "Preview and tune preprocessing stages: Original, CLAHE, Ridge, Clean Mask, and Seed Mask.",
        )

    def _build_hough_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)

        controls = QGroupBox("Hough Seeds")
        controls.setMaximumWidth(340)
        controls.setToolTip("Detect line segments from the seed mask and choose seed pixels for event tracing.")
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
            """
        )
        control_layout = QVBoxLayout(controls)
        params_box = QGroupBox("Parameters")
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
        control_layout.addWidget(params_box)

        run_button = QPushButton("Detect Hough Lines And Seeds")
        run_button.setToolTip("Runs full-resolution preprocessing, detects Hough line segments, then selects seed points from the strongest pixel on each line.")
        run_button.clicked.connect(self._run_hough)
        control_layout.addWidget(run_button)

        self.hough_progress = QLabel("Ready.")
        self.hough_progress.setWordWrap(True)
        self.hough_progress.setToolTip("Shows whether Hough seed detection is idle, running, failed, or completed with elapsed time.")
        control_layout.addWidget(self.hough_progress)

        self.hough_stats = QLabel("No Hough result yet.")
        self.hough_stats.setWordWrap(True)
        self.hough_stats.setToolTip("After processing, shows detected Hough line count, seed candidate count, and selected seed count.")
        control_layout.addWidget(self.hough_stats)
        control_layout.addStretch(1)

        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_label = QLabel("Hough Lines And Selected Seeds")
        image_label.setToolTip("Overlay image: Hough lines are red and selected seeds are yellow.")
        image_label.setStyleSheet("font-weight: 700;")
        image_layout.addWidget(image_label)
        self.hough_preview = ZoomImageView()
        self.hough_preview.setMinimumSize(520, 420)
        self.hough_preview.setToolTip("Red pixels show detected Hough lines. Yellow points show selected seeds. Use =/+ and - to zoom, 0 to fit, or mouse wheel to zoom.")
        image_layout.addWidget(self.hough_preview, 1)

        layout.addWidget(controls)
        layout.addWidget(image_panel, 1)
        self.tabs.addTab(tab, "Hough Seeds")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(tab),
            "Run Hough line detection and select seed points for event tracing.",
        )
        self._sync_max_seed_enabled()

    def _build_trace_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)

        controls = QGroupBox("Trace Events")
        controls.setMaximumWidth(340)
        controls.setToolTip("Grow DIC events from Hough seeds and optionally merge compatible event fragments.")
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
            """
        )
        control_layout = QVBoxLayout(controls)
        params_box = QGroupBox("Parameters")
        params_box.setToolTip("Parameters for Java-style event growing and event merging.")
        form = QFormLayout(params_box)
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
        self.connect_gaps = QCheckBox("Connect merged event gaps")
        self.connect_gaps.setChecked(True)
        self.connect_gaps.setToolTip("Add thin bridge pixels between disconnected merged components.")
        form.addRow("Intensity difference tolerance", self.intensity_tolerance)
        form.addRow("Best fit line tolerance", self.bfl_tolerance)
        form.addRow("Minimum intensity", self.min_intensity)
        form.addRow("Minimum points", self.min_points)
        form.addRow("Duplicate overlap", self.duplicate_overlap)
        form.addRow("Merge distance", self.merge_distance)
        form.addRow("Merge angle", self.merge_angle)
        form.addRow("", self.connect_gaps)
        control_layout.addWidget(params_box)

        run_button = QPushButton("Trace Events From Hough Seeds")
        run_button.setToolTip("Grows events from the selected Hough seeds using the Java-style algorithm, then merges duplicate or nearby compatible events.")
        run_button.clicked.connect(self._run_trace)
        control_layout.addWidget(run_button)

        self.trace_progress = QLabel("Ready.")
        self.trace_progress.setWordWrap(True)
        self.trace_progress.setToolTip("Shows whether event tracing is idle, running, failed, or completed with elapsed time.")
        control_layout.addWidget(self.trace_progress)

        self.trace_stats = QLabel("No traced events yet.")
        self.trace_stats.setWordWrap(True)
        self.trace_stats.setToolTip("After tracing, shows detected event count, rejected seeds, merged groups, and event pixel count.")
        control_layout.addWidget(self.trace_stats)

        save_box = QGroupBox("Save")
        save_box.setToolTip("Save traced Hough events as CSV files using global image coordinates.")
        save_layout = QVBoxLayout(save_box)
        self.trace_save_prefix = QLineEdit()
        self.trace_save_prefix.setPlaceholderText("event_file_prefix")
        self.trace_save_prefix.setToolTip("File prefix used for <prefix>_events.csv and <prefix>_event_pixels.csv.")
        save_layout.addWidget(self.trace_save_prefix)
        save_button = QPushButton("Save Detected Events")
        save_button.setToolTip("Save detected events and their pixel coordinates to CSV files.")
        save_button.clicked.connect(self._save_trace_events)
        save_layout.addWidget(save_button)
        self.trace_save_status = QLabel("")
        self.trace_save_status.setWordWrap(True)
        self.trace_save_status.setToolTip("Shows the saved CSV file locations after saving.")
        save_layout.addWidget(self.trace_save_status)
        control_layout.addWidget(save_box)
        control_layout.addStretch(1)

        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_label = QLabel("Detected Events")
        image_label.setToolTip("Overlay image showing event pixels grown from Hough seeds.")
        image_label.setStyleSheet("font-weight: 700;")
        image_layout.addWidget(image_label)
        self.trace_preview = ZoomImageView()
        self.trace_preview.setMinimumSize(520, 420)
        self.trace_preview.setToolTip("Detected event pixels are shown as an overlay. Use =/+ and - to zoom, 0 to fit, or mouse wheel to zoom.")
        image_layout.addWidget(self.trace_preview, 1)

        layout.addWidget(controls)
        layout.addWidget(image_panel, 1)
        self.tabs.addTab(tab, "Trace Events")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(tab),
            "Trace events from selected Hough seeds and inspect the detected-event overlay.",
        )

    def _scroll_tab(self) -> tuple[QScrollArea, QWidget]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        return scroll, content

    def _choose_image(self) -> None:
        start = str(Path(self._image_path).parent) if self._image_path else str(Path.cwd())
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Choose BLN image",
            start,
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )
        if path:
            self._load_image_metadata(path)

    def _load_image_metadata(self, path: str) -> None:
        with busy_cursor():
            self.status_label.setText("Loading image...")
            try:
                width, height = image_size(path)
                self._summary = image_value_summary(path)
            except Exception as exc:  # pragma: no cover - UI safety path
                self.status_label.setText(f"Could not load image: {exc}")
                return

        self._image_path = path
        self.dimension_label.setText(f"Original dimension: {width} x {height} pixels")
        self.status_label.setText(f"Loaded image: {width} x {height}.")
        display_min, display_max = default_display_range(self._summary)
        self._range_controls_updating = True
        self.display_min.blockSignals(True)
        self.display_max.blockSignals(True)
        self.display_min_slider.blockSignals(True)
        self.display_max_slider.blockSignals(True)
        self.display_min.setValue(display_min)
        self.display_max.setValue(display_max)
        self._sync_range_sliders_from_spins()
        self.display_min_slider.blockSignals(False)
        self.display_max_slider.blockSignals(False)
        self.display_min.blockSignals(False)
        self.display_max.blockSignals(False)
        self._range_controls_updating = False
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
        self._run_region()
        self._update_default_trace_prefix()

    def _params(self) -> AutoPipelineParams:
        if not self._image_path:
            raise ValueError("No BLN image is loaded.")
        return AutoPipelineParams(
            image_path=self._image_path,
            display_min=float(self.display_min.value()),
            display_max=float(self.display_max.value()),
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
            use_skeletonize=bool(self.use_skeletonize.isChecked()),
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
            connect_merged_event_gaps=bool(self.connect_gaps.isChecked()),
        )

    def _run_region(self) -> None:
        try:
            self._region = load_region(self._params())
        except Exception as exc:  # pragma: no cover - UI safety path
            self.status_label.setText(f"Region load failed: {exc}")
            return
        origin = self._region["origin"]
        height, width = self._region["display_rgb"].shape[:2]
        self.status_label.setText(f"Loaded region at x={origin[0]}, y={origin[1]}, size {width} x {height}.")
        self._preprocess = None
        self._preprocess_preview = None
        self._hough = None
        self._trace = None
        self._show_display_stage()
        self._update_default_trace_prefix()

    def _run_preprocessing(self) -> None:
        with busy_cursor():
            try:
                self._preprocess_preview = run_preprocessing_preview(self._params())
            except Exception as exc:  # pragma: no cover - UI safety path
                self.status_label.setText(f"Preprocessing failed: {exc}")
                return
        self.status_label.setText("Preprocessing preview updated.")
        self._preprocess = None
        self._hough = None
        self._trace = None
        self._show_display_stage()

    def _run_hough(self) -> None:
        started = time.perf_counter()
        self.hough_progress.setText("Processing Hough lines and seeds...")
        self.hough_stats.setText("")
        self.status_label.setText("Processing Hough lines and seeds...")
        QApplication.processEvents()
        with busy_cursor():
            try:
                self._preprocess = run_preprocessing(self._params())
            except Exception as exc:  # pragma: no cover - UI safety path
                self.status_label.setText(f"Full-resolution preprocessing failed: {exc}")
                self.hough_progress.setText("Processing failed during full-resolution preprocessing.")
                return
            try:
                self._hough = run_hough_seed_detection(self._preprocess, self._params())
            except Exception as exc:  # pragma: no cover - UI safety path
                self.status_label.setText(f"Hough seed detection failed: {exc}")
                self.hough_progress.setText("Processing failed during Hough seed detection.")
                return
        elapsed = time.perf_counter() - started
        if self.hough_use_all_seeds.isChecked():
            self.hough_max_seeds.setMaximum(max(1, self._hough["candidate_seed_count"]))
            self.hough_max_seeds.setValue(max(1, self._hough["candidate_seed_count"]))
        self.hough_preview.set_array(self._hough["overlay"])
        self.hough_stats.setText(
            f"Detected Hough lines: {self._hough['raw_count']:,}\n"
            f"Hough seed candidates: {self._hough['candidate_seed_count']:,}\n"
            f"Selected seeds: {len(self._hough['seeds']):,}"
        )
        self.hough_progress.setText(f"Processed in {elapsed:.2f} seconds.")
        self.status_label.setText(f"Hough line and seed detection complete in {elapsed:.2f} seconds.")
        self._trace = None

    def _run_trace(self) -> None:
        started = time.perf_counter()
        self.trace_progress.setText("Tracing events from Hough seeds...")
        self.trace_stats.setText("")
        self.status_label.setText("Tracing events from Hough seeds...")
        QApplication.processEvents()
        with busy_cursor():
            if self._hough is None:
                self.trace_progress.setText("Hough seeds missing. Running Hough Step 1 first...")
                QApplication.processEvents()
                self._run_hough()
            if self._preprocess is None or self._hough is None:
                self.trace_progress.setText("Trace could not start because Hough seeds are unavailable.")
                return
            try:
                self._trace = trace_hough_events(self._preprocess, self._hough, self._params())
            except Exception as exc:  # pragma: no cover - UI safety path
                self.status_label.setText(f"Event tracing failed: {exc}")
                self.trace_progress.setText("Event tracing failed.")
                return
        elapsed = time.perf_counter() - started
        self.trace_preview.set_array(downsample_rgb_for_view(self._trace["overlay"]))
        total_pixels = sum(line.size for line in self._trace["accepted"])
        self.trace_stats.setText(
            f"Detected events: {self._trace['accepted_count']:,}\n"
            f"Rejected seeds: {self._trace['rejected']:,}\n"
            f"Merged groups: {self._trace['merged']:,}\n"
            f"Event pixels: {total_pixels:,}"
        )
        self.trace_progress.setText(f"Processed in {elapsed:.2f} seconds.")
        self.status_label.setText(f"Event tracing complete in {elapsed:.2f} seconds.")
        self._update_default_trace_prefix()

    def _default_trace_prefix(self) -> str:
        if not self._image_path:
            return "hough_events"
        image_stem = Path(self._image_path).stem
        return (
            f"{image_stem}_hough_"
            f"x{int(self.crop_x.value())}_y{int(self.crop_y.value())}_"
            f"w{int(self.crop_width.value())}_h{int(self.crop_height.value())}"
        )

    def _update_default_trace_prefix(self) -> None:
        if not hasattr(self, "trace_save_prefix"):
            return
        if not self.trace_save_prefix.text().strip():
            self.trace_save_prefix.setText(self._default_trace_prefix())

    def _save_trace_events(self) -> None:
        if self._trace is None or self._preprocess is None:
            self.trace_save_status.setText("Run Trace Events before saving.")
            return
        safe_prefix = sanitize_file_prefix(self.trace_save_prefix.text())
        if not safe_prefix:
            self.trace_save_status.setText("Enter a file prefix before saving.")
            return

        out_dir = Path(__file__).resolve().parents[2] / "tests" / "outputs" / "hough_events"
        out_dir.mkdir(parents=True, exist_ok=True)
        events_path = out_dir / f"{safe_prefix}_events.csv"
        pixels_path = out_dir / f"{safe_prefix}_event_pixels.csv"
        crop_x, crop_y = self._preprocess["crop"]["origin"]
        created_at = datetime.now().astimezone().isoformat(timespec="seconds")

        event_rows: list[dict] = []
        pixel_rows: list[dict] = []
        assigned_pixels: set[tuple[int, int]] = set()
        duplicate_pixel_count = 0
        skipped_event_count = 0
        saved_event_index = 0

        for line in self._trace["accepted"]:
            unique_points: list[tuple[int, int]] = []
            for point in sorted(line.points, key=lambda p: (p.y, p.x)):
                global_pixel = (crop_x + int(point.x), crop_y + int(point.y))
                if global_pixel in assigned_pixels:
                    duplicate_pixel_count += 1
                    continue
                assigned_pixels.add(global_pixel)
                unique_points.append(global_pixel)
            if not unique_points:
                skipped_event_count += 1
                continue

            saved_event_index += 1
            event_id = f"{safe_prefix}_{saved_event_index:04d}"
            event_rows.append(
                {
                    "event_id": event_id,
                    "method": "hough",
                    "num_pixels": len(unique_points),
                    "crop_x": crop_x,
                    "crop_y": crop_y,
                    "image_path": self._image_path,
                    "created_at": created_at,
                }
            )
            for pixel_x, pixel_y in unique_points:
                pixel_rows.append(
                    {
                        "event_id": event_id,
                        "pixel_x": pixel_x,
                        "pixel_y": pixel_y,
                    }
                )

        with busy_cursor():
            with events_path.open("w", newline="") as events_file:
                writer = csv.DictWriter(
                    events_file,
                    fieldnames=[
                        "event_id",
                        "method",
                        "num_pixels",
                        "crop_x",
                        "crop_y",
                        "image_path",
                        "created_at",
                    ],
                )
                writer.writeheader()
                writer.writerows(event_rows)
            with pixels_path.open("w", newline="") as pixels_file:
                writer = csv.DictWriter(
                    pixels_file,
                    fieldnames=["event_id", "pixel_x", "pixel_y"],
                )
                writer.writeheader()
                writer.writerows(pixel_rows)

        detail = (
            f"Saved {len(event_rows)} events and {len(pixel_rows)} unique pixels.\n"
            f"{events_path}\n{pixels_path}"
        )
        if duplicate_pixel_count:
            detail += f"\nRemoved {duplicate_pixel_count} duplicate shared pixels."
        if skipped_event_count:
            detail += f"\nSkipped {skipped_event_count} duplicate-only events."
        self.trace_save_status.setText(detail)
        self.status_label.setText("Detected events saved.")

    def _set_auto_range(self) -> None:
        if not self._summary:
            return
        self.display_min.setValue(self._summary["p0_5"])
        self.display_max.setValue(self._summary["p99_5"])

    def _set_reset_range(self) -> None:
        if not self._summary:
            return
        self.display_min.setValue(self._summary["min"])
        self.display_max.setValue(self._summary["max"])

    def _display_spin_changed(self) -> None:
        if self._range_controls_updating:
            return
        self._range_controls_updating = True
        self.display_min_slider.blockSignals(True)
        self.display_max_slider.blockSignals(True)
        self._sync_range_sliders_from_spins()
        self.display_min_slider.blockSignals(False)
        self.display_max_slider.blockSignals(False)
        self._range_controls_updating = False
        self._refresh_crop_display()

    def _display_slider_changed(self) -> None:
        if self._range_controls_updating or not self._summary:
            return
        raw_min = self._summary["min"]
        raw_max = self._summary["max"]
        span = raw_max - raw_min
        if span <= 0:
            return
        min_value = raw_min + span * (self.display_min_slider.value() / 1000.0)
        max_value = raw_min + span * (self.display_max_slider.value() / 1000.0)
        if max_value <= min_value:
            sender = self.sender()
            if sender is self.display_min_slider:
                max_value = min(raw_max, min_value + span / 1000.0)
            else:
                min_value = max(raw_min, max_value - span / 1000.0)
        self._range_controls_updating = True
        self.display_min.blockSignals(True)
        self.display_max.blockSignals(True)
        self.display_min.setValue(min_value)
        self.display_max.setValue(max_value)
        self.display_min.blockSignals(False)
        self.display_max.blockSignals(False)
        self._range_controls_updating = False
        self._refresh_crop_display()

    def _sync_range_sliders_from_spins(self) -> None:
        if not self._summary:
            return
        raw_min = self._summary["min"]
        raw_max = self._summary["max"]
        span = raw_max - raw_min
        if span <= 0:
            self.display_min_slider.setValue(0)
            self.display_max_slider.setValue(1000)
            return
        min_pos = int(round((self.display_min.value() - raw_min) / span * 1000.0))
        max_pos = int(round((self.display_max.value() - raw_min) / span * 1000.0))
        self.display_min_slider.setValue(max(0, min(1000, min_pos)))
        self.display_max_slider.setValue(max(0, min(1000, max_pos)))

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

    def _clear_downstream(self) -> None:
        self._region = None
        self._preprocess = None
        self._preprocess_preview = None
        self._hough = None
        self._trace = None
        for preview in (self.display_preview, self.hough_preview, self.trace_preview):
            preview.set_array(None)

    def _refresh_crop_canvas(self) -> None:
        if not self._image_path:
            return
        with busy_cursor():
            try:
                preview_rgb, scale_x, scale_y = load_downsampled_preview(
                    self._image_path,
                    float(self.display_min.value()),
                    float(self.display_max.value()),
                )
            except Exception as exc:
                self.status_label.setText(f"Could not show crop image: {exc}")
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
            self._run_region()

    def _refresh_crop_display(self) -> None:
        if not self._image_path:
            return
        self._refresh_crop_canvas()
        self._preprocess_preview = None
        self._run_region()

    def _set_display_stage(self, stage: str) -> None:
        self._display_stage = stage
        self._update_stage_buttons()
        self._update_stage_options()
        self._refresh_display()

    def _update_stage_options(self) -> None:
        if not hasattr(self, "clahe_options"):
            return
        self._update_stage_buttons()
        stage_indices = {
            "original": 0,
            "clahe": 1,
            "ridge": 2,
            "clean": 3,
            "seed": 4,
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
            "seed": self.display_seed_button,
        }
        for stage, button in buttons.items():
            button.setChecked(stage == self._display_stage)

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
            self._run_region()
            return
        self._run_preprocessing()

    def _show_display_stage(self) -> None:
        if self._display_stage == "original":
            if self._region is None:
                self._run_region()
            if self._region is not None:
                self.display_preview.set_array(self._region["display_rgb"])
            return
        if self._preprocess_preview is None:
            self.display_preview.set_array(None)
            return
        if self._display_stage == "clahe":
            self.display_preview.set_array(self._preprocess_preview["enhanced"])
        elif self._display_stage == "ridge":
            self.display_preview.set_array(display_image(self._preprocess_preview["ridges"], mode="magma"))
        elif self._display_stage == "clean":
            self.display_preview.set_array(self._preprocess_preview["candidate_clean"])
        elif self._display_stage == "seed":
            self.display_preview.set_array(self._preprocess_preview["display_mask"])

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
