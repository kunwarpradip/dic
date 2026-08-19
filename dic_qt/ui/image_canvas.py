from __future__ import annotations

import time
from uuid import UUID

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QPolygonF, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ..core.algorithm import is_point_in_line
from ..core.models import DicLine, Point


class ImageCanvas(QGraphicsView):
    seed_clicked = Signal(int, int)
    empty_clicked = Signal(int, int)
    image_mouse_moved = Signal(int, int)
    line_toggled = Signal(object)
    cut_completed = Signal(object, object)
    area_erase_completed = Signal(object)
    pixel_edit_started = Signal()
    pixel_edit_requested = Signal(object, object, int)
    pixel_edit_finished = Signal()
    zoom_changed = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_rgb: np.ndarray | None = None
        self._image_bytes: bytes | None = None
        self._lines: list[DicLine] = []
        self._visible_ids: set[UUID] = set()
        self._create_mode = True
        self._cut_mode = False
        self._edit_mode = "select"
        self._brush_radius = 2
        self._cut_start: QPointF | None = None
        self._area_points: list[QPointF] = []
        self._zoom_level = 0
        self._zoom_factor = 1.0
        self._highlight_line: DicLine | None = None
        self._boundary_mask: np.ndarray | None = None
        self._boundary_visible = False
        self._last_mouse_pixel: Point | None = None
        self._last_mouse_emit_time = 0.0
        self._line_by_id: dict[UUID, DicLine] = {}
        self._visible_lines: list[DicLine] = []
        self._pixel_line_index: dict[int, UUID] = {}
        self._overlay_pixmap: QPixmap | None = None
        self._overlay_bytes: bytes | None = None
        self._boundary_pixmap: QPixmap | None = None
        self._boundary_bytes: bytes | None = None
        self._live_edit_line: DicLine | None = None
        self._space_pan_active = False
        self._locator_line: DicLine | None = None
        self._locator_visible = False
        self._locator_ticks_remaining = 0
        self._locator_timer = QTimer(self)
        self._locator_timer.setInterval(180)
        self._locator_timer.timeout.connect(self._blink_locator)

    @property
    def zoom_level(self) -> int:
        return self._zoom_level

    @property
    def zoom_factor(self) -> float:
        return self._zoom_factor

    def set_image(self, image_rgb: np.ndarray) -> None:
        self._image_rgb = image_rgb
        self._overlay_pixmap = None
        self._overlay_bytes = None
        self._boundary_pixmap = None
        self._boundary_bytes = None
        contiguous = np.ascontiguousarray(image_rgb)
        self._image_bytes = contiguous.tobytes()
        h, w, _ = contiguous.shape
        qimage = QImage(
            self._image_bytes,
            w,
            h,
            w * 3,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage)
        self.scene().clear()
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self.scene().setSceneRect(0, 0, w, h)
        self.fit_to_window()

    def set_lines(self, lines: list[DicLine], visible_ids: set[UUID]) -> None:
        self._lines = [line for line in lines if isinstance(line, DicLine)]
        self._visible_ids = set(visible_ids)
        self._line_by_id = {line.id: line for line in self._lines}
        self._visible_lines = [line for line in self._lines if line.id in self._visible_ids]
        self._rebuild_pixel_line_index()
        self._rebuild_event_overlay()
        self._live_edit_line = None
        if self._highlight_line is not None and self._highlight_line.id not in self._line_by_id:
            self._highlight_line = None
        self.viewport().update()

    def remove_lines(self, line_ids: set[UUID], visible_ids: set[UUID]) -> None:
        ids = set(line_ids)
        if not ids:
            return
        self._lines = [line for line in self._lines if line.id not in ids]
        self._visible_ids = set(visible_ids)
        self._line_by_id = {line.id: line for line in self._lines}
        self._visible_lines = [line for line in self._lines if line.id in self._visible_ids]
        self._rebuild_pixel_line_index()
        self._rebuild_event_overlay()
        if self._highlight_line is not None and self._highlight_line.id in ids:
            self._highlight_line = None
        if self._live_edit_line is not None and self._live_edit_line.id in ids:
            self._live_edit_line = None
        self.viewport().update()

    def set_boundary_mask(self, mask: np.ndarray | None) -> None:
        self._boundary_mask = mask.astype(bool, copy=False) if mask is not None else None
        self._rebuild_boundary_overlay()
        self.viewport().update()

    def set_boundary_visible(self, visible: bool) -> None:
        self._boundary_visible = bool(visible)
        self.viewport().update()

    def set_create_mode(self, enabled: bool) -> None:
        self._create_mode = enabled

    def set_cut_mode(self, enabled: bool) -> None:
        self._cut_mode = enabled
        self._cut_start = None

    def set_edit_mode(self, mode: str) -> None:
        self._edit_mode = mode if mode in {"select", "add", "erase", "area_erase"} else "select"
        self._area_points = []
        self._update_brush_cursor()

    def set_brush_radius(self, radius: int) -> None:
        self._brush_radius = max(1, int(radius))
        self._update_brush_cursor()

    def set_live_edit_line(self, line: DicLine | None) -> None:
        self._live_edit_line = line
        self._highlight_line = line
        self.viewport().update()

    def locate_line(self, line: DicLine | None, center: bool = True) -> None:
        if line is None or not line.points:
            self._locator_timer.stop()
            self._locator_line = None
            self._locator_visible = False
            self._locator_ticks_remaining = 0
            self.viewport().update()
            return
        self._locator_line = line
        self._locator_visible = True
        self._locator_ticks_remaining = 18
        if center:
            x0, y0, x1, y1 = self._line_bounds(line)
            self.centerOn((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        self._locator_timer.start()
        self.viewport().update()

    def zoom_in(self) -> None:
        if not self._cursor_is_over_viewport():
            return
        self._apply_zoom(1, QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def zoom_out(self) -> None:
        if not self._cursor_is_over_viewport():
            return
        self._apply_zoom(-1, QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def fit_to_window(self) -> None:
        if self._image_rgb is None:
            return
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = 0
        self._zoom_factor = 1.0
        self.zoom_changed.emit(self._zoom_level)

    def drawForeground(self, painter: QPainter, rect) -> None:
        super().drawForeground(painter, rect)
        if self._image_rgb is None:
            return

        if self._boundary_visible and self._boundary_pixmap is not None:
            self._draw_cached_pixmap_region(painter, rect, self._boundary_pixmap)

        if self._overlay_pixmap is not None:
            self._draw_cached_pixmap_region(painter, rect, self._overlay_pixmap)

        if self._highlight_line is not None:
            point_pen = QPen(Qt.GlobalColor.yellow)
            point_pen.setWidth(2)
            painter.setPen(point_pen)
            for p in self._highlight_line.points:
                painter.drawPoint(p.x, p.y)

        if self._live_edit_line is not None and self._live_edit_line is not self._highlight_line:
            live_pen = QPen(Qt.GlobalColor.yellow)
            live_pen.setWidth(2)
            painter.setPen(live_pen)
            for p in self._live_edit_line.points:
                painter.drawPoint(p.x, p.y)

        if self._locator_line is not None and self._locator_visible:
            self._draw_locator(painter, self._locator_line)

        if self._cut_start is not None and self._cut_mode:
            cursor = self.mapToScene(self.mapFromGlobal(self.cursor().pos()))
            cpen = QPen(Qt.GlobalColor.green)
            cpen.setWidth(3)
            painter.setPen(cpen)
            painter.drawLine(self._cut_start, cursor)

        if self._area_points and self._edit_mode == "area_erase":
            polygon = QPolygonF(self._area_points)
            painter.setBrush(QColor(20, 184, 166, 35))
            area_pen = QPen(QColor(20, 184, 166))
            area_pen.setWidth(3)
            painter.setPen(area_pen)
            if len(polygon) > 1:
                painter.drawPolygon(polygon)
                painter.drawPolyline(polygon)
                painter.drawLine(self._area_points[-1], self._area_points[0])
            else:
                painter.drawPoint(self._area_points[0])
            painter.setBrush(Qt.BrushStyle.NoBrush)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._space_pan_active:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._cut_mode:
            self._cut_start = self.mapToScene(event.position().toPoint())
            self.viewport().update()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._edit_mode in {"add", "erase"}:
            world = self.mapToScene(event.position().toPoint())
            self.pixel_edit_started.emit()
            self.pixel_edit_requested.emit(
                self._edit_mode,
                Point(int(world.x()), int(world.y())),
                self._brush_radius,
            )
            return
        if event.button() == Qt.MouseButton.LeftButton and self._edit_mode == "area_erase":
            start = self.mapToScene(event.position().toPoint())
            self._area_points = [start]
            self.viewport().update()
            return
        if event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.RightButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._space_pan_active:
            super().mouseReleaseEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._edit_mode in {"add", "erase"}:
            self.pixel_edit_finished.emit()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._edit_mode == "area_erase" and self._area_points:
            end = self.mapToScene(event.position().toPoint())
            self._append_area_point(end, force=True)
            polygon_points = [Point(int(point.x()), int(point.y())) for point in self._area_points]
            self.area_erase_completed.emit(polygon_points)
            self._area_points = []
            self.viewport().update()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._cut_mode and self._cut_start is not None:
            end = self.mapToScene(event.position().toPoint())
            self.cut_completed.emit(
                Point(int(self._cut_start.x()), int(self._cut_start.y())),
                Point(int(end.x()), int(end.y())),
            )
            self._cut_start = None
            self.viewport().update()
            return
        if event.button() == Qt.MouseButton.LeftButton and not self._cut_mode:
            world = self.mapToScene(event.position().toPoint())
            point = Point(int(world.x()), int(world.y()))
            line = self._line_at_point(point)
            if line is not None:
                self.line_toggled.emit(line.id)
            elif self._create_mode:
                self.seed_clicked.emit(point.x, point.y)
            else:
                self.empty_clicked.emit(point.x, point.y)
            return
        super().mouseReleaseEvent(event)
        if not self._space_pan_active:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        world = self.mapToScene(event.position().toPoint())
        x, y = int(world.x()), int(world.y())
        point = Point(x, y)
        now = time.monotonic()
        if point != self._last_mouse_pixel and now - self._last_mouse_emit_time >= 0.033:
            self._last_mouse_pixel = point
            self._last_mouse_emit_time = now
            self.image_mouse_moved.emit(x, y)
        if event.buttons() & Qt.MouseButton.LeftButton and self._edit_mode in {"add", "erase"}:
            self.pixel_edit_requested.emit(self._edit_mode, point, self._brush_radius)
            return
        if event.buttons() & Qt.MouseButton.LeftButton and self._edit_mode == "area_erase":
            self._append_area_point(QPointF(x, y))
            self.viewport().update()
            return
        previous_highlight_id = self._highlight_line.id if self._highlight_line is not None else None
        self._highlight_line = self._line_at_point(point)
        current_highlight_id = self._highlight_line.id if self._highlight_line is not None else None
        if current_highlight_id != previous_highlight_id or self._cut_start is not None or self._area_points:
            self.viewport().update()
        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._image_rgb is None:
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self._apply_zoom(1, QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        elif delta < 0:
            self._apply_zoom(-1, QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def enterEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().enterEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan_active = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan_active = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._update_brush_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _line_at_point(self, point: Point) -> DicLine | None:
        if self._image_rgb is not None and self._pixel_line_index:
            h, w, _ = self._image_rgb.shape
            radius = 3
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy >= radius * radius:
                        continue
                    x = point.x + dx
                    y = point.y + dy
                    if 0 <= x < w and 0 <= y < h:
                        line_id = self._pixel_line_index.get(y * w + x)
                        if line_id is not None:
                            return self._line_by_id.get(line_id)
        return None

    def _append_area_point(self, point: QPointF, force: bool = False) -> None:
        if not self._area_points:
            self._area_points.append(point)
            return
        previous = self._area_points[-1]
        dx = point.x() - previous.x()
        dy = point.y() - previous.y()
        if force or dx * dx + dy * dy >= 4.0:
            self._area_points.append(point)

    def _blink_locator(self) -> None:
        if self._locator_line is None:
            self._locator_timer.stop()
            return
        self._locator_visible = not self._locator_visible
        self._locator_ticks_remaining -= 1
        if self._locator_ticks_remaining <= 0:
            self._locator_timer.stop()
            self._locator_line = None
            self._locator_visible = False
        self.viewport().update()

    def _draw_locator(self, painter: QPainter, line: DicLine) -> None:
        if not line.points:
            return
        x0, y0, x1, y1 = self._line_bounds(line)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        pad = max(8.0, min(48.0, max(x1 - x0, y1 - y0) * 0.25 + 6.0))
        rect = QRectF(x0 - pad, y0 - pad, (x1 - x0) + 2 * pad + 1, (y1 - y0) + 2 * pad + 1)

        outline = QPen(QColor(255, 255, 0))
        outline.setWidth(3)
        painter.setPen(outline)
        painter.drawRect(rect)
        painter.drawLine(QPointF(cx - pad, cy), QPointF(cx + pad, cy))
        painter.drawLine(QPointF(cx, cy - pad), QPointF(cx, cy + pad))

        point_pen = QPen(QColor(236, 72, 153))
        point_pen.setWidth(3)
        painter.setPen(point_pen)
        for point in line.points:
            painter.drawPoint(point.x, point.y)

    @staticmethod
    def _line_bounds(line: DicLine) -> tuple[int, int, int, int]:
        xs = [point.x for point in line.points]
        ys = [point.y for point in line.points]
        return min(xs), min(ys), max(xs), max(ys)

    def _rebuild_pixel_line_index(self) -> None:
        self._pixel_line_index = {}
        if self._image_rgb is None or not self._lines:
            return
        h, w, _ = self._image_rgb.shape
        for line in self._lines:
            for point in line.points:
                if 0 <= point.x < w and 0 <= point.y < h:
                    self._pixel_line_index[point.y * w + point.x] = line.id

    def _rebuild_event_overlay(self) -> None:
        self._overlay_pixmap = None
        self._overlay_bytes = None
        if self._image_rgb is None or not self._visible_lines:
            return
        h, w, _ = self._image_rgb.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        for line in self._visible_lines:
            if not line.points:
                continue
            xs = np.fromiter((point.x for point in line.points), dtype=np.int32, count=len(line.points))
            ys = np.fromiter((point.y for point in line.points), dtype=np.int32, count=len(line.points))
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            overlay[ys[valid], xs[valid]] = (255, 0, 0, 210)
        self._overlay_bytes = np.ascontiguousarray(overlay).tobytes()
        qimage = QImage(self._overlay_bytes, w, h, w * 4, QImage.Format.Format_RGBA8888)
        self._overlay_pixmap = QPixmap.fromImage(qimage)

    def _rebuild_boundary_overlay(self) -> None:
        self._boundary_pixmap = None
        self._boundary_bytes = None
        if self._boundary_mask is None:
            return
        h, w = self._boundary_mask.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        rows, cols = np.where(self._boundary_mask)
        overlay[rows, cols] = (37, 99, 235, 210)
        self._boundary_bytes = np.ascontiguousarray(overlay).tobytes()
        qimage = QImage(self._boundary_bytes, w, h, w * 4, QImage.Format.Format_RGBA8888)
        self._boundary_pixmap = QPixmap.fromImage(qimage)

    @staticmethod
    def _draw_cached_pixmap_region(painter: QPainter, rect, pixmap: QPixmap) -> None:
        source = QRectF(
            max(0.0, float(rect.left()) - 1.0),
            max(0.0, float(rect.top()) - 1.0),
            max(1.0, float(rect.width()) + 2.0),
            max(1.0, float(rect.height()) + 2.0),
        )
        painter.drawPixmap(source, pixmap, source)

    def _apply_zoom(self, direction: int, anchor: QGraphicsView.ViewportAnchor) -> None:
        if self._image_rgb is None:
            return
        if direction > 0 and self._zoom_level < 20:
            increment = 2 if self._zoom_level <= 4 else 1
            self._zoom_level = min(20, self._zoom_level + increment)
            factor = 1.25
        elif direction < 0 and self._zoom_level > 0:
            increment = 2 if self._zoom_level - 2 <= 4 else 1
            self._zoom_level = max(0, self._zoom_level - increment)
            factor = 0.8
        else:
            return
        if self._zoom_level == 0:
            self.fit_to_window()
            return
        self.setTransformationAnchor(anchor)
        self.scale(factor, factor)
        self._zoom_factor *= factor
        self.zoom_changed.emit(self._zoom_level)

    def _cursor_is_over_viewport(self) -> bool:
        return self.viewport().rect().contains(self.viewport().mapFromGlobal(self.cursor().pos()))

    def _update_brush_cursor(self) -> None:
        if self._edit_mode == "area_erase":
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if self._edit_mode not in {"add", "erase"}:
            self.unsetCursor()
            return
        size = max(12, min(96, int(self._brush_radius * 2 + 8)))
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(220, 38, 38) if self._edit_mode == "erase" else QColor(37, 99, 235)
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawEllipse(4, 4, size - 8, size - 8)
        painter.end()
        self.setCursor(QCursor(pixmap, size // 2, size // 2))
