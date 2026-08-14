from __future__ import annotations

from uuid import UUID

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ..core.algorithm import is_point_in_line
from ..core.models import DicLine, Point


class ImageCanvas(QGraphicsView):
    seed_clicked = Signal(int, int)
    empty_clicked = Signal(int, int)
    image_mouse_moved = Signal(int, int)
    line_toggled = Signal(object)
    cut_completed = Signal(object, object)
    pixel_edit_requested = Signal(object, object, int)
    zoom_changed = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
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
        self._zoom_level = 0
        self._zoom_factor = 1.0
        self._highlight_line: DicLine | None = None
        self._boundary_mask: np.ndarray | None = None
        self._boundary_visible = False

    @property
    def zoom_level(self) -> int:
        return self._zoom_level

    @property
    def zoom_factor(self) -> float:
        return self._zoom_factor

    def set_image(self, image_rgb: np.ndarray) -> None:
        self._image_rgb = image_rgb
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
        self.viewport().update()

    def set_boundary_mask(self, mask: np.ndarray | None) -> None:
        self._boundary_mask = mask.astype(bool, copy=False) if mask is not None else None
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
        self._edit_mode = mode if mode in {"select", "add", "erase"} else "select"
        self._update_brush_cursor()

    def set_brush_radius(self, radius: int) -> None:
        self._brush_radius = max(1, int(radius))
        self._update_brush_cursor()

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

        if self._boundary_visible and self._boundary_mask is not None:
            h, w = self._boundary_mask.shape[:2]
            x0 = max(0, int(rect.left()) - 1)
            x1 = min(w, int(rect.right()) + 2)
            y0 = max(0, int(rect.top()) - 1)
            y1 = min(h, int(rect.bottom()) + 2)
            if x1 > x0 and y1 > y0:
                rows, cols = np.where(self._boundary_mask[y0:y1, x0:x1])
                boundary_pen = QPen(QColor(37, 99, 235))
                boundary_pen.setWidth(1)
                painter.setPen(boundary_pen)
                for row, col in zip(rows, cols):
                    painter.drawPoint(int(x0 + col), int(y0 + row))

        pen = QPen(Qt.GlobalColor.red)
        pen.setWidth(1)
        painter.setPen(pen)
        for line in self._lines:
            if not isinstance(line, DicLine):
                continue
            if line.id not in self._visible_ids:
                continue
            for p in line.points:
                painter.drawPoint(p.x, p.y)

        if self._highlight_line is not None:
            point_pen = QPen(Qt.GlobalColor.yellow)
            point_pen.setWidth(2)
            painter.setPen(point_pen)
            for p in self._highlight_line.points:
                painter.drawPoint(p.x, p.y)

        if self._cut_start is not None and self._cut_mode:
            cursor = self.mapToScene(self.mapFromGlobal(self.cursor().pos()))
            cpen = QPen(Qt.GlobalColor.green)
            cpen.setWidth(3)
            painter.setPen(cpen)
            painter.drawLine(self._cut_start, cursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._cut_mode:
            self._cut_start = self.mapToScene(event.position().toPoint())
            self.viewport().update()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._edit_mode in {"add", "erase"}:
            world = self.mapToScene(event.position().toPoint())
            self.pixel_edit_requested.emit(
                self._edit_mode,
                Point(int(world.x()), int(world.y())),
                self._brush_radius,
            )
            return
        if event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.RightButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._edit_mode in {"add", "erase"}:
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
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        world = self.mapToScene(event.position().toPoint())
        x, y = int(world.x()), int(world.y())
        self.image_mouse_moved.emit(x, y)
        if event.buttons() & Qt.MouseButton.LeftButton and self._edit_mode in {"add", "erase"}:
            self.pixel_edit_requested.emit(self._edit_mode, Point(x, y), self._brush_radius)
            return
        self._highlight_line = self._line_at_point(Point(x, y))
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

    def _line_at_point(self, point: Point) -> DicLine | None:
        for line in self._lines:
            if not isinstance(line, DicLine):
                continue
            if is_point_in_line(point, line, 3.0):
                return line
        return None

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
