from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
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
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ..core.alignment import AlignmentPoint


class AlignmentCanvas(QGraphicsView):
    point_clicked = Signal(float, float)
    zoom_changed = Signal(int)
    cursor_changed = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setScene(QGraphicsScene(self))
        self._items: dict[str, QGraphicsPixmapItem] = {}
        self._image_bytes: dict[str, bytes] = {}
        self._image_shapes: dict[str, tuple[int, int]] = {}
        self._points: dict[str, list[AlignmentPoint]] = {"dic": [], "ebsd": []}
        self._active_kind = "dic"
        self._zoom_level = 0
        self._zoom_factor = 1.0
        self._highlight: tuple[str, int] | None = None
        self._view_states: dict[str, tuple[object, int, int, int, float]] = {}
        self._space_pan_active = False
        self._panning = False
        self._twinkle_on = False
        self._twinkle_timer = QTimer(self)
        self._twinkle_timer.setInterval(450)
        self._twinkle_timer.timeout.connect(self._advance_twinkle)
        self._twinkle_timer.start()

    @property
    def active_kind(self) -> str:
        return self._active_kind

    @property
    def zoom_factor(self) -> float:
        return self._zoom_factor

    def set_image(self, kind: str, image_rgb: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(image_rgb)
        image_bytes = contiguous.tobytes()
        h, w, _ = contiguous.shape
        qimage = QImage(
            image_bytes,
            w,
            h,
            w * 3,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage)
        if kind in self._items:
            self._items[kind].setPixmap(pixmap)
        else:
            self._items[kind] = self.scene().addPixmap(pixmap)
        self._image_bytes[kind] = image_bytes
        self._image_shapes[kind] = (h, w)
        self.set_active_kind(kind)

    def set_points(self, kind: str, points: list[AlignmentPoint]) -> None:
        self._points[kind] = list(points)
        self.viewport().update()

    def set_active_kind(self, kind: str) -> None:
        if self.active_image_loaded():
            self._save_view_state(self._active_kind)
        self._active_kind = kind
        for item_kind, item in self._items.items():
            item.setVisible(True)
            item.setOpacity(1.0 if item_kind == kind else 0.30)
            item.setZValue(2 if item_kind == kind else 1)
        if kind in self._image_shapes:
            h, w = self._image_shapes[kind]
            self.scene().setSceneRect(0, 0, w, h)
            if kind in self._view_states:
                self._restore_view_state(kind)
            else:
                self.fit_to_window()
        self.viewport().update()

    def set_highlighted_point(self, kind: str | None, point_id: int | None) -> None:
        self._highlight = (kind, point_id) if kind is not None and point_id is not None else None
        if self._highlight is not None:
            point = self._point_by_id(kind, point_id)
            if point is not None:
                self.centerOn(point.x, point.y)
        self.viewport().update()

    def active_image_loaded(self) -> bool:
        return self._active_kind in self._image_shapes

    def zoom_in(self) -> None:
        if self._cursor_is_over_viewport():
            self._apply_zoom(1)

    def zoom_out(self) -> None:
        if self._cursor_is_over_viewport():
            self._apply_zoom(-1)

    def fit_to_window(self) -> None:
        if not self.active_image_loaded():
            return
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = 0
        self._zoom_factor = 1.0
        self.zoom_changed.emit(self._zoom_level)

    def drawForeground(self, painter: QPainter, rect) -> None:
        super().drawForeground(painter, rect)
        view_scale = max(abs(painter.transform().m11()), 1e-6)
        marker_radius = 11.0 / view_scale
        marker_border = 3.0 / view_scale
        text_size = 12.0 / view_scale
        halo_radius = (26.0 if self._twinkle_on else 20.0) / view_scale
        halo_width = (5.0 if self._twinkle_on else 3.0) / view_scale
        for kind, points in self._points.items():
            if kind not in self._image_shapes:
                continue
            if kind != self._active_kind and kind in self._items and self._items[kind].opacity() <= 0:
                continue
            fill_color = QColor(220, 0, 0) if kind == "dic" else QColor(0, 0, 0)
            border_color = QColor(255, 255, 255)
            if kind != self._active_kind:
                fill_color.setAlpha(135)
                border_color.setAlpha(160)
            painter.setPen(QPen(border_color, marker_border))
            painter.setBrush(QBrush(fill_color))
            font = painter.font()
            font.setPointSizeF(text_size)
            painter.setFont(font)
            for point in points:
                highlighted = self._highlight == (kind, point.id)
                if highlighted:
                    halo_color = QColor(255, 235, 0) if self._twinkle_on else QColor(255, 255, 255)
                    halo_pen = QPen(halo_color, halo_width)
                    painter.setPen(halo_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(QPointF(point.x, point.y), halo_radius, halo_radius)
                    painter.setPen(QPen(border_color, marker_border))
                    painter.setBrush(QBrush(fill_color))
                painter.drawEllipse(QPointF(point.x, point.y), marker_radius, marker_radius)
                painter.setPen(QPen(border_color, max(1.5 / view_scale, marker_border * 0.6)))
                painter.drawText(
                    QPointF(point.x + marker_radius + (4.0 / view_scale), point.y - marker_radius),
                    str(point.id),
                )

    def mousePressEvent(self, event: QMouseEvent) -> None:
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
        if event.button() == Qt.MouseButton.LeftButton and self.active_image_loaded():
            point = self.mapToScene(event.position().toPoint())
            self.point_clicked.emit(float(point.x()), float(point.y()))
            return
        super().mouseReleaseEvent(event)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def keyPressEvent(self, event: QKeyEvent) -> None:
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

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        point = self.mapToScene(event.position().toPoint())
        self.cursor_changed.emit(float(point.x()), float(point.y()))
        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.active_image_loaded():
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self._apply_zoom(1)
        elif delta < 0:
            self._apply_zoom(-1)

    def _apply_zoom(self, direction: int) -> None:
        if not self.active_image_loaded():
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
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scale(factor, factor)
        self._zoom_factor *= factor
        self.zoom_changed.emit(self._zoom_level)

    def _cursor_is_over_viewport(self) -> bool:
        return self.viewport().rect().contains(self.viewport().mapFromGlobal(self.cursor().pos()))

    def _save_view_state(self, kind: str) -> None:
        self._view_states[kind] = (
            self.transform(),
            self.horizontalScrollBar().value(),
            self.verticalScrollBar().value(),
            self._zoom_level,
            self._zoom_factor,
        )

    def _restore_view_state(self, kind: str) -> None:
        transform, h_value, v_value, zoom_level, zoom_factor = self._view_states[kind]
        self.setTransform(transform)
        self.horizontalScrollBar().setValue(h_value)
        self.verticalScrollBar().setValue(v_value)
        self._zoom_level = zoom_level
        self._zoom_factor = zoom_factor
        self.zoom_changed.emit(self._zoom_level)

    def _point_by_id(self, kind: str | None, point_id: int | None) -> AlignmentPoint | None:
        if kind is None or point_id is None:
            return None
        for point in self._points.get(kind, []):
            if point.id == point_id:
                return point
        return None

    def _advance_twinkle(self) -> None:
        if self._highlight is None:
            return
        self._twinkle_on = not self._twinkle_on
        self.viewport().update()
