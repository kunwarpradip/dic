from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel


class PreviewPanel(QLabel):
    def __init__(self, size: int = 201) -> None:
        super().__init__()
        self.preview_size = size
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Preview")

    def update_preview(self, image_rgb: np.ndarray | None, x: int, y: int) -> None:
        if image_rgb is None:
            self.setText("Preview")
            return
        h, w, _ = image_rgb.shape
        half = self.preview_size // 2
        x0 = max(0, min(x - half, w - self.preview_size))
        y0 = max(0, min(y - half, h - self.preview_size))
        crop = image_rgb[y0 : y0 + self.preview_size, x0 : x0 + self.preview_size]
        if crop.shape[0] != self.preview_size or crop.shape[1] != self.preview_size:
            return
        contiguous = np.ascontiguousarray(crop)
        qimage = QImage(
            contiguous.data,
            self.preview_size,
            self.preview_size,
            self.preview_size * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        painter = QPainter(qimage)
        painter.setPen(Qt.GlobalColor.green)
        c = self.preview_size // 2
        painter.drawLine(c - 2, c, c + 2, c)
        painter.drawLine(c, c - 2, c, c + 2)
        painter.end()
        self.setPixmap(QPixmap.fromImage(qimage))
