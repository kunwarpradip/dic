from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core.alignment import (
    AlignmentPoint,
    alignment_transformation_metadata,
    align_ebsd_to_dic,
    load_alignment_points,
    paired_points,
    save_aligned_image,
    save_alignment_metadata,
    save_alignment_points,
)
from ..core.image_io import load_image_data
from ..core.repository import app_data_dir_for_image
from .alignment_canvas import AlignmentCanvas


class AlignmentPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.dic_path: Path | None = None
        self.ebsd_path: Path | None = None
        self.dic_rgb: np.ndarray | None = None
        self.ebsd_rgb: np.ndarray | None = None
        self.points: dict[str, list[AlignmentPoint]] = {"dic": [], "ebsd": []}
        self._refreshing_lists = False
        self._selected_point: tuple[str, int] | None = None

        self.canvas = AlignmentCanvas()
        self.load_dic_button = QPushButton("Load DIC file")
        self.load_ebsd_button = QPushButton("Load EBSD file")
        self.toggle_button = QPushButton("Toggle DIC/EBSD")
        self.run_button = QPushButton("Run Alignment")
        self.run_button.setEnabled(False)
        self.save_transform_button = QPushButton("Save Transformation")
        self.save_transform_button.setEnabled(False)
        self.paste_points_button = QPushButton("Load Control Points")
        self.clear_active_button = QPushButton("Clear Active Points")
        self.zoom_label = QLabel("Zoom: -")
        self.active_label = QLabel("Active: DIC")
        self.cursor_label = QLabel("Cursor: -")
        self.status_label = QLabel("Load DIC and EBSD images")
        self.dic_list = QListWidget()
        self.ebsd_list = QListWidget()

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.load_dic_button)
        toolbar.addWidget(self.load_ebsd_button)
        toolbar.addWidget(self.toggle_button)
        toolbar.addWidget(self.run_button)
        toolbar.addWidget(self.save_transform_button)
        toolbar.addWidget(self.paste_points_button)
        toolbar.addWidget(self.clear_active_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.active_label)
        toolbar.addWidget(self.zoom_label)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(toolbar)
        left_layout.addWidget(self.canvas, 1)
        left_layout.addWidget(self.cursor_label)
        left_layout.addWidget(self.status_label)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("DIC Points"))
        right_layout.addWidget(self.dic_list)
        right_layout.addWidget(QLabel("EBSD Points"))
        right_layout.addWidget(self.ebsd_list)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        self.load_dic_button.clicked.connect(self.load_dic)
        self.load_ebsd_button.clicked.connect(self.load_ebsd)
        self.toggle_button.clicked.connect(self.toggle_active_image)
        self.run_button.clicked.connect(self.run_alignment)
        self.save_transform_button.clicked.connect(self.save_transformation)
        self.paste_points_button.clicked.connect(self.load_control_points_for_active_image)
        self.clear_active_button.clicked.connect(self.clear_active_points)
        self.canvas.point_clicked.connect(self.handle_point_clicked)
        self.canvas.cursor_changed.connect(self.update_cursor)
        self.canvas.zoom_changed.connect(self.update_zoom_label)
        self.dic_list.itemSelectionChanged.connect(
            lambda: self.handle_point_list_selection("dic")
        )
        self.ebsd_list.itemSelectionChanged.connect(
            lambda: self.handle_point_list_selection("ebsd")
        )

    def load_dic(self) -> None:
        path = self._choose_image("Choose DIC image")
        if path is None:
            return
        loaded = load_image_data(path)
        self.dic_path = path
        self.dic_rgb = loaded.display_rgb
        self.points["dic"] = load_alignment_points(path)
        self.canvas.set_image("dic", self.dic_rgb)
        self.active_label.setText("Active: DIC")
        self.canvas.set_points("dic", self.points["dic"])
        self.refresh_point_lists()
        self.status_label.setText(f"Loaded DIC {path.name} | {loaded.display_note}")

    def load_ebsd(self) -> None:
        path = self._choose_image("Choose EBSD image")
        if path is None:
            return
        loaded = load_image_data(path)
        self.ebsd_path = path
        self.ebsd_rgb = loaded.display_rgb
        self.points["ebsd"] = load_alignment_points(path)
        self.canvas.set_image("ebsd", self.ebsd_rgb)
        self.active_label.setText("Active: EBSD")
        self.canvas.set_points("ebsd", self.points["ebsd"])
        self.refresh_point_lists()
        self.status_label.setText(f"Loaded EBSD {path.name} | {loaded.display_note}")

    def toggle_active_image(self) -> None:
        new_kind = "ebsd" if self.canvas.active_kind == "dic" else "dic"
        self.canvas.set_active_kind(new_kind)
        self.active_label.setText(f"Active: {new_kind.upper()}")
        self.status_label.setText(
            f"Active image is {new_kind.upper()} | click to add/remove control points"
        )

    def handle_point_clicked(self, x: float, y: float) -> None:
        kind = self.canvas.active_kind
        image_path = self._path_for_kind(kind)
        if image_path is None:
            return
        points = self.points[kind]
        closest = self._closest_point(points, x, y, tolerance=55.0)
        if closest is not None:
            point_id = closest.id
            self.points[kind] = [
                point for point in self.points[kind] if point.id != point_id
            ]
            save_alignment_points(image_path, self.points[kind])
            self.canvas.set_points(kind, self.points[kind])
            self.status_label.setText(f"Removed {kind.upper()} control point {point_id}")
            if self._selected_point == (kind, point_id):
                self._selected_point = None
                self.canvas.set_highlighted_point(None, None)
        else:
            next_id = self._next_point_id(kind)
            new_point = AlignmentPoint(id=next_id, x=float(x), y=float(y))
            self.points[kind].append(new_point)
            save_alignment_points(image_path, self.points[kind])
            self.canvas.set_points(kind, self.points[kind])
            self._selected_point = (kind, next_id)
            self.canvas.set_highlighted_point(kind, next_id)
            self.status_label.setText(
                f"Added {kind.upper()} point {next_id} at x={x:.1f}, y={y:.1f}"
            )
        self.refresh_point_lists()

    def run_alignment(self) -> None:
        if self.dic_path is None or self.ebsd_path is None:
            self.status_label.setText("Load both DIC and EBSD images before alignment")
            return
        if self.dic_rgb is None or self.ebsd_rgb is None:
            return
        try:
            result = align_ebsd_to_dic(
                self.ebsd_rgb,
                self.dic_rgb,
                self.points["ebsd"],
                self.points["dic"],
            )
        except Exception as exc:
            self.status_label.setText(f"Alignment failed: {exc}")
            return

        output_path = app_data_dir_for_image(self.dic_path) / "OUTPUT_FROM_DIC_GUI.tif"
        save_aligned_image(output_path, result.aligned_rgb)
        self.status_label.setText(
            "Saved alignment\n"
            f"Image: {output_path}\n"
            f"Matched points: {len(result.point_ids)}"
        )

    def save_transformation(self) -> None:
        if self.dic_path is None or self.ebsd_path is None:
            self.status_label.setText("Load both DIC and EBSD images before saving transformation")
            return
        if self.dic_rgb is None or self.ebsd_rgb is None:
            return
        try:
            metadata = alignment_transformation_metadata(
                self.ebsd_rgb,
                self.dic_rgb,
                self.points["ebsd"],
                self.points["dic"],
            )
        except Exception as exc:
            self.status_label.setText(f"Could not save transformation: {exc}")
            return

        output_path = app_data_dir_for_image(self.dic_path) / "alignment_transformation.json"
        metadata = {
            **metadata,
            "dic_image_path": str(self.dic_path),
            "ebsd_image_path": str(self.ebsd_path),
            "saved_file": str(output_path),
        }
        save_alignment_metadata(output_path, metadata)
        self.status_label.setText(
            "Saved transformation\n"
            f"File: {output_path}\n"
            f"Matched points: {len(metadata['matched_point_ids'])}\n"
            "Includes control points and x/y polynomial coefficients"
        )

    def load_control_points_for_active_image(self) -> None:
        kind = self.canvas.active_kind
        image_path = self._path_for_kind(kind)
        if image_path is None:
            self.status_label.setText(f"Load a {kind.upper()} image before pasting points")
            return

        dialog = LoadControlPointsDialog(kind.upper(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            points_xy = parse_alignment_points_text(dialog.points_text())
        except ValueError as exc:
            QMessageBox.warning(self, "Could Not Parse Points", str(exc))
            return

        if dialog.replace_existing():
            new_points = [
                AlignmentPoint(id=index, x=x, y=y)
                for index, (x, y) in enumerate(points_xy, start=1)
            ]
        else:
            next_id = self._next_point_id(kind)
            new_points = list(self.points[kind])
            for offset, (x, y) in enumerate(points_xy):
                new_points.append(AlignmentPoint(id=next_id + offset, x=x, y=y))

        self.points[kind] = new_points
        save_alignment_points(image_path, self.points[kind])
        self.canvas.set_points(kind, self.points[kind])
        self.refresh_point_lists()
        self.status_label.setText(
            f"Loaded {len(points_xy)} pasted {kind.upper()} points"
            + ("; replaced existing points" if dialog.replace_existing() else "; appended to existing points")
        )

    def clear_active_points(self) -> None:
        kind = self.canvas.active_kind
        path = self._path_for_kind(kind)
        if path is None:
            return
        self.points[kind] = []
        save_alignment_points(path, self.points[kind])
        self.canvas.set_points(kind, [])
        self.refresh_point_lists()
        self.status_label.setText(f"Cleared {kind.upper()} control points")

    def refresh_point_lists(self) -> None:
        self._refreshing_lists = True
        self.dic_list.clear()
        self.ebsd_list.clear()
        for point in sorted(self.points["dic"], key=lambda p: p.id):
            self.dic_list.addItem(f"{point.id}: x={point.x:.1f}, y={point.y:.1f}")
            self.dic_list.item(self.dic_list.count() - 1).setData(
                Qt.ItemDataRole.UserRole,
                point.id,
            )
        for point in sorted(self.points["ebsd"], key=lambda p: p.id):
            self.ebsd_list.addItem(f"{point.id}: x={point.x:.1f}, y={point.y:.1f}")
            self.ebsd_list.item(self.ebsd_list.count() - 1).setData(
                Qt.ItemDataRole.UserRole,
                point.id,
            )
        self._restore_selected_list_item()
        self._refreshing_lists = False
        _, _, ids = paired_points(self.points["dic"], self.points["ebsd"])
        can_transform = len(ids) >= 6
        self.run_button.setEnabled(can_transform)
        self.save_transform_button.setEnabled(can_transform)

    def handle_point_list_selection(self, kind: str) -> None:
        if self._refreshing_lists:
            return
        list_widget = self.dic_list if kind == "dic" else self.ebsd_list
        item = list_widget.currentItem()
        if item is None:
            return
        point_id = int(item.data(Qt.ItemDataRole.UserRole))
        self._selected_point = (kind, point_id)
        self.canvas.set_active_kind(kind)
        self.active_label.setText(f"Active: {kind.upper()}")
        self.canvas.set_highlighted_point(kind, point_id)
        self.status_label.setText(f"Selected {kind.upper()} point {point_id}")

    def update_zoom_label(self, zoom_level: int) -> None:
        if zoom_level == 0:
            self.zoom_label.setText("Zoom: Fit")
        else:
            self.zoom_label.setText(f"Zoom: {self.canvas.zoom_factor * 100:.0f}%")

    def update_cursor(self, x: float, y: float) -> None:
        self.cursor_label.setText(f"Cursor: x={x:.1f}, y={y:.1f}")

    def _choose_image(self, title: str) -> Path | None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            str(Path.cwd()),
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)",
        )
        if not path:
            return None
        return Path(path)

    def _path_for_kind(self, kind: str) -> Path | None:
        return self.dic_path if kind == "dic" else self.ebsd_path

    def _next_point_id(self, kind: str) -> int:
        existing = {point.id for point in self.points[kind]}
        next_id = 1
        while next_id in existing:
            next_id += 1
        return next_id

    def _restore_selected_list_item(self) -> None:
        if self._selected_point is None:
            return
        kind, point_id = self._selected_point
        list_widget = self.dic_list if kind == "dic" else self.ebsd_list
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole)) == point_id:
                item.setSelected(True)
                list_widget.setCurrentItem(item)
                return

    @staticmethod
    def _closest_point(
        points: list[AlignmentPoint],
        x: float,
        y: float,
        tolerance: float,
    ) -> AlignmentPoint | None:
        best: AlignmentPoint | None = None
        best_distance = tolerance
        for point in points:
            distance = ((point.x - x) ** 2 + (point.y - y) ** 2) ** 0.5
            if distance <= best_distance:
                best = point
                best_distance = distance
        return best


class LoadControlPointsDialog(QDialog):
    def __init__(self, image_label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Load {image_label} Control Points")
        self.resize(560, 420)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText(
            "Paste one point per row:\n"
            "x<TAB>y\n"
            "x y\n"
            "x,y\n\n"
            "Example:\n"
            "9922.5    5633.5\n"
            "530.5     5737.5"
        )
        self.replace_checkbox = QCheckBox("Replace existing active-image points")
        self.replace_checkbox.setChecked(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"Paste {image_label} control points. Point IDs will be assigned by row order."
            )
        )
        layout.addWidget(self.editor, 1)
        layout.addWidget(self.replace_checkbox)
        layout.addWidget(buttons)

    def points_text(self) -> str:
        return self.editor.toPlainText()

    def replace_existing(self) -> bool:
        return self.replace_checkbox.isChecked()


def parse_alignment_points_text(text: str) -> list[tuple[float, float]]:
    text = text.strip()
    if not text:
        raise ValueError("Paste at least one x/y point row.")

    json_points = _parse_alignment_points_json(text)
    if json_points is not None:
        return json_points

    points: list[tuple[float, float]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = [part for part in re.split(r"[\s,;]+", line) if part]
        if len(parts) < 2:
            raise ValueError(f"Line {line_number} does not contain both x and y values.")
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"Line {line_number} has a non-numeric x/y value: {line}") from exc
        points.append((x, y))

    if not points:
        raise ValueError("No valid x/y point rows were found.")
    return points


def _parse_alignment_points_json(text: str) -> list[tuple[float, float]] | None:
    if not text.startswith("["):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON points could not be parsed: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("JSON points must be a list.")

    points: list[tuple[float, float]] = []
    for index, item in enumerate(data, start=1):
        try:
            if isinstance(item, dict):
                x = float(item["x"])
                y = float(item["y"])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                x = float(item[0])
                y = float(item[1])
            else:
                raise TypeError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"JSON point {index} must be an object with x/y values or a two-value list."
            ) from exc
        points.append((x, y))

    if not points:
        raise ValueError("JSON point list is empty.")
    return points
