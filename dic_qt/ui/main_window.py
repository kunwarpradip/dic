from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL

import copy
import numpy as np
import pandas as pd
from PIL import Image
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.algorithm import detect_line_from_seed, cut_selected_lines, merge_lines
from ..core.image_io import load_image_data
from ..core.models import DicLine, DicSession, Point
from ..core.repository import DicLineRepository, db_path_for_image
from .alignment_panel import AlignmentPanel
from .auto_detection_panel import AutoDetectionPanel
from .classification_panel import ClassificationPanel
from .image_canvas import ImageCanvas
from .line_list_panel import LineListPanel
from .preview_panel import PreviewPanel
from .project_files_panel import ProjectFilesPanel
from .settings_panel import SettingsPanel


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DIC")
        self.resize(1200, 820)
        self.session = DicSession()
        self.raw_image: np.ndarray | None = None
        self.image_rgb: np.ndarray | None = None
        self.detection_rgb: np.ndarray | None = None
        self.repository: DicLineRepository | None = None
        self._undo_stack: list[tuple[list[DicLine], set[UUID]]] = []
        self._redo_stack: list[tuple[list[DicLine], set[UUID]]] = []
        self._max_history = 40
        self._manual_boundary_mask: np.ndarray | None = None
        self._manual_boundary_path: str | None = None

        self.canvas = ImageCanvas()
        self.line_list = LineListPanel()
        self.settings = SettingsPanel()
        self.preview = PreviewPanel()
        self.zoom_label = QLabel("Zoom: -")
        self.cursor_label = QLabel("Cursor: -")
        self.status_label = QLabel("No image loaded")

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        image_toolbar = QHBoxLayout()
        image_toolbar.addStretch(1)
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        self.undo_button.setToolTip("Undo the last manual edit.")
        self.redo_button.setToolTip("Redo the last undone manual edit.")
        image_toolbar.addWidget(self.undo_button)
        image_toolbar.addWidget(self.redo_button)
        image_toolbar.addWidget(self.zoom_label)
        left_layout.addLayout(image_toolbar)
        left_layout.addWidget(self.canvas, 1)

        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout(bottom_panel)
        bottom_layout.addWidget(self.preview)
        bottom_layout.addWidget(self.settings, 1)

        main_left = QWidget()
        main_left_layout = QVBoxLayout(main_left)
        main_left_layout.addWidget(left_panel, 1)
        main_left_layout.addWidget(bottom_panel)
        main_left_layout.addWidget(self.cursor_label)
        main_left_layout.addWidget(self.status_label)

        splitter = QSplitter()
        splitter.addWidget(main_left)
        splitter.addWidget(self.line_list)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.tabs = QTabWidget()
        self.project_files_panel = ProjectFilesPanel()
        self.tabs.addTab(self.project_files_panel, "Project Files")
        self.auto_detection_panel = AutoDetectionPanel()
        self.tabs.addTab(self.auto_detection_panel, "Auto Detection")
        self.tabs.addTab(splitter, "Manual Review")
        self.classification_panel = ClassificationPanel(
            self.auto_detection_panel.alignment_analysis_panel.classification_context,
            self.reviewed_event_tables,
        )
        self.tabs.addTab(self.classification_panel, "Classification")
        self.alignment_panel = AlignmentPanel()
        self.tabs.addTab(self.alignment_panel, "Alignment")
        self.setCentralWidget(self.tabs)
        self.project_files_panel.files_changed.connect(self.auto_detection_panel.set_project_files)
        self.project_files_panel.files_changed.connect(self.project_files_changed_for_manual_boundary)
        self.auto_detection_panel.set_project_files(self.project_files_panel.selected_files())
        self.auto_detection_panel.alignment_analysis_panel.shape_analysis_ready.connect(
            self.load_shape_events_for_manual_review
        )

        self.settings.choose_file_button.clicked.connect(self.choose_file)
        self.line_list.cut_mode_changed.connect(self.canvas.set_cut_mode)
        self.line_list.edit_mode_changed.connect(self.canvas.set_edit_mode)
        self.line_list.brush_radius_changed.connect(self.canvas.set_brush_radius)
        self.line_list.seed_grow_mode_changed.connect(self.set_create_mode)
        self.line_list.boundary_overlay_changed.connect(self.set_manual_boundary_overlay)
        self.line_list.boundary_cut_requested.connect(self.cut_selected_events_by_boundary)
        self.line_list.split_disconnected_requested.connect(self.split_disconnected_events)
        self.zoom_in_shortcut = QShortcut(QKeySequence("+"), self)
        self.zoom_in_shortcut.activated.connect(self.zoom_in_current_tab)
        self.zoom_in_equal_shortcut = QShortcut(QKeySequence("="), self)
        self.zoom_in_equal_shortcut.activated.connect(self.zoom_in_current_tab)
        self.zoom_out_shortcut = QShortcut(QKeySequence("-"), self)
        self.zoom_out_shortcut.activated.connect(self.zoom_out_current_tab)
        self.toggle_alignment_shortcut = QShortcut(QKeySequence("T"), self)
        self.toggle_alignment_shortcut.activated.connect(self.toggle_alignment_current_tab)
        self.select_all_shortcut = QShortcut(QKeySequence("A"), self)
        self.select_all_shortcut.activated.connect(self.select_all_events)
        self.delete_selected_shortcut = QShortcut(QKeySequence("D"), self)
        self.delete_selected_shortcut.activated.connect(self.delete_selected_events)
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.undo_manual_edit)
        self.redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        self.redo_shortcut.activated.connect(self.redo_manual_edit)
        self.undo_button.clicked.connect(self.undo_manual_edit)
        self.redo_button.clicked.connect(self.redo_manual_edit)
        self.canvas.seed_clicked.connect(self.create_event_from_seed)
        self.canvas.empty_clicked.connect(self.empty_image_clicked)
        self.canvas.image_mouse_moved.connect(self.update_preview)
        self.canvas.line_toggled.connect(self.toggle_line_visibility)
        self.canvas.cut_completed.connect(self.cut_lines)
        self.canvas.pixel_edit_requested.connect(self.edit_selected_event_pixels)
        self.canvas.zoom_changed.connect(self.zoom_changed)
        self.line_list.visibility_changed.connect(self.set_line_visibility)
        self.line_list.selection_changed_for_actions.connect(self.show_selected_events_only)
        self.line_list.select_all_requested.connect(self.select_all_events)
        self.line_list.merge_requested.connect(self.merge_selected_lines)
        self.line_list.delete_requested.connect(self.delete_selected_lines)
        self.line_list.seed_grow_checkbox.setChecked(True)
        self.canvas.set_create_mode(True)

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose DIC image",
            str(Path.cwd()),
            "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        self.load_image(Path(path))

    def zoom_in_current_tab(self) -> None:
        if self.tabs.currentWidget() is self.alignment_panel:
            self.alignment_panel.canvas.zoom_in()
        else:
            self.canvas.zoom_in()

    def zoom_out_current_tab(self) -> None:
        if self.tabs.currentWidget() is self.alignment_panel:
            self.alignment_panel.canvas.zoom_out()
        else:
            self.canvas.zoom_out()

    def toggle_alignment_current_tab(self) -> None:
        if self.tabs.currentWidget() is self.alignment_panel:
            self.alignment_panel.toggle_active_image()

    def load_image(self, path: Path) -> None:
        loaded = load_image_data(path)
        self.raw_image = loaded.raw
        self.image_rgb = loaded.display_rgb
        self.detection_rgb = loaded.detection_rgb
        h, w, _ = self.image_rgb.shape
        db_path = db_path_for_image(path)
        self.repository = DicLineRepository(db_path)
        lines = self.repository.load_lines()
        self.session = DicSession(
            image_path=str(path),
            db_path=str(db_path),
            image_width=w,
            image_height=h,
            lines=lines,
            visible_line_ids={line.id for line in lines},
        )
        self.canvas.set_image(self.image_rgb)
        self._manual_boundary_mask = None
        self._manual_boundary_path = None
        self.line_list.boundary_overlay_checkbox.blockSignals(True)
        self.line_list.boundary_overlay_checkbox.setChecked(False)
        self.line_list.boundary_overlay_checkbox.blockSignals(False)
        self.canvas.set_boundary_mask(None)
        self.canvas.set_boundary_visible(False)
        self._clear_history()
        self.refresh_lines()
        self.status_label.setText(
            f"Loaded {path.name} | DB: {db_path} | {len(lines)} lines | "
            f"{loaded.display_note} | Zoom in, then click a bright seed pixel "
            "to create an event"
        )

    def load_shape_events_for_manual_review(self, event_info: object, morphology: object) -> None:
        project_files = self.project_files_panel.selected_files()
        bln_path = project_files.get("bln_image", "").strip()
        if not bln_path:
            self.status_label.setText("Cannot populate Manual Review: no BLN image is loaded in Project Files.")
            return
        try:
            path = Path(bln_path).expanduser()
            loaded = load_image_data(path)
            event_pixels = event_info["event_pixels"].copy()
            event_types = morphology[["event_id", "event_type"]].copy()
        except Exception as exc:
            self.status_label.setText(f"Cannot populate Manual Review: {exc}")
            return

        h, w, _ = loaded.display_rgb.shape
        self.raw_image = loaded.raw
        self.image_rgb = loaded.display_rgb
        self.detection_rgb = loaded.detection_rgb
        self.repository = None
        lines: list[DicLine] = []
        event_pixels["event_id"] = event_pixels["event_id"].astype(str)
        event_types["event_id"] = event_types["event_id"].astype(str)
        event_pixels = event_pixels[event_pixels["event_id"].isin(set(event_types["event_id"]))]
        type_lookup = dict(zip(event_types["event_id"], event_types["event_type"]))
        for event_id, group in event_pixels.groupby("event_id", sort=False):
            points = {
                Point(int(x), int(y))
                for x, y in group[["pixel_x", "pixel_y"]].itertuples(index=False)
                if 0 <= int(x) < w and 0 <= int(y) < h
            }
            if not points:
                continue
            line = DicLine(
                id=uuid5(NAMESPACE_URL, f"dic-auto-review:{event_id}"),
                is_manual=False,
                points=points,
            )
            line.event_type = type_lookup.get(str(event_id), "unclassified")
            line.source_event_id = str(event_id)
            lines.append(line)
        self.session = DicSession(
            image_path=str(path),
            db_path=None,
            image_width=w,
            image_height=h,
            lines=lines,
            visible_line_ids={line.id for line in lines},
        )
        self._manual_boundary_mask = None
        self._manual_boundary_path = None
        self.canvas.set_image(self.image_rgb)
        if self.line_list.boundary_overlay_checkbox.isChecked():
            self.set_manual_boundary_overlay(True)
        else:
            self.canvas.set_boundary_mask(None)
            self.canvas.set_boundary_visible(False)
        self._clear_history()
        self.refresh_lines()
        self.tabs.setCurrentIndex(2)
        loaded_types = sorted({str(getattr(line, "event_type", "unclassified")) for line in lines})
        loaded_type_text = ", ".join(event_type.replace("_", " ") for event_type in loaded_types) or "none"
        self.status_label.setText(
            f"Manual Review populated with {len(lines)} auto events: {loaded_type_text}. "
            "Select one event, then use Add pixels or Erase pixels for brush edits."
        )

    def reviewed_event_tables(self) -> dict:
        event_rows = []
        pixel_rows = []
        for line in self.session.lines:
            if not isinstance(line, DicLine) or not line.points:
                continue
            event_id = str(line.id)
            xs = [point.x for point in line.points]
            ys = [point.y for point in line.points]
            event_rows.append(
                {
                    "event_id": event_id,
                    "event_type": getattr(line, "event_type", "manual" if line.is_manual else "reviewed"),
                    "num_pixels": int(len(line.points)),
                    "bbox_min_x": int(min(xs)),
                    "bbox_max_x": int(max(xs)),
                    "bbox_min_y": int(min(ys)),
                    "bbox_max_y": int(max(ys)),
                }
            )
            for point in sorted(line.points):
                pixel_rows.append({"event_id": event_id, "pixel_x": int(point.x), "pixel_y": int(point.y)})
        return {
            "events": pd.DataFrame(event_rows),
            "event_pixels": pd.DataFrame(pixel_rows),
        }

    def create_event_from_seed(self, x: int, y: int) -> None:
        if self.detection_rgb is None:
            self.status_label.setText("No detection image is loaded for seed growing.")
            return
        settings = self.settings.settings()
        result = detect_line_from_seed(
            x,
            y,
            self.detection_rgb,
            intensity_difference_tolerance=settings.intensity_difference_tolerance,
            bfl_tolerance=settings.bfl_tolerance,
            min_intensity=settings.min_intensity,
        )
        line = result.line
        if line is None:
            intensity = (
                "outside image"
                if result.seed_blue_intensity is None
                else str(result.seed_blue_intensity)
            )
            self.status_label.setText(
                f"No event at x={x}, y={y} | detection={intensity} | "
                f"{result.rejection_reason}"
            )
            return
        line.is_manual = True
        line.event_type = "manual"
        self._push_undo_state()
        self.session.add_line(line)
        if self.repository is not None:
            self.repository.save_new_lines([line])
        self.refresh_lines()
        self.status_label.setText(
            f"Added event {line.id} from x={x}, y={y} | "
            f"detection={result.seed_blue_intensity} | {line.size} points"
        )

    def empty_image_clicked(self, x: int, y: int) -> None:
        self.status_label.setText(
            f"Create event from click is off; no event created at x={x}, y={y}"
        )

    def set_create_mode(self, enabled: bool) -> None:
        self.line_list.seed_grow_checkbox.blockSignals(True)
        self.line_list.seed_grow_checkbox.setChecked(enabled)
        self.line_list.seed_grow_checkbox.blockSignals(False)
        self.canvas.set_create_mode(enabled)
        if enabled:
            self.status_label.setText("Create event from click is on")
        else:
            self.status_label.setText("Create event from click is off")

    def project_files_changed_for_manual_boundary(self, _paths: dict) -> None:
        self._manual_boundary_mask = None
        self._manual_boundary_path = None
        if self.line_list.boundary_overlay_checkbox.isChecked() and self._image_is_loaded():
            self.set_manual_boundary_overlay(True)

    def set_line_visibility(self, line_id: UUID, visible: bool) -> None:
        if visible:
            self.session.visible_line_ids.add(line_id)
        else:
            self.session.visible_line_ids.discard(line_id)
        self.canvas.set_lines(self.session.lines, self.session.visible_line_ids)

    def toggle_line_visibility(self, line_id: UUID) -> None:
        self.line_list.toggle_line(line_id)

    def merge_selected_lines(self, selected_ids: set[UUID]) -> None:
        if not selected_ids:
            return
        self._push_undo_state()
        new_lines, merged = merge_lines(self.session.lines, selected_ids)
        if merged is None:
            self._pop_empty_undo_state()
            return
        self.session.set_lines(new_lines)
        self.session.visible_line_ids.add(merged.id)
        if self.repository is not None:
            self.repository.replace_all(self.session.lines)
        self.refresh_lines()
        self.status_label.setText(f"Merged {len(selected_ids)} lines into {merged.id}")

    def delete_selected_lines(self, selected_ids: set[UUID]) -> None:
        if not selected_ids:
            if not selected_ids:
                self.status_label.setText("No selected events to delete")
            return
        self._push_undo_state()
        self.session.delete_lines(selected_ids)
        if self.repository is not None:
            self.repository.delete_lines(selected_ids)
        self.refresh_lines()
        self.status_label.setText(f"Deleted {len(selected_ids)} events")

    def select_all_events(self) -> None:
        if not self._image_is_loaded():
            return
        count = self.line_list.select_all_lines()
        self.session.visible_line_ids = {line.id for line in self.session.lines}
        self.canvas.set_lines(self.session.lines, self.session.visible_line_ids)
        if count == 0:
            self.status_label.setText("No events to select")
            return
        self.status_label.setText(f"Displaying all {count} events")

    def show_selected_events_only(self) -> None:
        if not self._image_is_loaded():
            return
        selected_ids = self.line_list.selected_line_ids()
        if not selected_ids:
            return
        self.session.visible_line_ids = set(selected_ids)
        self.canvas.set_lines(self.session.lines, self.session.visible_line_ids)
        self.status_label.setText(f"Displaying {len(selected_ids)} selected event(s)")

    def delete_selected_events(self) -> None:
        if not self._image_is_loaded():
            return
        self.delete_selected_lines(self.line_list.checked_line_ids())

    def cut_lines(self, start: Point, end: Point) -> None:
        if self.canvas.zoom_level <= 0:
            return
        selected_ids = self.line_list.selected_line_ids()
        if not selected_ids:
            selected_ids = set(self.session.visible_line_ids)
        cut_width = max(2, int(2 * (self.session.image_height / max(1, self.canvas.height())) / 7))
        new_lines, daughters = cut_selected_lines(
            self.session.lines,
            selected_ids,
            start,
            end,
            cut_width,
        )
        if not daughters:
            self.status_label.setText("Cut did not intersect a selected visible line")
            return
        self._push_undo_state()
        self.session.set_lines(new_lines)
        self.session.visible_line_ids.update(line.id for line in daughters)
        if self.repository is not None:
            self.repository.replace_all(self.session.lines)
        self.refresh_lines()
        self.status_label.setText(f"Cut line into {len(daughters)} daughter lines")

    def split_disconnected_events(self, selected_ids: set[UUID]) -> None:
        if not self._image_is_loaded():
            return
        if not selected_ids:
            self.status_label.setText("Select or check at least one event before splitting disconnected pieces.")
            return
        new_lines: list[DicLine] = []
        replacement_ids: set[UUID] = set()
        split_event_count = 0
        created_count = 0
        for line in self.session.lines:
            if line.id not in selected_ids:
                new_lines.append(line)
                continue
            components = self._connected_event_components(line.points)
            if len(components) <= 1:
                new_lines.append(line)
                continue
            split_event_count += 1
            created_count += len(components)
            for component in components:
                replacement = self._line_with_points(line, component, keep_id=False)
                new_lines.append(replacement)
                replacement_ids.add(replacement.id)

        if split_event_count == 0:
            self.status_label.setText("Selected events are already single connected pieces.")
            return

        self._push_undo_state()
        self.session.set_lines(new_lines)
        self.session.visible_line_ids.update(replacement_ids)
        self.session.visible_line_ids.difference_update(selected_ids - replacement_ids)
        if self.repository is not None:
            self.repository.replace_all(self.session.lines)
        self.refresh_lines()
        self.status_label.setText(
            f"Split {split_event_count} event(s) into {created_count} disconnected pieces."
        )

    def set_manual_boundary_overlay(self, visible: bool) -> None:
        if not visible:
            self.canvas.set_boundary_visible(False)
            return
        mask = self._load_manual_boundary_mask()
        if mask is None:
            self.line_list.boundary_overlay_checkbox.blockSignals(True)
            self.line_list.boundary_overlay_checkbox.setChecked(False)
            self.line_list.boundary_overlay_checkbox.blockSignals(False)
            self.canvas.set_boundary_visible(False)
            return
        self.canvas.set_boundary_mask(mask)
        self.canvas.set_boundary_visible(True)
        self.status_label.setText("Showing EBSD boundary overlay in Manual Review.")

    def cut_selected_events_by_boundary(self) -> None:
        if not self._image_is_loaded():
            return
        mask = self._load_manual_boundary_mask()
        if mask is None:
            return
        selected_ids = self.line_list.selected_line_ids() or self.line_list.checked_line_ids()
        if not selected_ids:
            self.status_label.setText("Select or check at least one event before cutting by boundary.")
            return

        new_lines: list[DicLine] = []
        replacement_ids: set[UUID] = set()
        changed_count = 0
        removed_pixels = 0
        created_count = 0
        for line in self.session.lines:
            if line.id not in selected_ids:
                new_lines.append(line)
                continue
            replacements, removed = self._split_event_by_boundary(line, mask)
            if removed == 0:
                new_lines.append(line)
                continue
            changed_count += 1
            removed_pixels += removed
            created_count += len(replacements)
            new_lines.extend(replacements)
            replacement_ids.update(replacement.id for replacement in replacements)

        if changed_count == 0:
            self.status_label.setText("Boundary cut did not intersect the selected events.")
            return

        self._push_undo_state()
        self.session.set_lines(new_lines)
        self.session.visible_line_ids.update(replacement_ids)
        self.session.visible_line_ids.difference_update(selected_ids - replacement_ids)
        if self.repository is not None:
            self.repository.replace_all(self.session.lines)
        self.refresh_lines()
        if self.line_list.boundary_overlay_checkbox.isChecked():
            self.canvas.set_boundary_mask(mask)
            self.canvas.set_boundary_visible(True)
        self.status_label.setText(
            f"Boundary cut updated {changed_count} event(s), removed {removed_pixels:,} boundary pixels, "
            f"and produced {created_count} event segment(s)."
        )

    def edit_selected_event_pixels(self, mode: str, center: Point, radius: int) -> None:
        selected_ids = self.line_list.selected_line_ids() or self.line_list.checked_line_ids()
        if len(selected_ids) != 1:
            self.status_label.setText("Select exactly one event before adding or erasing pixels.")
            return
        line_id = next(iter(selected_ids))
        line = self.session.line_by_id(line_id)
        if line is None:
            return
        radius = max(1, int(radius))
        brush_points = {
            Point(center.x + dx, center.y + dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius
            and 0 <= center.x + dx < self.session.image_width
            and 0 <= center.y + dy < self.session.image_height
        }
        if mode == "add":
            if brush_points.issubset(line.points):
                return
            self._push_undo_state()
            line.points.update(brush_points)
        elif mode == "erase":
            if not (line.points & brush_points):
                return
            self._push_undo_state()
            line.points.difference_update(brush_points)
        if self.repository is not None:
            self.repository.save_new_lines([line])
        self.refresh_lines()

    def update_preview(self, x: int, y: int) -> None:
        self.preview.update_preview(self.image_rgb, x, y)
        if self.image_rgb is None:
            self.cursor_label.setText("Cursor: -")
            return
        h, w, _ = self.image_rgb.shape
        if 0 <= x < w and 0 <= y < h:
            display_blue = int(self.image_rgb[y, x, 2])
            detection_blue = (
                int(self.detection_rgb[y, x, 2])
                if self.detection_rgb is not None
                else display_blue
            )
            self.cursor_label.setText(
                f"Cursor: x={x}, y={y}, display={display_blue}, detection={detection_blue}"
            )
        else:
            self.cursor_label.setText(f"Cursor: x={x}, y={y}, outside image")

    def zoom_changed(self, zoom_level: int) -> None:
        image_loaded = self.image_rgb is not None
        enabled = zoom_level > 0
        self.line_list.set_actions_enabled(image_loaded, enabled)
        if not image_loaded:
            self.zoom_label.setText("Zoom: -")
        elif zoom_level == 0:
            self.zoom_label.setText("Zoom: Fit")
        else:
            self.zoom_label.setText(f"Zoom: {self.canvas.zoom_factor * 100:.0f}%")

    def refresh_lines(self) -> None:
        invalid_count = self._discard_invalid_lines()
        self.line_list.set_lines(self.session.lines, self.session.visible_line_ids)
        self.canvas.set_lines(self.session.lines, self.session.visible_line_ids)
        self._update_history_buttons()
        if invalid_count:
            self.status_label.setText(
                f"Discarded {invalid_count} invalid in-memory event from a rejected seed"
            )

    def undo_manual_edit(self) -> None:
        if not self._undo_stack:
            self.status_label.setText("Nothing to undo")
            return
        self._redo_stack.append(self._snapshot_state())
        self._restore_state(self._undo_stack.pop())
        self.status_label.setText("Undid last manual edit")

    def redo_manual_edit(self) -> None:
        if not self._redo_stack:
            self.status_label.setText("Nothing to redo")
            return
        self._undo_stack.append(self._snapshot_state())
        self._restore_state(self._redo_stack.pop())
        self.status_label.setText("Redid manual edit")

    def _push_undo_state(self) -> None:
        self._undo_stack.append(self._snapshot_state())
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_history_buttons()

    def _pop_empty_undo_state(self) -> None:
        if self._undo_stack:
            self._undo_stack.pop()
        self._update_history_buttons()

    def _snapshot_state(self) -> tuple[list[DicLine], set[UUID]]:
        return copy.deepcopy(self.session.lines), set(self.session.visible_line_ids)

    def _restore_state(self, state: tuple[list[DicLine], set[UUID]]) -> None:
        lines, visible_ids = state
        self.session.set_lines(copy.deepcopy(lines))
        self.session.visible_line_ids = set(visible_ids)
        if self.repository is not None:
            self.repository.replace_all(self.session.lines)
        self.refresh_lines()

    def _clear_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_history_buttons()

    def _update_history_buttons(self) -> None:
        self.undo_button.setEnabled(bool(self._undo_stack))
        self.redo_button.setEnabled(bool(self._redo_stack))

    def _discard_invalid_lines(self) -> int:
        valid_lines = [line for line in self.session.lines if isinstance(line, DicLine)]
        invalid_count = len(self.session.lines) - len(valid_lines)
        if not invalid_count:
            return 0
        self.session.lines = valid_lines
        self.session.visible_line_ids.intersection_update(line.id for line in valid_lines)
        return invalid_count

    def _image_is_loaded(self) -> bool:
        return self.image_rgb is not None

    def _load_manual_boundary_mask(self) -> np.ndarray | None:
        if self.session.image_width <= 0 or self.session.image_height <= 0:
            self.status_label.setText("Load the BLN image before showing or using the boundary.")
            return None
        boundary_path = self.project_files_panel.selected_files().get("ebsd_boundary_image", "").strip()
        if not boundary_path:
            self.status_label.setText("No EBSD boundary image is set in Project Files.")
            return None
        path = str(Path(boundary_path).expanduser())
        if self._manual_boundary_mask is not None and self._manual_boundary_path == path:
            return self._manual_boundary_mask
        try:
            with Image.open(path) as img:
                arr = np.asarray(img)
        except Exception as exc:
            self.status_label.setText(f"Could not load EBSD boundary image: {exc}")
            return None

        arr = np.squeeze(arr)
        if arr.ndim == 3:
            gray = arr[..., :3].astype(np.float32).mean(axis=2)
        else:
            gray = arr.astype(np.float32)
        max_value = float(np.nanmax(gray)) if gray.size else 0.0
        if max_value > 1.0:
            gray = gray / max_value
        mask = np.nan_to_num(gray, nan=1.0) < 0.6
        h, w = mask.shape[:2]
        if h < self.session.image_height or w < self.session.image_width:
            self.status_label.setText(
                "EBSD boundary image is smaller than the loaded DIC image. "
                f"Boundary: {w} x {h}; DIC: {self.session.image_width} x {self.session.image_height}."
            )
            return None
        self._manual_boundary_path = path
        self._manual_boundary_mask = mask[: self.session.image_height, : self.session.image_width]
        return self._manual_boundary_mask

    def _split_event_by_boundary(self, line: DicLine, boundary_mask: np.ndarray) -> tuple[list[DicLine], int]:
        boundary_points = {
            point
            for point in line.points
            if 0 <= point.x < boundary_mask.shape[1]
            and 0 <= point.y < boundary_mask.shape[0]
            and bool(boundary_mask[point.y, point.x])
        }
        if not boundary_points:
            return [line], 0
        remaining = line.points - boundary_points
        if not remaining:
            return [], len(boundary_points)
        components = self._connected_event_components(remaining)
        if len(components) == 1:
            return [self._line_with_points(line, components[0], keep_id=True)], len(boundary_points)
        return [self._line_with_points(line, component, keep_id=False) for component in components], len(boundary_points)

    @staticmethod
    def _line_with_points(line: DicLine, points: set[Point], keep_id: bool) -> DicLine:
        new_line = DicLine(
            id=line.id if keep_id else uuid4(),
            is_manual=line.is_manual,
            points=set(points),
            seed_point=None,
            intensity_difference_tolerance=line.intensity_difference_tolerance,
            bfl_tolerance=line.bfl_tolerance,
            min_intensity=line.min_intensity,
        )
        new_line.event_type = getattr(line, "event_type", "manual" if line.is_manual else "reviewed")
        new_line.source_event_id = getattr(line, "source_event_id", str(line.id))
        return new_line

    @staticmethod
    def _connected_event_components(points: set[Point]) -> list[set[Point]]:
        remaining = set(points)
        components: list[set[Point]] = []
        neighbors = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
        while remaining:
            start = remaining.pop()
            component = {start}
            stack = [start]
            while stack:
                current = stack.pop()
                for dx, dy in neighbors:
                    neighbor = Point(current.x + dx, current.y + dy)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            components.append(component)
        return sorted(components, key=len, reverse=True)
