from __future__ import annotations

from pathlib import Path
from uuid import UUID

import numpy as np
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
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
from .image_canvas import ImageCanvas
from .line_list_panel import LineListPanel
from .preview_panel import PreviewPanel
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
        self.tabs.addTab(splitter, "Manual Detection")
        self.auto_detection_panel = AutoDetectionPanel()
        self.tabs.addTab(self.auto_detection_panel, "Auto Detection")
        self.alignment_panel = AlignmentPanel()
        self.tabs.addTab(self.alignment_panel, "Alignment")
        self.setCentralWidget(self.tabs)

        self.settings.choose_file_button.clicked.connect(self.choose_file)
        self.settings.create_event_checkbox.toggled.connect(self.set_create_mode)
        self.line_list.cut_mode_changed.connect(self.canvas.set_cut_mode)
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
        self.canvas.seed_clicked.connect(self.create_event_from_seed)
        self.canvas.empty_clicked.connect(self.empty_image_clicked)
        self.canvas.image_mouse_moved.connect(self.update_preview)
        self.canvas.line_toggled.connect(self.toggle_line_visibility)
        self.canvas.cut_completed.connect(self.cut_lines)
        self.canvas.zoom_changed.connect(self.zoom_changed)
        self.line_list.visibility_changed.connect(self.set_line_visibility)
        self.line_list.select_all_requested.connect(self.select_all_events)
        self.line_list.merge_requested.connect(self.merge_selected_lines)
        self.line_list.delete_requested.connect(self.delete_selected_lines)
        self.canvas.set_create_mode(self.settings.create_event_checkbox.isChecked())

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
        self.refresh_lines()
        self.status_label.setText(
            f"Loaded {path.name} | DB: {db_path} | {len(lines)} lines | "
            f"{loaded.display_note} | Zoom in, then click a bright seed pixel "
            "to create an event"
        )

    def create_event_from_seed(self, x: int, y: int) -> None:
        if self.detection_rgb is None or self.repository is None:
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
        self.session.add_line(line)
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
        self.canvas.set_create_mode(enabled)
        if enabled:
            self.status_label.setText("Create event from click is on")
        else:
            self.status_label.setText("Create event from click is off")

    def set_line_visibility(self, line_id: UUID, visible: bool) -> None:
        if visible:
            self.session.visible_line_ids.add(line_id)
        else:
            self.session.visible_line_ids.discard(line_id)
        self.canvas.set_lines(self.session.lines, self.session.visible_line_ids)

    def toggle_line_visibility(self, line_id: UUID) -> None:
        self.line_list.toggle_line(line_id)

    def merge_selected_lines(self, selected_ids: set[UUID]) -> None:
        if self.repository is None:
            return
        new_lines, merged = merge_lines(self.session.lines, selected_ids)
        if merged is None:
            return
        self.session.set_lines(new_lines)
        self.session.visible_line_ids.add(merged.id)
        self.repository.replace_all(self.session.lines)
        self.refresh_lines()
        self.status_label.setText(f"Merged {len(selected_ids)} lines into {merged.id}")

    def delete_selected_lines(self, selected_ids: set[UUID]) -> None:
        if self.repository is None or not selected_ids:
            if not selected_ids:
                self.status_label.setText("No selected events to delete")
            return
        self.session.delete_lines(selected_ids)
        self.repository.delete_lines(selected_ids)
        self.refresh_lines()
        self.status_label.setText(f"Deleted {len(selected_ids)} events")

    def select_all_events(self) -> None:
        if not self._image_is_loaded():
            return
        count = self.line_list.select_all_lines()
        if count == 0:
            self.status_label.setText("No events to select")
            return
        self.status_label.setText(f"Selected {count} events for overlay")

    def delete_selected_events(self) -> None:
        if not self._image_is_loaded():
            return
        self.delete_selected_lines(self.line_list.checked_line_ids())

    def cut_lines(self, start: Point, end: Point) -> None:
        if self.repository is None or self.canvas.zoom_level <= 0:
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
        self.session.set_lines(new_lines)
        self.session.visible_line_ids.update(line.id for line in daughters)
        self.repository.replace_all(self.session.lines)
        self.line_list.cut_checkbox.setChecked(False)
        self.refresh_lines()
        self.status_label.setText(f"Cut line into {len(daughters)} daughter lines")

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
        if invalid_count:
            self.status_label.setText(
                f"Discarded {invalid_count} invalid in-memory event from a rejected seed"
            )

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
