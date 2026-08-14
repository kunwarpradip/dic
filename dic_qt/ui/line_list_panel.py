from __future__ import annotations

from uuid import UUID

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.models import DicLine

EVENT_TYPE_ORDER = ["linear", "irregular", "blob_like", "fragmented", "small_noise", "unclassified"]
EVENT_TYPE_LABELS = {
    "linear": "Linear",
    "irregular": "Irregular",
    "blob_like": "Blob Like",
    "fragmented": "Fragmented",
    "small_noise": "Small Noise",
    "unclassified": "Unclassified",
    "manual": "Manual",
}


class LineListPanel(QWidget):
    visibility_changed = Signal(object, bool)
    selection_changed_for_actions = Signal()
    select_all_requested = Signal()
    merge_requested = Signal(object)
    delete_requested = Signal(object)
    cut_mode_changed = Signal(bool)
    edit_mode_changed = Signal(str)
    brush_radius_changed = Signal(int)
    seed_grow_mode_changed = Signal(bool)
    boundary_overlay_changed = Signal(bool)
    boundary_cut_requested = Signal()
    split_disconnected_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.edit_mode = QComboBox()
        self.edit_mode.addItems(["Select events", "Add pixels", "Erase pixels", "Draw cut line", "Grow by seed"])
        self.edit_mode.setEnabled(False)
        self.edit_mode.setToolTip(
            "Choose the active manual interaction: select events, add pixels, erase pixels, draw a cut line, or grow a new event from a seed click."
        )
        self.brush_radius = QSpinBox()
        self.brush_radius.setRange(1, 25)
        self.brush_radius.setValue(2)
        self.brush_radius.setEnabled(False)
        self.brush_radius.setToolTip("Brush radius in pixels for add/erase editing.")
        self.boundary_overlay_checkbox = QCheckBox("Show Boundary")
        self.boundary_overlay_checkbox.setEnabled(False)
        self.boundary_overlay_checkbox.setToolTip("Overlay the EBSD boundary image from Project Files on the manual review canvas.")
        self.boundary_cut_button = QPushButton("Cut By Boundary")
        self.boundary_cut_button.setEnabled(False)
        self.boundary_cut_button.setToolTip("Cut selected or visible checked events wherever they intersect the EBSD boundary pixels.")
        self.split_disconnected_button = QPushButton("Split Disconnected Pieces")
        self.split_disconnected_button.setEnabled(False)
        self.split_disconnected_button.setToolTip("After erasing connecting pixels, split selected or checked events into separate connected components.")
        self.event_count_label = QLabel("Events: 0")
        self.event_count_label.setToolTip("Total number of events currently loaded into Manual Review.")
        self.all_checkbox = QCheckBox("All")
        self.all_checkbox.setToolTip("Show or hide all events in the currently selected event-type tab.")
        self.list_tabs = QTabWidget()
        self.list_widgets: dict[str, QListWidget] = {}
        self._rebuild_tabs(["manual"])
        self.select_all_button = QPushButton("Select All")
        self.merge_button = QPushButton("Merge")
        self.delete_button = QPushButton("Delete Selected")
        self.select_all_button.setToolTip("Display every loaded event in the overlay, across all event-type tabs.")
        self.merge_button.setToolTip("Merge all currently checked events into one manual event.")
        self.delete_button.setToolTip("Delete all currently checked events from Manual Review.")
        self.select_all_button.setEnabled(False)
        self.merge_button.setEnabled(False)
        self.delete_button.setEnabled(False)

        header = QHBoxLayout()
        header.addWidget(self.all_checkbox)
        header.addStretch(1)

        buttons = QHBoxLayout()
        buttons.addWidget(self.select_all_button)
        buttons.addWidget(self.merge_button)
        buttons.addWidget(self.delete_button)

        layout = QVBoxLayout(self)
        edit_row = QHBoxLayout()
        edit_row.addWidget(self.edit_mode, 1)
        edit_row.addWidget(QLabel("Brush"))
        edit_row.addWidget(self.brush_radius)
        layout.addLayout(edit_row)
        boundary_row = QHBoxLayout()
        boundary_row.addWidget(self.boundary_overlay_checkbox)
        boundary_row.addWidget(self.boundary_cut_button)
        layout.addLayout(boundary_row)
        layout.addWidget(self.split_disconnected_button)
        layout.addWidget(self.event_count_label)
        layout.addLayout(header)
        layout.addWidget(self.list_tabs, 1)
        layout.addLayout(buttons)

        self.all_checkbox.toggled.connect(self._toggle_all)
        self.list_tabs.currentChanged.connect(lambda _index: self._sync_all_checkbox())
        self.select_all_button.clicked.connect(lambda: self.select_all_requested.emit())
        self.merge_button.clicked.connect(self._merge_clicked)
        self.delete_button.clicked.connect(self._delete_clicked)
        self.edit_mode.currentTextChanged.connect(self._edit_mode_changed)
        self.brush_radius.valueChanged.connect(self.brush_radius_changed.emit)
        self.boundary_overlay_checkbox.toggled.connect(self.boundary_overlay_changed.emit)
        self.boundary_cut_button.clicked.connect(lambda checked=False: self.boundary_cut_requested.emit())
        self.split_disconnected_button.clicked.connect(self._split_disconnected_clicked)

    def set_actions_enabled(self, enabled: bool, cut_enabled: bool) -> None:
        self.select_all_button.setEnabled(enabled)
        self.merge_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)
        self.edit_mode.setEnabled(enabled)
        self.brush_radius.setEnabled(enabled)
        self.boundary_overlay_checkbox.setEnabled(enabled)
        self.boundary_cut_button.setEnabled(enabled)
        self.split_disconnected_button.setEnabled(enabled)
        if not enabled:
            self.edit_mode.setCurrentIndex(0)
            self.boundary_overlay_checkbox.setChecked(False)

    def set_draw_cut_enabled(self, enabled: bool) -> None:
        target = "Draw cut line" if enabled else "Select events"
        index = self.edit_mode.findText(target)
        if index >= 0:
            self.edit_mode.setCurrentIndex(index)
        else:
            self.cut_mode_changed.emit(enabled)

    def set_lines(self, lines: list[DicLine], visible_ids: set[UUID]) -> None:
        valid_lines = [line for line in lines if isinstance(line, DicLine)]
        buckets = self._buckets_for_lines(valid_lines)
        self._rebuild_tabs(buckets)
        for widget in self.list_widgets.values():
            widget.blockSignals(True)
            widget.clear()
        self.event_count_label.setText(f"Events: {len(valid_lines)}")
        counts = {bucket: 0 for bucket in buckets}
        for line in sorted(valid_lines, key=lambda item: item.size, reverse=True):
            event_type = getattr(line, "event_type", "")
            bucket = self._bucket_for_line(line)
            if bucket is None:
                continue
            counts[bucket] += 1
            item_label = f"{event_type}: {line.id}" if event_type else str(line.id)
            item_label = f"{item_label} ({line.size:,} px)"
            item = QListWidgetItem(item_label)
            item.setData(Qt.ItemDataRole.UserRole, line.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if line.id in visible_ids
                else Qt.CheckState.Unchecked
            )
            if line.is_manual:
                item.setForeground(Qt.GlobalColor.blue)
            elif event_type == "linear":
                item.setForeground(Qt.GlobalColor.darkGreen)
            elif event_type == "irregular":
                item.setForeground(Qt.GlobalColor.darkYellow)
            elif event_type == "blob_like":
                item.setForeground(Qt.GlobalColor.darkMagenta)
            elif event_type == "fragmented":
                item.setForeground(Qt.GlobalColor.darkCyan)
            elif event_type == "small_noise":
                item.setForeground(Qt.GlobalColor.darkRed)
            self.list_widgets[bucket].addItem(item)
        for key, widget in self.list_widgets.items():
            widget.blockSignals(False)
            self.list_tabs.setTabText(
                self.list_tabs.indexOf(widget),
                f"{EVENT_TYPE_LABELS.get(key, key.replace('_', ' ').title())} ({counts.get(key, 0)})",
            )
        self._sync_all_checkbox()

    def selected_line_ids(self) -> set[UUID]:
        ids: set[UUID] = set()
        for widget in self.list_widgets.values():
            ids.update(
                item.data(Qt.ItemDataRole.UserRole)
                for item in widget.selectedItems()
            )
        return ids

    def select_all_lines(self) -> int:
        total = 0
        for widget in self.list_widgets.values():
            widget.blockSignals(True)
            for i in range(widget.count()):
                widget.item(i).setCheckState(Qt.CheckState.Checked)
                total += 1
            widget.clearSelection()
            widget.blockSignals(False)
            for i in range(widget.count()):
                line_id = widget.item(i).data(Qt.ItemDataRole.UserRole)
                self.visibility_changed.emit(line_id, True)
        self._sync_all_checkbox()
        self.selection_changed_for_actions.emit()
        return total

    def checked_line_ids(self) -> set[UUID]:
        ids: set[UUID] = set()
        for widget in self.list_widgets.values():
            for i in range(widget.count()):
                item = widget.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    ids.add(item.data(Qt.ItemDataRole.UserRole))
        return ids

    def toggle_line(self, line_id: UUID) -> None:
        for widget in self.list_widgets.values():
            for i in range(widget.count()):
                item = widget.item(i)
                if item.data(Qt.ItemDataRole.UserRole) != line_id:
                    continue
                new_state = (
                    Qt.CheckState.Unchecked
                    if item.checkState() == Qt.CheckState.Checked
                    else Qt.CheckState.Checked
                )
                item.setCheckState(new_state)
                item.setSelected(new_state == Qt.CheckState.Checked)
                return

    def select_line(self, line_id: UUID) -> bool:
        found = False
        for widget in self.list_widgets.values():
            widget.blockSignals(True)
            for i in range(widget.count()):
                item = widget.item(i)
                selected = item.data(Qt.ItemDataRole.UserRole) == line_id
                item.setSelected(selected)
                if selected:
                    found = True
                    self.list_tabs.setCurrentWidget(widget)
                    widget.scrollToItem(item)
            widget.blockSignals(False)
        if found:
            self.selection_changed_for_actions.emit()
        return found

    def _toggle_all(self, checked: bool) -> None:
        widget = self.current_list_widget()
        if widget is None:
            return
        widget.blockSignals(True)
        for i in range(widget.count()):
            widget.item(i).setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
        widget.blockSignals(False)
        for i in range(widget.count()):
            line_id = widget.item(i).data(Qt.ItemDataRole.UserRole)
            self.visibility_changed.emit(line_id, checked)

    def _item_changed(self, item: QListWidgetItem) -> None:
        line_id = item.data(Qt.ItemDataRole.UserRole)
        visible = item.checkState() == Qt.CheckState.Checked
        self.visibility_changed.emit(line_id, visible)
        self._sync_all_checkbox()

    def _selection_changed(self) -> None:
        self.selection_changed_for_actions.emit()

    def _merge_clicked(self) -> None:
        self.merge_requested.emit(self.checked_line_ids())

    def _delete_clicked(self) -> None:
        self.delete_requested.emit(self.checked_line_ids())

    def _split_disconnected_clicked(self) -> None:
        selected_ids = self.selected_line_ids() or self.checked_line_ids()
        self.split_disconnected_requested.emit(selected_ids)

    def _edit_mode_changed(self, text: str) -> None:
        self.cut_mode_changed.emit(text.startswith("Draw"))
        self.seed_grow_mode_changed.emit(text.startswith("Grow"))
        if text.startswith("Add"):
            self.edit_mode_changed.emit("add")
        elif text.startswith("Erase"):
            self.edit_mode_changed.emit("erase")
        else:
            self.edit_mode_changed.emit("select")

    def _sync_all_checkbox(self) -> None:
        widget = self.current_list_widget()
        if widget is None:
            self.all_checkbox.setChecked(False)
            return
        if widget.count() == 0:
            self.all_checkbox.setChecked(False)
            return
        all_checked = all(
            widget.item(i).checkState() == Qt.CheckState.Checked
            for i in range(widget.count())
        )
        self.all_checkbox.blockSignals(True)
        self.all_checkbox.setChecked(all_checked)
        self.all_checkbox.blockSignals(False)

    def current_list_widget(self) -> QListWidget | None:
        return self.list_tabs.currentWidget()

    @staticmethod
    def _bucket_for_line(line: DicLine) -> str | None:
        if line.is_manual:
            return "manual"
        event_type = str(getattr(line, "event_type", "") or "unclassified")
        return event_type

    def _buckets_for_lines(self, lines: list[DicLine]) -> list[str]:
        present = {self._bucket_for_line(line) for line in lines}
        present.discard(None)
        ordered = [key for key in EVENT_TYPE_ORDER if key in present]
        extras = sorted(key for key in present if key not in EVENT_TYPE_ORDER and key != "manual")
        if "manual" in present or not ordered and not extras:
            extras.append("manual")
        elif "manual" in present:
            extras.append("manual")
        return ordered + extras

    def _rebuild_tabs(self, buckets: list[str]) -> None:
        buckets = buckets or ["manual"]
        current_key = None
        current_widget = self.current_list_widget() if self.list_tabs.count() else None
        for key, widget in self.list_widgets.items():
            if widget is current_widget:
                current_key = key
                break
        while self.list_tabs.count():
            widget = self.list_tabs.widget(0)
            self.list_tabs.removeTab(0)
            widget.deleteLater()
        self.list_widgets = {}
        for key in buckets:
            widget = QListWidget()
            widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            widget.itemChanged.connect(self._item_changed)
            widget.itemSelectionChanged.connect(self._selection_changed)
            self.list_widgets[key] = widget
            self.list_tabs.addTab(widget, EVENT_TYPE_LABELS.get(key, key.replace("_", " ").title()))
        if current_key in self.list_widgets:
            self.list_tabs.setCurrentWidget(self.list_widgets[current_key])


def _line_sort_distance(line: DicLine) -> float:
    if not line.points:
        return 0.0
    x = sum(p.x for p in line.points) / len(line.points)
    y = sum(p.y for p in line.points) / len(line.points)
    return x * x + y * y
