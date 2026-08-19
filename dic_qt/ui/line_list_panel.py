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
    visibility_many_changed = Signal(object, bool)
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
    locate_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._lines_by_bucket: dict[str, list[DicLine]] = {}
        self._line_bucket: dict[UUID, str] = {}
        self._visible_ids: set[UUID] = set()
        self._page_by_bucket: dict[str, int] = {}
        self._page_size = 200
        self.edit_mode = QComboBox()
        self.edit_mode.addItems(["Select events", "Add pixels", "Erase pixels", "Erase drawn area", "Draw cut line", "Grow by seed"])
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
        self.page_label = QLabel("Page 0 / 0")
        self.page_label.setToolTip("Only a page of events is rendered in the list for speed; overlay visibility still applies to all events.")
        self.prev_page_button = QPushButton("Previous")
        self.next_page_button = QPushButton("Next")
        self.prev_page_button.setToolTip("Show the previous page of events for the current event type.")
        self.next_page_button.setToolTip("Show the next page of events for the current event type.")
        self.select_all_button = QPushButton("Select All")
        self.locate_button = QPushButton("Locate Selected")
        self.merge_button = QPushButton("Merge")
        self.delete_button = QPushButton("Delete Selected")
        self.select_all_button.setToolTip("Display every loaded event in the overlay, across all event-type tabs.")
        self.locate_button.setToolTip("Blink and center the currently selected event so it is easier to find in the image.")
        self.merge_button.setToolTip("Merge all currently checked events into one manual event.")
        self.delete_button.setToolTip("Delete all currently checked events from Manual Review.")
        self.select_all_button.setEnabled(False)
        self.locate_button.setEnabled(False)
        self.merge_button.setEnabled(False)
        self.delete_button.setEnabled(False)

        header = QHBoxLayout()
        header.addWidget(self.all_checkbox)
        header.addStretch(1)

        page_row = QHBoxLayout()
        page_row.addWidget(self.prev_page_button)
        page_row.addWidget(self.page_label, 1)
        page_row.addWidget(self.next_page_button)

        buttons = QHBoxLayout()
        buttons.addWidget(self.select_all_button)
        buttons.addWidget(self.locate_button)
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
        layout.addLayout(page_row)
        layout.addWidget(self.list_tabs, 1)
        layout.addLayout(buttons)

        self.all_checkbox.toggled.connect(self._toggle_all)
        self.list_tabs.currentChanged.connect(lambda _index: self._current_tab_changed())
        self.prev_page_button.clicked.connect(lambda checked=False: self._change_page(-1))
        self.next_page_button.clicked.connect(lambda checked=False: self._change_page(1))
        self.select_all_button.clicked.connect(lambda: self.select_all_requested.emit())
        self.locate_button.clicked.connect(lambda checked=False: self.locate_requested.emit())
        self.merge_button.clicked.connect(self._merge_clicked)
        self.delete_button.clicked.connect(self._delete_clicked)
        self.edit_mode.currentTextChanged.connect(self._edit_mode_changed)
        self.brush_radius.valueChanged.connect(self.brush_radius_changed.emit)
        self.boundary_overlay_checkbox.toggled.connect(self.boundary_overlay_changed.emit)
        self.boundary_cut_button.clicked.connect(lambda checked=False: self.boundary_cut_requested.emit())
        self.split_disconnected_button.clicked.connect(self._split_disconnected_clicked)

    def set_actions_enabled(self, enabled: bool, cut_enabled: bool) -> None:
        self.select_all_button.setEnabled(enabled)
        self.locate_button.setEnabled(enabled)
        self.merge_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)
        self.prev_page_button.setEnabled(enabled)
        self.next_page_button.setEnabled(enabled)
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
        self.event_count_label.setText(f"Events: {len(valid_lines)}")
        self._visible_ids = set(visible_ids)
        self._lines_by_bucket = {bucket: [] for bucket in buckets}
        self._line_bucket = {}
        for line in sorted(valid_lines, key=lambda item: item.size, reverse=True):
            bucket = self._bucket_for_line(line)
            if bucket is None:
                continue
            self._lines_by_bucket.setdefault(bucket, []).append(line)
            self._line_bucket[line.id] = bucket
        for key, widget in self.list_widgets.items():
            max_page = self._max_page(key)
            self._page_by_bucket[key] = min(self._page_by_bucket.get(key, 0), max_page)
            self.list_tabs.setTabText(
                self.list_tabs.indexOf(widget),
                f"{EVENT_TYPE_LABELS.get(key, key.replace('_', ' ').title())} ({len(self._lines_by_bucket.get(key, []))})",
            )
        self._render_all_pages()
        self._sync_all_checkbox()

    def remove_lines(self, line_ids: set[UUID], visible_ids: set[UUID]) -> None:
        ids = set(line_ids)
        if not ids:
            return
        self._visible_ids = set(visible_ids)
        affected_buckets = set()
        for line_id in ids:
            bucket = self._line_bucket.pop(line_id, None)
            if bucket is not None:
                affected_buckets.add(bucket)
        for bucket in affected_buckets:
            self._lines_by_bucket[bucket] = [
                line for line in self._lines_by_bucket.get(bucket, []) if line.id not in ids
            ]
        empty_buckets = [
            bucket
            for bucket, lines in self._lines_by_bucket.items()
            if not lines and len(self._lines_by_bucket) > 1
        ]
        for bucket in empty_buckets:
            widget = self.list_widgets.pop(bucket, None)
            self._lines_by_bucket.pop(bucket, None)
            self._page_by_bucket.pop(bucket, None)
            if widget is not None:
                index = self.list_tabs.indexOf(widget)
                if index >= 0:
                    self.list_tabs.removeTab(index)
                widget.deleteLater()
        total = sum(len(lines) for lines in self._lines_by_bucket.values())
        self.event_count_label.setText(f"Events: {total}")
        for bucket, widget in self.list_widgets.items():
            max_page = self._max_page(bucket)
            self._page_by_bucket[bucket] = min(self._page_by_bucket.get(bucket, 0), max_page)
            self.list_tabs.setTabText(
                self.list_tabs.indexOf(widget),
                f"{EVENT_TYPE_LABELS.get(bucket, bucket.replace('_', ' ').title())} ({len(self._lines_by_bucket.get(bucket, []))})",
            )
        if not self.list_widgets:
            self._rebuild_tabs(["manual"])
            self._lines_by_bucket = {"manual": []}
            self._page_by_bucket = {"manual": 0}
        self._render_all_pages()
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
        all_ids = {line.id for lines in self._lines_by_bucket.values() for line in lines}
        changed_ids = all_ids - self._visible_ids
        self._visible_ids = set(all_ids)
        if changed_ids:
            self.visibility_many_changed.emit(changed_ids, True)
        for widget in self.list_widgets.values():
            widget.clearSelection()
        self._render_all_pages()
        self._sync_all_checkbox()
        self.selection_changed_for_actions.emit()
        return len(all_ids)

    def checked_line_ids(self) -> set[UUID]:
        return set(self._visible_ids)

    def toggle_line(self, line_id: UUID) -> None:
        visible = line_id not in self._visible_ids
        if visible:
            self._visible_ids.add(line_id)
        else:
            self._visible_ids.discard(line_id)
        self.visibility_changed.emit(line_id, visible)
        self.select_line(line_id)

    def select_line(self, line_id: UUID) -> bool:
        bucket = self._line_bucket.get(line_id)
        if bucket is None or bucket not in self.list_widgets:
            return False
        lines = self._lines_by_bucket.get(bucket, [])
        index = next((i for i, line in enumerate(lines) if line.id == line_id), None)
        if index is None:
            return False
        self._page_by_bucket[bucket] = index // self._page_size
        widget = self.list_widgets[bucket]
        self.list_tabs.setCurrentWidget(widget)
        self._render_bucket_page(bucket)
        widget.blockSignals(True)
        for i in range(widget.count()):
            item = widget.item(i)
            selected = item.data(Qt.ItemDataRole.UserRole) == line_id
            item.setSelected(selected)
            if selected:
                widget.scrollToItem(item)
        widget.blockSignals(False)
        self.selection_changed_for_actions.emit()
        return True

    def _toggle_all(self, checked: bool) -> None:
        bucket = self.current_bucket()
        if bucket is None:
            return
        bucket_ids = {line.id for line in self._lines_by_bucket.get(bucket, [])}
        if checked:
            changed_ids = bucket_ids - self._visible_ids
            self._visible_ids.update(bucket_ids)
        else:
            changed_ids = bucket_ids & self._visible_ids
            self._visible_ids.difference_update(bucket_ids)
        if changed_ids:
            self.visibility_many_changed.emit(changed_ids, checked)
        self._render_bucket_page(bucket)

    def _item_changed(self, item: QListWidgetItem) -> None:
        line_id = item.data(Qt.ItemDataRole.UserRole)
        visible = item.checkState() == Qt.CheckState.Checked
        if visible:
            self._visible_ids.add(line_id)
        else:
            self._visible_ids.discard(line_id)
        self.visibility_changed.emit(line_id, visible)
        self._sync_all_checkbox()

    def _selection_changed(self) -> None:
        self.selection_changed_for_actions.emit()

    def _merge_clicked(self) -> None:
        self.merge_requested.emit(self.checked_line_ids())

    def _delete_clicked(self) -> None:
        self.delete_requested.emit(self.selected_line_ids() or self.checked_line_ids())

    def _split_disconnected_clicked(self) -> None:
        selected_ids = self.selected_line_ids() or self.checked_line_ids()
        self.split_disconnected_requested.emit(selected_ids)

    def _edit_mode_changed(self, text: str) -> None:
        self.cut_mode_changed.emit(text.startswith("Draw"))
        self.seed_grow_mode_changed.emit(text.startswith("Grow"))
        if text.startswith("Add"):
            self.edit_mode_changed.emit("add")
        elif text.startswith("Erase"):
            if text.startswith("Erase drawn area"):
                self.edit_mode_changed.emit("area_erase")
            else:
                self.edit_mode_changed.emit("erase")
        else:
            self.edit_mode_changed.emit("select")

    def _sync_all_checkbox(self) -> None:
        bucket = self.current_bucket()
        if bucket is None:
            self.all_checkbox.setChecked(False)
            self._sync_page_controls()
            return
        bucket_ids = {line.id for line in self._lines_by_bucket.get(bucket, [])}
        if not bucket_ids:
            self.all_checkbox.setChecked(False)
            self._sync_page_controls()
            return
        all_checked = bucket_ids.issubset(self._visible_ids)
        self.all_checkbox.blockSignals(True)
        self.all_checkbox.setChecked(all_checked)
        self.all_checkbox.blockSignals(False)
        self._sync_page_controls()

    def current_list_widget(self) -> QListWidget | None:
        return self.list_tabs.currentWidget()

    def current_bucket(self) -> str | None:
        current_widget = self.current_list_widget()
        for key, widget in self.list_widgets.items():
            if widget is current_widget:
                return key
        return None

    def _current_tab_changed(self) -> None:
        self._render_current_page()
        self._sync_all_checkbox()

    def _change_page(self, delta: int) -> None:
        bucket = self.current_bucket()
        if bucket is None:
            return
        current = self._page_by_bucket.get(bucket, 0)
        self._page_by_bucket[bucket] = max(0, min(self._max_page(bucket), current + int(delta)))
        self._render_bucket_page(bucket)
        self._sync_all_checkbox()

    def _render_all_pages(self) -> None:
        for bucket in self.list_widgets:
            self._render_bucket_page(bucket)
        self._sync_page_controls()

    def _render_current_page(self) -> None:
        bucket = self.current_bucket()
        if bucket is not None:
            self._render_bucket_page(bucket)

    def _render_bucket_page(self, bucket: str) -> None:
        widget = self.list_widgets.get(bucket)
        if widget is None:
            return
        lines = self._lines_by_bucket.get(bucket, [])
        page = max(0, min(self._page_by_bucket.get(bucket, 0), self._max_page(bucket)))
        self._page_by_bucket[bucket] = page
        start = page * self._page_size
        end = min(len(lines), start + self._page_size)
        widget.blockSignals(True)
        widget.clear()
        for line in lines[start:end]:
            widget.addItem(self._item_for_line(line))
        widget.blockSignals(False)
        self._sync_page_controls()

    def _item_for_line(self, line: DicLine) -> QListWidgetItem:
        event_type = getattr(line, "event_type", "")
        item_label = f"{event_type}: {line.id}" if event_type else str(line.id)
        item_label = f"{item_label} ({line.size:,} px)"
        item = QListWidgetItem(item_label)
        item.setData(Qt.ItemDataRole.UserRole, line.id)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            Qt.CheckState.Checked
            if line.id in self._visible_ids
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
        return item

    def _sync_page_controls(self) -> None:
        bucket = self.current_bucket()
        if bucket is None:
            self.page_label.setText("Page 0 / 0")
            self.prev_page_button.setEnabled(False)
            self.next_page_button.setEnabled(False)
            return
        total = len(self._lines_by_bucket.get(bucket, []))
        max_page = self._max_page(bucket)
        page = min(self._page_by_bucket.get(bucket, 0), max_page)
        start = page * self._page_size + 1 if total else 0
        end = min(total, (page + 1) * self._page_size)
        self.page_label.setText(f"Page {page + 1} / {max_page + 1}  ({start}-{end} of {total})")
        enabled = self.edit_mode.isEnabled()
        self.prev_page_button.setEnabled(enabled and page > 0)
        self.next_page_button.setEnabled(enabled and page < max_page)

    def _max_page(self, bucket: str) -> int:
        total = len(self._lines_by_bucket.get(bucket, []))
        if total <= 0:
            return 0
        return (total - 1) // self._page_size

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
