from __future__ import annotations

from uuid import UUID

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.models import DicLine


class LineListPanel(QWidget):
    visibility_changed = Signal(object, bool)
    selection_changed_for_actions = Signal()
    select_all_requested = Signal()
    merge_requested = Signal(object)
    delete_requested = Signal(object)
    cut_mode_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.cut_checkbox = QCheckBox("Cut Event")
        self.cut_checkbox.setEnabled(False)
        self.event_count_label = QLabel("Events: 0")
        self.all_checkbox = QCheckBox("All")
        self.manual_label = QLabel("Manual")
        self.manual_label.setStyleSheet("color: blue")
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.select_all_button = QPushButton("Select All")
        self.merge_button = QPushButton("Merge")
        self.delete_button = QPushButton("Delete Selected")
        self.select_all_button.setEnabled(False)
        self.merge_button.setEnabled(False)
        self.delete_button.setEnabled(False)

        header = QHBoxLayout()
        header.addWidget(self.all_checkbox)
        header.addStretch(1)
        header.addWidget(self.manual_label)

        buttons = QHBoxLayout()
        buttons.addWidget(self.select_all_button)
        buttons.addWidget(self.merge_button)
        buttons.addWidget(self.delete_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.cut_checkbox)
        layout.addWidget(self.event_count_label)
        layout.addLayout(header)
        layout.addWidget(self.list_widget, 1)
        layout.addLayout(buttons)

        self.all_checkbox.toggled.connect(self._toggle_all)
        self.list_widget.itemChanged.connect(self._item_changed)
        self.list_widget.itemSelectionChanged.connect(self._selection_changed)
        self.select_all_button.clicked.connect(lambda: self.select_all_requested.emit())
        self.merge_button.clicked.connect(self._merge_clicked)
        self.delete_button.clicked.connect(self._delete_clicked)
        self.cut_checkbox.toggled.connect(self.cut_mode_changed.emit)

    def set_actions_enabled(self, enabled: bool, cut_enabled: bool) -> None:
        self.select_all_button.setEnabled(enabled)
        self.merge_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)
        self.cut_checkbox.setEnabled(cut_enabled)
        if not cut_enabled:
            self.cut_checkbox.setChecked(False)

    def set_lines(self, lines: list[DicLine], visible_ids: set[UUID]) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        valid_lines = [line for line in lines if isinstance(line, DicLine)]
        self.event_count_label.setText(f"Events: {len(valid_lines)}")
        for line in sorted(valid_lines, key=_line_sort_distance):
            item = QListWidgetItem(str(line.id))
            item.setData(Qt.ItemDataRole.UserRole, line.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if line.id in visible_ids
                else Qt.CheckState.Unchecked
            )
            if line.is_manual:
                item.setForeground(Qt.GlobalColor.blue)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)
        self._sync_all_checkbox()

    def selected_line_ids(self) -> set[UUID]:
        return {
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.list_widget.selectedItems()
        }

    def select_all_lines(self) -> int:
        self._toggle_all(True)
        self._sync_all_checkbox()
        return self.list_widget.count()

    def checked_line_ids(self) -> set[UUID]:
        ids: set[UUID] = set()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                ids.add(item.data(Qt.ItemDataRole.UserRole))
        return ids

    def toggle_line(self, line_id: UUID) -> None:
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) != line_id:
                continue
            new_state = (
                Qt.CheckState.Unchecked
                if item.checkState() == Qt.CheckState.Checked
                else Qt.CheckState.Checked
            )
            item.setCheckState(new_state)
            item.setSelected(new_state == Qt.CheckState.Checked)
            break

    def _toggle_all(self, checked: bool) -> None:
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
        self.list_widget.blockSignals(False)
        for i in range(self.list_widget.count()):
            line_id = self.list_widget.item(i).data(Qt.ItemDataRole.UserRole)
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

    def _sync_all_checkbox(self) -> None:
        if self.list_widget.count() == 0:
            self.all_checkbox.setChecked(False)
            return
        all_checked = all(
            self.list_widget.item(i).checkState() == Qt.CheckState.Checked
            for i in range(self.list_widget.count())
        )
        self.all_checkbox.blockSignals(True)
        self.all_checkbox.setChecked(all_checked)
        self.all_checkbox.blockSignals(False)


def _line_sort_distance(line: DicLine) -> float:
    if not line.points:
        return 0.0
    x = sum(p.x for p in line.points) / len(line.points)
    y = sum(p.y for p in line.points) / len(line.points)
    return x * x + y * y
