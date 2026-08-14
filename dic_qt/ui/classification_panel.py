from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dic_qt.core.alignment_analysis import score_events_with_trace_alignment

CLASSIFICATION_COLUMN_TOOLTIPS = {
    "event_id": "Unique event identifier from Manual Review.",
    "classification": "likely_real means the event direction matches a selected slip/twin trace within tolerance; likely_noise means it does not.",
    "score": "Angle-match score: 1 is perfect alignment, 0 is exactly at tolerance, negative values are outside tolerance.",
    "best_mode": "Selected crystal mode whose trace direction best matches this event.",
    "best_trace_kind": "Slip or twin category of the best matching trace.",
    "best_angle_error_deg": "Smallest angle difference, in degrees, between the event direction and any selected trace.",
    "event_center_x": "Event centroid x-coordinate in DIC pixel coordinates.",
    "event_center_y": "Event centroid y-coordinate in DIC pixel coordinates.",
    "linearity": "PCA/SVD line-likeness score for the event pixels; higher values are more line-like.",
}


class ClassificationPanel(QWidget):
    def __init__(self, alignment_context_provider: Callable[[], dict], reviewed_events_provider: Callable[[], dict]) -> None:
        super().__init__()
        self._alignment_context_provider = alignment_context_provider
        self._reviewed_events_provider = reviewed_events_provider
        self._score_result = None

        layout = QVBoxLayout(self)
        title = QLabel("Classification")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title)

        description = QLabel(
            "Classify the events currently in Manual Review after manual edits are complete. "
            "This uses the ANG/transform and crystal setup from Auto Detection > Alignment."
        )
        description.setWordWrap(True)
        description.setToolTip("Classification should usually be run after auto detection, boundary cuts, event-shape analysis, and manual review.")
        layout.addWidget(description)

        controls = QGroupBox("Classification Parameters")
        controls.setToolTip("Parameters that control nearest ANG lookup and event-shape filtering before trace matching.")
        form = QFormLayout(controls)
        self.lookup_distance = self._double_spin(1.0, 500.0, 1.0, 20.0, 1)
        self.lookup_distance.setToolTip(
            "Maximum allowed distance, in DIC pixels, from an event center to the nearest transformed ANG point. "
            "If the nearest ANG point is farther away, the event is treated as unreliable."
        )
        self.minimum_linearity = self._double_spin(0.0, 1.0, 0.05, 0.0, 2)
        self.minimum_linearity.setToolTip(
            "Minimum PCA/SVD line-likeness required before an event can be considered likely real. "
            "Use 0 to rely only on trace-angle matching."
        )
        lookup_label = QLabel("Max center lookup distance")
        lookup_label.setToolTip(self.lookup_distance.toolTip())
        linearity_label = QLabel("Minimum linearity")
        linearity_label.setToolTip(self.minimum_linearity.toolTip())
        form.addRow(lookup_label, self.lookup_distance)
        form.addRow(linearity_label, self.minimum_linearity)
        layout.addWidget(controls)

        actions = QHBoxLayout()
        run_button = QPushButton("Classify Reviewed Events")
        run_button.setToolTip("Score the current Manual Review events against the selected crystal slip/twin trace vectors.")
        run_button.clicked.connect(self.classify_events)
        actions.addWidget(run_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.summary = QLabel("No reviewed events classified yet.")
        self.summary.setWordWrap(True)
        self.summary.setToolTip("Summary of matched likely-real events, likely-noise events, and matched slip/twin counts.")
        layout.addWidget(self.summary)

        self.table = QTableWidget(0, 9)
        self.table.setToolTip("Classification result preview. Hover column headers for each field meaning.")
        self.table.setHorizontalHeaderLabels(
            ["event_id", "class", "score", "best mode", "kind", "angle error", "center x", "center y", "linearity"]
        )
        self._apply_header_tooltips(
            ["event_id", "classification", "score", "best_mode", "best_trace_kind", "best_angle_error_deg", "event_center_x", "event_center_y", "linearity"]
        )
        layout.addWidget(self.table, 1)

    def classify_events(self) -> None:
        try:
            reviewed = self._reviewed_events_provider()
            if reviewed["events"].empty or reviewed["event_pixels"].empty:
                raise ValueError("Manual Review does not contain any events to classify.")
            context = self._alignment_context_provider()
            self._score_result = score_events_with_trace_alignment(
                reviewed["events"],
                reviewed["event_pixels"],
                context["dic_coords"],
                angle_tolerance_deg=float(context["angle_tolerance_deg"]),
                max_lookup_distance=float(self.lookup_distance.value()),
                min_linearity=float(self.minimum_linearity.value()),
                crystal_config=context["crystal_config"],
                morphology=None,
                included_shape_types=None,
            )
        except Exception as exc:
            self._set_error(f"Classification failed: {exc}")
            return

        counts = self._score_result["classification"].value_counts().to_dict()
        kinds = (
            self._score_result.loc[
                self._score_result["classification"] == "likely_real", "best_trace_kind"
            ].value_counts().to_dict()
            if "best_trace_kind" in self._score_result
            else {}
        )
        self._set_neutral(
            f"Classified {len(self._score_result):,} reviewed events | "
            f"matched {counts.get('likely_real', 0):,} | noise {counts.get('likely_noise', 0):,} | "
            f"matched slips {kinds.get('slip', 0):,} | matched twins {kinds.get('twin', 0):,}"
        )
        self._fill_table(
            self._score_result.head(300),
            ["event_id", "classification", "score", "best_mode", "best_trace_kind", "best_angle_error_deg", "event_center_x", "event_center_y", "linearity"],
        )

    def _fill_table(self, frame, columns: list[str]) -> None:
        self.table.setRowCount(len(frame))
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self._apply_header_tooltips(columns)
        if len(frame) == 0:
            return
        for row_index, row in enumerate(frame[columns].itertuples(index=False)):
            for col_index, value in enumerate(row):
                self.table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()

    def _apply_header_tooltips(self, columns: list[str]) -> None:
        for index, column in enumerate(columns):
            item = self.table.horizontalHeaderItem(index)
            if item is not None:
                item.setToolTip(CLASSIFICATION_COLUMN_TOOLTIPS.get(column, column))

    def _set_neutral(self, text: str) -> None:
        self.summary.setText(text)
        self.summary.setStyleSheet("color: #374151;")

    def _set_error(self, text: str) -> None:
        self.summary.setText(text)
        self.summary.setStyleSheet("color: #b91c1c; font-weight: 700;")

    @staticmethod
    def _double_spin(minimum: float, maximum: float, step: float, value: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin
