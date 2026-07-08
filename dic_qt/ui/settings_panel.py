from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class DetectionSettings:
    intensity_difference_tolerance: int
    bfl_tolerance: float
    min_intensity: int


class SettingsPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.choose_file_button = QPushButton("Choose File")
        self.create_event_checkbox = QCheckBox("Create event from click")
        self.create_event_checkbox.setChecked(True)
        self.reset_button = QPushButton("Reset settings")

        self.intensity_tolerance = QSpinBox()
        self.intensity_tolerance.setRange(0, 255)
        self.intensity_tolerance.setValue(75)

        self.bfl_tolerance = QDoubleSpinBox()
        self.bfl_tolerance.setRange(0.01, 1000.0)
        self.bfl_tolerance.setDecimals(2)
        self.bfl_tolerance.setSingleStep(0.5)
        self.bfl_tolerance.setValue(7.0)

        self.min_intensity = QSpinBox()
        self.min_intensity.setRange(0, 255)
        self.min_intensity.setValue(120)

        form = QFormLayout()
        form.addRow("Pixel intensity change tolerance", self.intensity_tolerance)
        form.addRow("Best fit line tolerance", self.bfl_tolerance)
        form.addRow("Minimum intensity", self.min_intensity)

        layout = QVBoxLayout(self)
        layout.addWidget(self.choose_file_button)
        layout.addWidget(self.create_event_checkbox)
        layout.addLayout(form)
        layout.addWidget(self.reset_button)
        layout.addStretch(1)

        self.reset_button.clicked.connect(self.reset_defaults)

    def settings(self) -> DetectionSettings:
        return DetectionSettings(
            intensity_difference_tolerance=self.intensity_tolerance.value(),
            bfl_tolerance=self.bfl_tolerance.value(),
            min_intensity=self.min_intensity.value(),
        )

    def reset_defaults(self) -> None:
        self.intensity_tolerance.setValue(75)
        self.bfl_tolerance.setValue(7.0)
        self.min_intensity.setValue(120)
