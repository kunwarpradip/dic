from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class FileSlot:
    key: str
    label: str
    required: bool
    dialog_filter: str


FILE_SLOTS = [
    FileSlot("bln_image", "BLN Image", True, "Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg);;All files (*)"),
    FileSlot("grx_image", "GRX Image", False, "Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg);;All files (*)"),
    FileSlot("ebsd_aligned_image", "EBSD Aligned Image", True, "Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg);;All files (*)"),
    FileSlot("ebsd_boundary_image", "EBSD Boundary Image", True, "Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg);;All files (*)"),
    FileSlot("ang_file", "ANG File", True, "ANG files (*.ang);;All files (*)"),
    FileSlot("transform_json", "Transformation JSON", True, "JSON files (*.json);;All files (*)"),
]

ANALYSIS_DEFAULTS_PATH = Path(__file__).resolve().parents[2] / ".dic_qt_analysis_defaults.json"


class ProjectFilesPanel(QWidget):
    files_changed = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.path_edits: dict[str, QLineEdit] = {}
        self.status_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Project Files")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        subtitle = QLabel(
            "Load the core DIC, EBSD, ANG, and transformation files first. "
            "The summary below checks the small set of metadata that is easiest to get wrong."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #4b5563;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        defaults_actions = QHBoxLayout()
        load_defaults_button = QPushButton("Load Analysis Defaults")
        load_defaults_button.setToolTip("Load the file paths previously saved as defaults for this local analysis workspace.")
        load_defaults_button.clicked.connect(lambda checked=False: self.load_analysis_defaults(refresh=True))
        save_defaults_button = QPushButton("Set Current Paths As Defaults")
        save_defaults_button.setToolTip("Save the currently filled file paths as the reusable defaults for this analysis workspace.")
        save_defaults_button.clicked.connect(lambda checked=False: self.save_analysis_defaults())
        clear_defaults_button = QPushButton("Clear Saved Defaults")
        clear_defaults_button.setToolTip("Remove the saved local analysis defaults file.")
        clear_defaults_button.clicked.connect(lambda checked=False: self.clear_analysis_defaults())
        defaults_actions.addWidget(load_defaults_button)
        defaults_actions.addWidget(save_defaults_button)
        defaults_actions.addWidget(clear_defaults_button)
        defaults_actions.addStretch(1)
        layout.addLayout(defaults_actions)

        files_group = QGroupBox("Files")
        files_layout = QGridLayout(files_group)
        files_layout.setColumnStretch(1, 1)
        for row, slot in enumerate(FILE_SLOTS):
            label_text = f"{slot.label}{' *' if slot.required else ' (optional)'}"
            label = QLabel(label_text)
            path_edit = QLineEdit()
            path_edit.setPlaceholderText("Choose file...")
            browse_button = QPushButton("Browse")
            status = QLabel("Not loaded")
            status.setStyleSheet("color: #6b7280;")

            self.path_edits[slot.key] = path_edit
            self.status_labels[slot.key] = status

            browse_button.clicked.connect(lambda checked=False, s=slot: self.choose_file(s))
            path_edit.editingFinished.connect(self.refresh_summary)

            files_layout.addWidget(label, row, 0)
            files_layout.addWidget(path_edit, row, 1)
            files_layout.addWidget(browse_button, row, 2)
            files_layout.addWidget(status, row, 3)
        layout.addWidget(files_group)

        self.summary_table = QTableWidget(0, 3)
        self.summary_table.setHorizontalHeaderLabels(["File", "Status", "Important Checks"])
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.setWordWrap(True)
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.summary_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.summary_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.summary_table.setStyleSheet(
            "QTableWidget { gridline-color: #e5e7eb; }"
            "QHeaderView::section { background: #f3f4f6; font-weight: 700; padding: 6px; }"
        )
        layout.addWidget(self.summary_table, 1)

        self.overall_label = QLabel("Required files are not loaded yet.")
        self.overall_label.setWordWrap(True)
        self.overall_label.setStyleSheet(
            "padding: 10px; border-radius: 6px; background: #f9fafb; color: #374151;"
        )
        layout.addWidget(self.overall_label)

        if ANALYSIS_DEFAULTS_PATH.exists():
            self.load_analysis_defaults(refresh=False)
        self.refresh_summary()

    def choose_file(self, slot: FileSlot) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Choose {slot.label}",
            str(Path.cwd()),
            slot.dialog_filter,
        )
        if not path:
            return
        self.path_edits[slot.key].setText(path)
        self.refresh_summary()

    def selected_files(self) -> dict[str, str]:
        return {
            slot.key: self.path_edits[slot.key].text().strip()
            for slot in FILE_SLOTS
        }

    def set_selected_files(self, paths: dict[str, str], refresh: bool = True) -> None:
        for slot in FILE_SLOTS:
            if slot.key in paths:
                self.path_edits[slot.key].setText(str(paths.get(slot.key, "") or ""))
        if refresh:
            self.refresh_summary()

    def load_analysis_defaults(self, refresh: bool = True) -> None:
        if not ANALYSIS_DEFAULTS_PATH.exists():
            self.overall_label.setText(f"No saved analysis defaults found at {ANALYSIS_DEFAULTS_PATH}.")
            self.overall_label.setStyleSheet(
                "padding: 10px; border-radius: 6px; background: #f9fafb; color: #374151;"
            )
            return
        try:
            data = json.loads(ANALYSIS_DEFAULTS_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            self.overall_label.setText(f"Could not load saved analysis defaults: {exc}")
            self.overall_label.setStyleSheet(
                "padding: 10px; border-radius: 6px; background: #fee2e2; color: #7f1d1d;"
            )
            return
        paths = data.get("paths", data)
        if not isinstance(paths, dict):
            self.overall_label.setText("Saved analysis defaults file is not in the expected format.")
            self.overall_label.setStyleSheet(
                "padding: 10px; border-radius: 6px; background: #fee2e2; color: #7f1d1d;"
            )
            return
        for key, path in paths.items():
            if key in self.path_edits:
                self.path_edits[key].setText(str(path))
        if refresh:
            self.refresh_summary()

    def save_analysis_defaults(self) -> None:
        payload = {
            "version": 1,
            "paths": self.selected_files(),
        }
        ANALYSIS_DEFAULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.refresh_summary()
        self.overall_label.setText(f"Saved current paths as analysis defaults:\n{ANALYSIS_DEFAULTS_PATH}")
        self.overall_label.setStyleSheet(
            "padding: 10px; border-radius: 6px; background: #dcfce7; color: #14532d;"
        )

    def clear_analysis_defaults(self) -> None:
        if ANALYSIS_DEFAULTS_PATH.exists():
            ANALYSIS_DEFAULTS_PATH.unlink()
        self.refresh_summary()
        self.overall_label.setText("Cleared saved analysis defaults.")
        self.overall_label.setStyleSheet(
            "padding: 10px; border-radius: 6px; background: #f9fafb; color: #374151;"
        )

    def refresh_summary(self) -> None:
        paths = self.selected_files()
        summaries: dict[str, dict[str, Any]] = {}
        for slot in FILE_SLOTS:
            summaries[slot.key] = self._summarize_slot(slot, paths[slot.key])

        self._add_cross_checks(summaries)
        self._render_summary(summaries)
        self.files_changed.emit(paths)

    def _summarize_slot(self, slot: FileSlot, path_text: str) -> dict[str, Any]:
        if not path_text:
            status = "Missing" if slot.required else "Optional"
            return {"slot": slot, "status": status, "checks": "No file selected.", "path": ""}

        path = Path(path_text).expanduser()
        if not path.exists():
            return {"slot": slot, "status": "Missing", "checks": "Path does not exist.", "path": str(path)}

        try:
            if slot.key.endswith("image"):
                return self._image_summary(slot, path)
            if slot.key == "ang_file":
                return self._ang_summary(slot, path)
            if slot.key == "transform_json":
                return self._transform_summary(slot, path)
        except Exception as exc:
            return {"slot": slot, "status": "Error", "checks": str(exc), "path": str(path)}

        return {"slot": slot, "status": "Loaded", "checks": f"Size: {format_file_size(path)}", "path": str(path)}

    def _image_summary(self, slot: FileSlot, path: Path) -> dict[str, Any]:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            frames = getattr(img, "n_frames", 1)
        checks = [
            f"Dimensions: {width} x {height}",
            f"Mode: {mode}",
            f"Frames: {frames}",
            f"Size: {format_file_size(path)}",
        ]
        return {
            "slot": slot,
            "status": "Loaded",
            "checks": " | ".join(checks),
            "path": str(path),
            "width": width,
            "height": height,
            "mode": mode,
        }

    def _ang_summary(self, slot: FileSlot, path: Path) -> dict[str, Any]:
        header = read_ang_header(path)
        xstep = header.get("XSTEP")
        ystep = header.get("YSTEP")
        nrows = header.get("NROWS")
        ncols_odd = header.get("NCOLS_ODD")
        ncols_even = header.get("NCOLS_EVEN")
        column_count = header.get("COLUMN_COUNT")
        units = header.get("COLUMN_UNITS")

        checks = [
            f"XSTEP: {xstep or 'not found'}",
            f"YSTEP: {ystep or 'not found'}",
            f"Rows: {nrows or 'not found'}",
            f"Cols odd/even: {ncols_odd or 'not found'} / {ncols_even or 'not found'}",
            f"Columns: {column_count or 'not found'}",
            f"Euler units: {shorten_units(units)}",
            f"Size: {format_file_size(path)}",
        ]
        status = "Loaded" if xstep and ystep else "Check"
        return {
            "slot": slot,
            "status": status,
            "checks": " | ".join(checks),
            "path": str(path),
            "xstep": xstep,
            "ystep": ystep,
            "nrows": nrows,
            "ncols_odd": ncols_odd,
            "ncols_even": ncols_even,
        }

    def _transform_summary(self, slot: FileSlot, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text())
        order = data.get("polynomial_order", "not found")
        family = data.get("transform_family", "not found")
        point_count = len(data.get("matched_point_ids", []) or [])
        dic_shape = data.get("dic_image_shape")
        ebsd_shape = data.get("ebsd_image_shape")
        forward_rmse = data.get("rmse_forward_pixels")
        inverse_rmse = data.get("rmse_inverse_pixels")

        checks = [
            f"Transform: {family}, order {order}",
            f"Control points: {point_count}",
            f"DIC shape in JSON: {format_shape(dic_shape)}",
            f"EBSD shape in JSON: {format_shape(ebsd_shape)}",
            f"Forward RMSE: {format_float(forward_rmse)} px",
            f"Inverse RMSE: {format_float(inverse_rmse)} px",
            f"Size: {format_file_size(path)}",
        ]
        return {
            "slot": slot,
            "status": "Loaded",
            "checks": " | ".join(checks),
            "path": str(path),
            "dic_shape": dic_shape,
            "ebsd_shape": ebsd_shape,
        }

    def _add_cross_checks(self, summaries: dict[str, dict[str, Any]]) -> None:
        bln = summaries.get("bln_image", {})
        bln_size = image_size_from_summary(bln)
        if not bln_size:
            return

        for key, label in [
            ("grx_image", "BLN"),
            ("ebsd_aligned_image", "BLN"),
            ("ebsd_boundary_image", "BLN"),
        ]:
            summary = summaries.get(key, {})
            size = image_size_from_summary(summary)
            if not size:
                continue
            match_text = "matches" if size == bln_size else "does not match"
            summary["checks"] = f"{summary['checks']} | Dimension check: {match_text} {label}"
            if key in {"ebsd_aligned_image", "ebsd_boundary_image"} and size != bln_size:
                summary["status"] = "Check"

        transform = summaries.get("transform_json", {})
        dic_shape = transform.get("dic_shape")
        transform_dic_size = size_from_shape(dic_shape)
        if transform_dic_size:
            match_text = "matches" if transform_dic_size == bln_size else "does not match"
            transform["checks"] = f"{transform['checks']} | JSON DIC shape {match_text} BLN"
            if transform_dic_size != bln_size:
                transform["status"] = "Check"

    def _render_summary(self, summaries: dict[str, dict[str, Any]]) -> None:
        self.summary_table.setRowCount(len(FILE_SLOTS))
        missing_required = []
        check_count = 0
        error_count = 0

        for row, slot in enumerate(FILE_SLOTS):
            summary = summaries[slot.key]
            status = summary["status"]
            if slot.required and status == "Missing":
                missing_required.append(slot.label)
            if status == "Check":
                check_count += 1
            if status == "Error":
                error_count += 1

            self.status_labels[slot.key].setText(status)
            self.status_labels[slot.key].setStyleSheet(status_style(status))
            values = [slot.label, status, summary["checks"]]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.summary_table.setItem(row, col, item)

        self.summary_table.resizeRowsToContents()

        if error_count:
            text = f"{error_count} file could not be read. Fix those first."
            style = "padding: 10px; border-radius: 6px; background: #fee2e2; color: #7f1d1d;"
        elif missing_required:
            text = "Missing required files: " + ", ".join(missing_required)
            style = "padding: 10px; border-radius: 6px; background: #fff7ed; color: #7c2d12;"
        elif check_count:
            text = f"Files loaded, but {check_count} dimension or metadata check needs attention."
            style = "padding: 10px; border-radius: 6px; background: #fef3c7; color: #78350f;"
        else:
            text = "All required files are loaded and the key dimensions look consistent."
            style = "padding: 10px; border-radius: 6px; background: #dcfce7; color: #14532d;"
        self.overall_label.setText(text)
        self.overall_label.setStyleSheet(style)


def read_ang_header(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            body = stripped[1:].strip()
            if body.upper() == "HEADER: END":
                break
            match = re.match(r"^([A-Za-z0-9_]+)\s*[:=]?\s*(.+?)\s*$", body)
            if match:
                header[match.group(1).strip().upper()] = match.group(2).strip()
    return header


def format_file_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


def format_shape(shape: Any) -> str:
    if not isinstance(shape, list) or len(shape) < 2:
        return "not found"
    return " x ".join(str(value) for value in shape)


def size_from_shape(shape: Any) -> tuple[int, int] | None:
    if not isinstance(shape, list) or len(shape) < 2:
        return None
    try:
        height = int(shape[0])
        width = int(shape[1])
    except (TypeError, ValueError):
        return None
    return width, height


def image_size_from_summary(summary: dict[str, Any]) -> tuple[int, int] | None:
    if "width" not in summary or "height" not in summary:
        return None
    return int(summary["width"]), int(summary["height"])


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "not found"


def shorten_units(units: str | None) -> str:
    if not units:
        return "not found"
    pieces = [piece.strip() for piece in units.split(",")]
    if len(pieces) >= 3:
        return ", ".join(pieces[:3])
    return units


def status_style(status: str) -> str:
    if status == "Loaded":
        return "color: #166534; font-weight: 600;"
    if status == "Check":
        return "color: #92400e; font-weight: 600;"
    if status == "Error":
        return "color: #991b1b; font-weight: 600;"
    if status == "Optional":
        return "color: #6b7280;"
    return "color: #b45309; font-weight: 600;"
