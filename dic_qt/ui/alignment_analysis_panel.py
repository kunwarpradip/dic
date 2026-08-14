from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QGraphicsScene,
    QGraphicsView,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

ALIGNMENT_PARAMETER_DEFAULTS_PATH = Path(__file__).resolve().parents[2] / ".dic_qt_alignment_parameter_defaults.json"
MAX_SHAPE_OVERLAY_DIM = 2400
SHAPE_OVERLAY_TYPES = ["linear", "blob_like", "irregular", "fragmented", "small_noise"]
SHAPE_PARAMETER_TOOLTIPS = {
    "Min pixels": "Events with fewer pixels than this are classified as small noise.",
    "Linearity": "Minimum PCA/SVD linearity score used to call an event line-like. Higher values are stricter.",
    "Aspect": "Minimum bounding-box aspect ratio used to help identify elongated events.",
    "Blob density": "Dense, compact events above this value are more likely to be labeled blob-like.",
}
SHAPE_TYPE_TOOLTIPS = {
    "linear": "Long, line-like events that pass the linearity and aspect checks.",
    "blob_like": "Compact, dense events that look more like filled spots than slip bands.",
    "irregular": "Events that are neither cleanly linear nor compact enough to be blob-like.",
    "fragmented": "Events with multiple disconnected components before manual review.",
    "small_noise": "Events below the minimum pixel threshold.",
}

from dic_qt.core.alignment_analysis import (
    CRYSTAL_MODE_OPTIONS,
    CRYSTAL_MODE_TRACE_COUNTS,
    DEFAULT_CRYSTAL_MODES,
    EBSD_REFERENCE_FRAMES,
    build_trace_preview,
    compute_event_morphology_features,
    default_crystal_config,
    dic_coordinate_metrics,
    load_alignment_transform_parameters,
    load_ang_core_data,
    ang_grid_to_ebsd_pixel_mapping,
    transform_ang_coordinates_to_dic,
)


def display_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[..., :3]
    arr = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


class ZoomImageView(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._image_bytes: bytes | None = None
        self._image_shape: tuple[int, int] | None = None
        self._zoom_level = 0
        self._space_pan_active = False
        self._panning = False

    def set_array(self, image: np.ndarray | None) -> None:
        self.scene().clear()
        if image is None:
            self._image_bytes = None
            self._image_shape = None
            return
        arr = np.ascontiguousarray(display_rgb(image))
        height, width, _ = arr.shape
        self._image_bytes = arr.tobytes()
        qimage = QImage(self._image_bytes, width, height, width * 3, QImage.Format.Format_RGB888)
        self.scene().addPixmap(QPixmap.fromImage(qimage))
        self._image_shape = (height, width)
        self.scene().setSceneRect(0, 0, width, height)
        self.fit_to_window()

    def fit_to_window(self) -> None:
        if self._image_shape is None:
            return
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = 0

    def zoom_in(self) -> None:
        self._apply_zoom(1)

    def zoom_out(self) -> None:
        self._apply_zoom(-1)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.text() in ("+", "="):
            self.zoom_in()
            event.accept()
            return
        if event.text() in ("-", "_"):
            self.zoom_out()
            event.accept()
            return
        if event.text() == "0":
            self.fit_to_window()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan_active = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan_active = False
            if not self._panning:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if (
            event.button() == Qt.MouseButton.MiddleButton
            or event.button() == Qt.MouseButton.RightButton
            or (event.button() == Qt.MouseButton.LeftButton and self._space_pan_active)
        ):
            self._panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            super().mouseReleaseEvent(event)
            self._panning = False
            if not self._space_pan_active:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
            return
        super().mouseReleaseEvent(event)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._image_shape is None:
            return
        if event.angleDelta().y() > 0:
            self.zoom_in()
        elif event.angleDelta().y() < 0:
            self.zoom_out()

    def _apply_zoom(self, direction: int) -> None:
        if self._image_shape is None:
            return
        if direction > 0 and self._zoom_level < 20:
            self._zoom_level += 1
            factor = 1.25
        elif direction < 0 and self._zoom_level > 0:
            self._zoom_level -= 1
            factor = 0.8
        else:
            return
        if self._zoom_level == 0:
            self.fit_to_window()
            return
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scale(factor, factor)


class AlignmentAnalysisPanel(QWidget):
    shape_analysis_ready = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self._project_files: dict[str, str] = {}
        self._crystal_config = default_crystal_config()
        self._transform: dict | None = None
        self._ang_core: dict | None = None
        self._dic_coords = None
        self._trace_preview = None
        self._event_info: dict | None = None
        self._morphology = None
        self._score_result = None
        self._boundary_cut_result: dict | None = None

        layout = QVBoxLayout(self)
        title = QLabel("Alignment")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title)
        self.status_label = QLabel("Use Project Files and Boundary Cuts, then run the alignment steps from left to right.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #374151;")
        layout.addWidget(self.status_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_crystal_tab()
        self._build_summary_tab()
        self._build_trace_tab()
        self._build_shape_tab()
        self._refresh_mode_list()
        self._load_parameter_defaults()

    def set_project_files(self, files: dict[str, str]) -> None:
        self._project_files = dict(files)
        self._refresh_file_summary_text()

    def set_boundary_cut_result(self, result: dict | None) -> None:
        self._boundary_cut_result = result
        self._morphology = None
        self._score_result = None
        if result is None:
            self._event_info = None
            if hasattr(self, "event_source_summary"):
                self.event_source_summary.setText("Run Boundary Cuts first. Alignment will use those cut events automatically.")
            if hasattr(self, "shape_summary"):
                self.shape_summary.setText("No event shape analysis yet.")
            self._clear_shape_overlays()
            return

        events = result.get("events")
        event_pixels = result.get("event_pixels")
        if events is None or event_pixels is None:
            self._event_info = None
            if hasattr(self, "event_source_summary"):
                self.event_source_summary.setText("Boundary cut result is missing event tables. Run Boundary Cuts again.")
            self._clear_shape_overlays()
            return

        metrics = self._event_metrics(events, event_pixels)
        self._event_info = {"events": events.copy(), "event_pixels": event_pixels.copy(), "metrics": metrics}
        if hasattr(self, "event_source_summary"):
            self.event_source_summary.setText(
                f"Using boundary-cut events from the previous step: "
                f"{metrics['event_count']:,} events, {metrics['pixel_count']:,} pixels."
            )
        if hasattr(self, "shape_summary"):
            self.shape_summary.setText("Boundary-cut events are ready. Click Analyze Shapes.")
        self._clear_shape_overlays()

    def _build_crystal_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        controls = QGroupBox("Crystal Setup")
        form = QFormLayout(controls)
        self.crystal_structure = QComboBox()
        self.crystal_structure.addItems(list(CRYSTAL_MODE_OPTIONS.keys()))
        self.crystal_structure.currentTextChanged.connect(self._refresh_mode_list)
        self.ebsd_reference = QComboBox()
        self.ebsd_reference.addItems(EBSD_REFERENCE_FRAMES)
        self.lattice_a = self._double_spin(0.000001, 100.0, 0.01, 2.95, 6)
        self.lattice_c = self._double_spin(0.000001, 100.0, 0.01, 4.686, 6)
        self.slip_tolerance = self._double_spin(0.0, 90.0, 0.5, 5.0, 2)
        self.twin_tolerance = self._double_spin(0.0, 90.0, 0.5, 7.0, 2)
        form.addRow("Crystal structure", self.crystal_structure)
        form.addRow("EBSD reference frame", self.ebsd_reference)
        form.addRow("<a> lattice parameter", self.lattice_a)
        form.addRow("<c> lattice parameter", self.lattice_c)
        form.addRow("Slip tolerance", self.slip_tolerance)
        form.addRow("Twin tolerance", self.twin_tolerance)
        apply_button = QPushButton("Apply Crystal Setup")
        apply_button.clicked.connect(self._apply_crystal_setup)
        form.addRow("", apply_button)
        default_button = QPushButton("Set As Default")
        default_button.setToolTip("Save the current crystal setup as the default for future app sessions.")
        default_button.clicked.connect(self._save_parameter_defaults)
        form.addRow("", default_button)
        layout.addWidget(controls)

        right = QGroupBox("Active Modes And Priority")
        right_layout = QVBoxLayout(right)
        self.mode_list = QListWidget()
        self.mode_list.setToolTip("Check modes to include. Order matters during scoring.")
        self.mode_list.itemChanged.connect(self._apply_crystal_setup)
        right_layout.addWidget(self.mode_list)
        buttons = QHBoxLayout()
        up_button = QPushButton("Up")
        down_button = QPushButton("Down")
        up_button.clicked.connect(lambda: self._move_mode(-1))
        down_button.clicked.connect(lambda: self._move_mode(1))
        buttons.addWidget(up_button)
        buttons.addWidget(down_button)
        right_layout.addLayout(buttons)
        self.crystal_summary = QLabel("")
        self.crystal_summary.setWordWrap(True)
        right_layout.addWidget(self.crystal_summary)
        layout.addWidget(right, 1)
        self.tabs.addTab(tab, "Crystal Setup")

    def _build_summary_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        run_button = QPushButton("Load ANG And Transformation Summary")
        run_button.clicked.connect(self._load_alignment_summary)
        layout.addWidget(run_button)
        self.summary_table = QTableWidget(0, 3)
        self.summary_table.setHorizontalHeaderLabels(["Item", "Value", "Description"])
        layout.addWidget(self.summary_table, 1)
        self.tabs.addTab(tab, "ANG / Transform")

    def _build_trace_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.trace_preview_rows = self._spin(1, 100000, 100, 100)
        controls.addWidget(QLabel("Preview rows"))
        controls.addWidget(self.trace_preview_rows)
        run_button = QPushButton("Generate Trace Vectors")
        run_button.clicked.connect(self._generate_trace_vectors)
        controls.addWidget(run_button)
        default_button = QPushButton("Set As Default")
        default_button.setToolTip("Save the current trace-vector preview settings as the default.")
        default_button.clicked.connect(self._save_parameter_defaults)
        controls.addWidget(default_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.trace_summary = QLabel("No trace vectors generated yet.")
        self.trace_summary.setWordWrap(True)
        layout.addWidget(self.trace_summary)
        self.trace_table = QTableWidget(0, 8)
        self.trace_table.setHorizontalHeaderLabels(["x_dic", "y_dic", "phi1", "PHI", "phi2", "selected_trace_count", "trace_labels", "trace_vectors"])
        layout.addWidget(self.trace_table, 1)
        self.tabs.addTab(tab, "Trace Vectors")

    def _build_shape_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        source_box = QGroupBox("Event Source")
        source_layout = QVBoxLayout(source_box)
        self.event_source_summary = QLabel("Run Boundary Cuts first. Alignment will use those cut events automatically.")
        self.event_source_summary.setWordWrap(True)
        source_layout.addWidget(self.event_source_summary)
        layout.addWidget(source_box)
        params = QHBoxLayout()
        self.shape_min_pixels = self._spin(1, 10000, 1, 40)
        self.shape_linearity = self._double_spin(0.0, 1.0, 0.01, 0.80, 2)
        self.shape_aspect = self._double_spin(1.0, 100.0, 0.5, 3.0, 2)
        self.shape_density = self._double_spin(0.01, 1.0, 0.01, 0.25, 2)
        self.shape_min_pixels.setToolTip(SHAPE_PARAMETER_TOOLTIPS["Min pixels"])
        self.shape_linearity.setToolTip(SHAPE_PARAMETER_TOOLTIPS["Linearity"])
        self.shape_aspect.setToolTip(SHAPE_PARAMETER_TOOLTIPS["Aspect"])
        self.shape_density.setToolTip(SHAPE_PARAMETER_TOOLTIPS["Blob density"])
        for label, widget in [
            ("Min pixels", self.shape_min_pixels),
            ("Linearity", self.shape_linearity),
            ("Aspect", self.shape_aspect),
            ("Blob density", self.shape_density),
        ]:
            label_widget = QLabel(label)
            label_widget.setToolTip(SHAPE_PARAMETER_TOOLTIPS[label])
            params.addWidget(label_widget)
            params.addWidget(widget)
        run_button = QPushButton("Analyze Shapes")
        run_button.setToolTip("Measure each cut event's size, connected components, line fit, aspect ratio, and density.")
        run_button.clicked.connect(self._analyze_shapes)
        params.addWidget(run_button)
        send_button = QPushButton("Send Selected Types To Manual Review")
        send_button.setToolTip("Populate Manual Review with the checked event-shape types.")
        send_button.clicked.connect(self._send_shape_analysis_to_review)
        params.addWidget(send_button)
        default_button = QPushButton("Set As Default")
        default_button.setToolTip("Save the current event-shape parameters as the default.")
        default_button.clicked.connect(self._save_parameter_defaults)
        params.addWidget(default_button)
        layout.addLayout(params)
        send_box = QGroupBox("Send To Manual Review")
        send_layout = QHBoxLayout(send_box)
        self.shape_send_checks: dict[str, QCheckBox] = {}
        for event_type in SHAPE_OVERLAY_TYPES:
            checkbox = QCheckBox(event_type.replace("_", " ").title())
            checkbox.setChecked(event_type in {"linear", "irregular"})
            checkbox.setToolTip(
                f"{SHAPE_TYPE_TOOLTIPS[event_type]} Check this to send these events to Manual Review."
            )
            self.shape_send_checks[event_type] = checkbox
            send_layout.addWidget(checkbox)
        send_layout.addStretch(1)
        layout.addWidget(send_box)
        self.shape_summary = QLabel("No event shape analysis yet.")
        self.shape_summary.setWordWrap(True)
        layout.addWidget(self.shape_summary)
        self.shape_overlay_status = QLabel("Shape overlays will appear after event-shape analysis.")
        self.shape_overlay_status.setWordWrap(True)
        layout.addWidget(self.shape_overlay_status)
        self.shape_overlay_tabs = QTabWidget()
        self.shape_overlay_views: dict[str, ZoomImageView] = {}
        for event_type in SHAPE_OVERLAY_TYPES:
            view = ZoomImageView()
            view.setMinimumSize(520, 320)
            view.setToolTip("Zoom with =/+ and -, reset with 0, mouse wheel zooms, and space-drag pans.")
            self.shape_overlay_views[event_type] = view
            self.shape_overlay_tabs.addTab(view, event_type.replace("_", " ").title())
        layout.addWidget(self.shape_overlay_tabs, 2)
        self.tabs.addTab(tab, "Event Shapes")

    def _refresh_mode_list(self, _text: str | None = None) -> None:
        if not hasattr(self, "mode_list"):
            return
        structure = self.crystal_structure.currentText()
        defaults = set(DEFAULT_CRYSTAL_MODES.get(structure, []))
        self.mode_list.clear()
        for mode in CRYSTAL_MODE_OPTIONS[structure]:
            item = QListWidgetItem(mode)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if mode in defaults else Qt.CheckState.Unchecked)
            self.mode_list.addItem(item)
        self._apply_crystal_setup()

    def _move_mode(self, direction: int) -> None:
        row = self.mode_list.currentRow()
        target = row + direction
        if row < 0 or target < 0 or target >= self.mode_list.count():
            return
        item = self.mode_list.takeItem(row)
        self.mode_list.insertItem(target, item)
        self.mode_list.setCurrentRow(target)
        self._apply_crystal_setup()

    def _apply_crystal_setup(self) -> None:
        active_modes = [
            self.mode_list.item(i).text()
            for i in range(self.mode_list.count())
            if self.mode_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        self._crystal_config = {
            "crystal_structure": self.crystal_structure.currentText(),
            "ebsd_reference_frame": self.ebsd_reference.currentText(),
            "active_modes": active_modes,
            "hcp_lattice_a": float(self.lattice_a.value()),
            "hcp_lattice_c": float(self.lattice_c.value()),
            "slip_angle_tolerance_deg": float(self.slip_tolerance.value()),
            "twin_angle_tolerance_deg": float(self.twin_tolerance.value()),
        }
        total_vectors = sum(CRYSTAL_MODE_TRACE_COUNTS.get(mode, 0) for mode in active_modes)
        self.crystal_summary.setText(
            f"{self._crystal_config['crystal_structure']} | {len(active_modes)} modes | "
            f"{total_vectors} trace vectors/event\nPriority: "
            + (", ".join(active_modes) if active_modes else "none")
        )
        if self.ebsd_reference.currentText().startswith("Oxford"):
            self.crystal_summary.setText(self.crystal_summary.text() + "\nOxford scoring is not implemented yet.")

    def _load_alignment_summary(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            transform_path = self._project_files.get("transform_json", "")
            ang_path = self._project_files.get("ang_file", "")
            if not transform_path or not Path(transform_path).expanduser().exists():
                raise ValueError("Load transformation JSON in Project Files first.")
            if not ang_path or not Path(ang_path).expanduser().exists():
                raise ValueError("Load ANG file in Project Files first.")
            self._set_status_neutral("Loading transformation JSON...")
            QApplication.processEvents()
            self._transform = load_alignment_transform_parameters(transform_path)
            self._set_status_neutral("Loading ANG file. Large ANG files can take a little while...")
            QApplication.processEvents()
            self._ang_core = load_ang_core_data(ang_path)
            self._set_status_neutral("Transforming ANG coordinates into DIC coordinates...")
            QApplication.processEvents()
            mapping = ang_grid_to_ebsd_pixel_mapping(self._ang_core["metadata"], self._transform["metadata"], self._ang_core["data"])
            self._dic_coords = transform_ang_coordinates_to_dic(
                self._ang_core["data"],
                self._transform["forward_params"],
                mapping["x_scale"],
                mapping["y_scale"],
            )
            dic_shape = self._transform["metadata"].get("dic_image_shape") or []
            metrics = dic_coordinate_metrics(
                self._dic_coords,
                int(dic_shape[1]) if len(dic_shape) >= 2 else None,
                int(dic_shape[0]) if len(dic_shape) >= 2 else None,
            )
        except Exception as exc:
            self._set_status_error(f"Alignment summary failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        rows = [
            ("Transform family", self._transform["metadata"].get("transform_family"), "Type of fitted EBSD-to-DIC coordinate transform stored in the JSON file."),
            ("Polynomial order", self._transform["metadata"].get("polynomial_order"), "Order of the polynomial mapping used for alignment."),
            ("Matched control points", self._transform["metadata"].get("matched_points"), "Number of paired EBSD/DIC control points used to fit the transform."),
            ("Forward RMSE px", self._transform["metadata"].get("rmse_forward_pixels"), "Average forward-fit error when EBSD control points are mapped into DIC pixels."),
            ("Inverse RMSE px", self._transform["metadata"].get("rmse_inverse_pixels"), "Average inverse-fit error when DIC control points are mapped back into EBSD pixels."),
            ("DIC image shape", self._transform["metadata"].get("dic_image_shape"), "Target DIC image size used as the output coordinate frame."),
            ("EBSD image shape", self._transform["metadata"].get("ebsd_image_shape"), "Original EBSD image size used when the transform was fitted."),
            ("ANG rows", f"{len(self._ang_core['data']):,}", "Number of orientation rows read from the ANG data section."),
            ("XSTEP", self._ang_core["metadata"].get("XSTEP"), "ANG scan step size in the x direction from the ANG header."),
            ("YSTEP", self._ang_core["metadata"].get("YSTEP"), "ANG scan step size in the y direction from the ANG header."),
            ("ANG grid", f"{mapping['ang_ncols']} x {mapping['ang_nrows']}", "Estimated ANG grid columns by rows after converting scan coordinates to a grid."),
            ("ANG to EBSD scale", f"x={mapping['x_scale']:.6g}, y={mapping['y_scale']:.6g}", "Scale used to convert ANG grid coordinates into EBSD image pixel coordinates."),
            ("DIC x range", f"{metrics['x_dic_min']:.1f} to {metrics['x_dic_max']:.1f}", "Range of transformed ANG x coordinates in the DIC coordinate system."),
            ("DIC y range", f"{metrics['y_dic_min']:.1f} to {metrics['y_dic_max']:.1f}", "Range of transformed ANG y coordinates in the DIC coordinate system."),
            ("DIC in/out bounds", f"{metrics['in_bounds_count']:,} / {metrics['out_of_bounds_count']:,}", "How many transformed ANG points fall inside versus outside the DIC image bounds."),
        ]
        self._fill_table(self.summary_table, rows)
        self._set_status_neutral("Loaded ANG and transformation summary.")

    def _generate_trace_vectors(self) -> None:
        self._apply_crystal_setup()
        if self._dic_coords is None:
            self._load_alignment_summary()
        if self._dic_coords is None:
            return
        try:
            self._trace_preview = build_trace_preview(self._dic_coords, self.trace_preview_rows.value(), self._crystal_config)
        except Exception as exc:
            self._set_label_error(self.trace_summary, f"Trace generation failed: {exc}")
            return
        active_modes = list(self._crystal_config["active_modes"])
        total_vectors = int(self._trace_preview["selected_trace_count"].iloc[0]) if len(self._trace_preview) else 0
        priority_text = ", ".join(active_modes) if active_modes else "none"
        self._set_label_neutral(
            self.trace_summary,
            f"Generated {len(self._trace_preview):,} preview rows. "
            f"Scoring will use {total_vectors} selected vectors/event from {len(active_modes)} selected mode(s).\n"
            f"Priority: {priority_text}"
        )
        self._fill_dataframe_table(
            self.trace_table,
            self._trace_preview.head(100),
            ["x_dic", "y_dic", "phi1", "PHI", "phi2", "selected_trace_count", "trace_labels", "trace_vectors"],
        )

    def _analyze_shapes(self) -> None:
        try:
            if self._event_info is None:
                raise ValueError("Run Boundary Cuts first. Alignment uses the cut events from the previous step.")
            self._morphology = compute_event_morphology_features(
                self._event_info["events"],
                self._event_info["event_pixels"],
                self.shape_min_pixels.value(),
                self.shape_linearity.value(),
                self.shape_aspect.value(),
                self.shape_density.value(),
            )
        except Exception as exc:
            self._set_label_error(self.shape_summary, f"Event shape analysis failed: {exc}")
            return
        counts = self._morphology["event_type"].value_counts().to_dict()
        metrics = self._event_info["metrics"]
        self._set_label_neutral(
            self.shape_summary,
            f"Events: {metrics['event_count']:,} | pixels: {metrics['pixel_count']:,} | "
            f"linear {counts.get('linear', 0):,}, blob {counts.get('blob_like', 0):,}, "
            f"irregular {counts.get('irregular', 0):,}, fragmented {counts.get('fragmented', 0):,}, "
            f"small noise {counts.get('small_noise', 0):,}"
        )
        self._build_shape_overlays()

    def _send_shape_analysis_to_review(self) -> None:
        if self._event_info is None or self._morphology is None:
            self._set_label_error(self.shape_summary, "Analyze event shapes before sending events to Manual Review.")
            return
        selected_types = {
            event_type
            for event_type, checkbox in self.shape_send_checks.items()
            if checkbox.isChecked()
        }
        if not selected_types:
            self._set_label_error(self.shape_summary, "Choose at least one event type to send to Manual Review.")
            return
        filtered = self._morphology[self._morphology["event_type"].isin(selected_types)].copy()
        if filtered.empty:
            self._set_label_error(
                self.shape_summary,
                "The selected event types are not present in the current shape-analysis result.",
            )
            return
        counts = filtered["event_type"].value_counts().to_dict()
        count_text = ", ".join(
            f"{event_type.replace('_', ' ')} {counts.get(event_type, 0):,}"
            for event_type in SHAPE_OVERLAY_TYPES
            if counts.get(event_type, 0)
        )
        self.shape_analysis_ready.emit(self._event_info, filtered)
        self._set_label_neutral(self.shape_summary, self.shape_summary.text() + f"\nSent to Manual Review: {count_text}.")

    def classification_context(self) -> dict:
        self._apply_crystal_setup()
        if self._dic_coords is None:
            self._load_alignment_summary()
        if self._dic_coords is None:
            raise ValueError("Load ANG and transformation summary in Auto Detection > Alignment first.")
        return {
            "dic_coords": self._dic_coords,
            "crystal_config": dict(self._crystal_config),
            "angle_tolerance_deg": float(self.slip_tolerance.value()),
        }

    def _build_shape_overlays(self) -> None:
        if self._event_info is None or self._morphology is None:
            self._clear_shape_overlays()
            return
        bln_path = self._project_files.get("bln_image", "").strip()
        if not bln_path:
            self._set_label_error(self.shape_overlay_status, "Load a BLN image in Project Files to build shape overlays.")
            return
        bln_path = str(Path(bln_path).expanduser())
        if not Path(bln_path).exists():
            self._set_label_error(self.shape_overlay_status, f"BLN image not found: {bln_path}")
            return

        event_pixels = self._event_info["event_pixels"]
        required = {"event_id", "pixel_x", "pixel_y"}
        if event_pixels.empty or not required.issubset(event_pixels.columns):
            self._set_label_error(self.shape_overlay_status, "Event pixels are empty or missing pixel_x/pixel_y columns.")
            return
        try:
            base_rgb, crop_info = self._shape_overlay_background(bln_path, event_pixels)
            morphology = self._morphology[["event_id", "event_type"]].copy()
            morphology["event_id"] = morphology["event_id"].astype(str)
            overlay_pixels = event_pixels.copy()
            overlay_pixels["event_id"] = overlay_pixels["event_id"].astype(str)
            overlay_pixels = overlay_pixels.merge(morphology, on="event_id", how="inner")
            counts: dict[str, int] = {}
            for event_type, view in self.shape_overlay_views.items():
                type_pixels = overlay_pixels[overlay_pixels["event_type"] == event_type]
                counts[event_type] = int(type_pixels["event_id"].nunique()) if not type_pixels.empty else 0
                view.set_array(self._draw_shape_overlay(base_rgb, type_pixels, crop_info))
                self.shape_overlay_tabs.setTabText(
                    self.shape_overlay_tabs.indexOf(view),
                    f"{event_type.replace('_', ' ').title()} ({counts[event_type]:,})",
                )
        except Exception as exc:
            self._set_label_error(self.shape_overlay_status, f"Shape overlay rendering failed: {exc}")
            return

        self._set_label_neutral(
            self.shape_overlay_status,
            "Shape overlays use the BLN image background; red pixels show the selected event-shape class."
        )

    def _clear_shape_overlays(self) -> None:
        if not hasattr(self, "shape_overlay_views"):
            return
        for event_type, view in self.shape_overlay_views.items():
            view.set_array(None)
            self.shape_overlay_tabs.setTabText(
                self.shape_overlay_tabs.indexOf(view),
                event_type.replace("_", " ").title(),
            )
        if hasattr(self, "shape_overlay_status"):
            self._set_label_neutral(self.shape_overlay_status, "Shape overlays will appear after event-shape analysis.")

    def _shape_overlay_background(self, bln_path: str, event_pixels) -> tuple[np.ndarray, dict]:
        xs = event_pixels["pixel_x"].to_numpy(dtype=np.int64)
        ys = event_pixels["pixel_y"].to_numpy(dtype=np.int64)
        with Image.open(bln_path) as image:
            width, height = image.size
            pad = 40
            x0 = max(0, int(xs.min()) - pad)
            y0 = max(0, int(ys.min()) - pad)
            x1 = min(width, int(xs.max()) + pad + 1)
            y1 = min(height, int(ys.max()) + pad + 1)
            crop = image.crop((x0, y0, x1, y1))
            crop_arr = np.asarray(crop).copy()
        rgb = self._normalize_bln_crop(crop_arr)
        crop_h, crop_w = rgb.shape[:2]
        scale = min(1.0, MAX_SHAPE_OVERLAY_DIM / max(crop_w, crop_h))
        if scale < 1.0:
            display_w = max(1, int(round(crop_w * scale)))
            display_h = max(1, int(round(crop_h * scale)))
            rgb = np.asarray(Image.fromarray(rgb).resize((display_w, display_h), Image.Resampling.BILINEAR)).copy()
        else:
            display_w, display_h = crop_w, crop_h
        return rgb, {
            "x0": x0,
            "y0": y0,
            "scale_x": display_w / max(crop_w, 1),
            "scale_y": display_h / max(crop_h, 1),
        }

    @staticmethod
    def _normalize_bln_crop(crop_arr: np.ndarray) -> np.ndarray:
        arr = np.squeeze(crop_arr)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            rgb = arr[..., :3].astype(np.float32, copy=False)
            lo, hi = np.percentile(rgb, [0.5, 99.5])
            if hi <= lo:
                scaled = np.zeros(rgb.shape, dtype=np.uint8)
            else:
                scaled = (np.clip((rgb - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
            return scaled
        values = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(values, [0.5, 99.5])
        if hi <= lo:
            gray = np.zeros(values.shape, dtype=np.uint8)
        else:
            gray = (np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)

    @staticmethod
    def _draw_shape_overlay(base_rgb: np.ndarray, type_pixels, crop_info: dict) -> np.ndarray:
        overlay = base_rgb.copy()
        if type_pixels.empty:
            return overlay
        cols = np.rint((type_pixels["pixel_x"].to_numpy(dtype=np.float64) - crop_info["x0"]) * crop_info["scale_x"]).astype(np.int64)
        rows = np.rint((type_pixels["pixel_y"].to_numpy(dtype=np.float64) - crop_info["y0"]) * crop_info["scale_y"]).astype(np.int64)
        valid = (rows >= 0) & (rows < overlay.shape[0]) & (cols >= 0) & (cols < overlay.shape[1])
        rows = rows[valid]
        cols = cols[valid]
        if len(rows) == 0:
            return overlay
        radius = 1 if max(crop_info["scale_x"], crop_info["scale_y"]) < 0.75 else 2
        for dr in range(-radius, radius + 1):
            rr = rows + dr
            row_ok = (rr >= 0) & (rr < overlay.shape[0])
            for dc in range(-radius, radius + 1):
                cc = cols + dc
                ok = row_ok & (cc >= 0) & (cc < overlay.shape[1])
                overlay[rr[ok], cc[ok]] = [255, 0, 0]
        return overlay

    def _refresh_file_summary_text(self) -> None:
        return

    @staticmethod
    def _fill_table(table: QTableWidget, rows: list[tuple[object, object, object]]) -> None:
        table.setRowCount(len(rows))
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Item", "Value", "Description"])
        for row, (key, value, description) in enumerate(rows):
            table.setItem(row, 0, QTableWidgetItem(str(key)))
            table.setItem(row, 1, QTableWidgetItem(str(value)))
            table.setItem(row, 2, QTableWidgetItem(str(description)))
        table.resizeColumnsToContents()

    @staticmethod
    def _fill_dataframe_table(table: QTableWidget, frame, columns: list[str]) -> None:
        if len(frame) == 0:
            table.setRowCount(0)
            table.setColumnCount(len(columns))
            table.setHorizontalHeaderLabels(columns)
            return
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Missing table columns: {missing}")
        table.setRowCount(len(frame))
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        for row_index, row in enumerate(frame[columns].itertuples(index=False)):
            for col_index, value in enumerate(row):
                table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()

    @staticmethod
    def _spin(minimum: int, maximum: int, step: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    @staticmethod
    def _event_metrics(events, event_pixels) -> dict[str, int]:
        event_count = int(events["event_id"].nunique()) if "event_id" in events else int(len(events))
        pixel_count = int(len(event_pixels))
        return {"event_count": event_count, "pixel_count": pixel_count}

    def _current_parameter_defaults(self) -> dict:
        return {
            "crystal_structure": self.crystal_structure.currentText(),
            "ebsd_reference": self.ebsd_reference.currentText(),
            "lattice_a": float(self.lattice_a.value()),
            "lattice_c": float(self.lattice_c.value()),
            "slip_tolerance": float(self.slip_tolerance.value()),
            "twin_tolerance": float(self.twin_tolerance.value()),
            "active_modes": [
                self.mode_list.item(i).text()
                for i in range(self.mode_list.count())
                if self.mode_list.item(i).checkState() == Qt.CheckState.Checked
            ],
            "mode_order": [self.mode_list.item(i).text() for i in range(self.mode_list.count())],
            "trace_preview_rows": int(self.trace_preview_rows.value()),
            "shape_min_pixels": int(self.shape_min_pixels.value()),
            "shape_linearity": float(self.shape_linearity.value()),
            "shape_aspect": float(self.shape_aspect.value()),
            "shape_density": float(self.shape_density.value()),
        }

    def _save_parameter_defaults(self) -> None:
        try:
            with ALIGNMENT_PARAMETER_DEFAULTS_PATH.open("w", encoding="utf-8") as handle:
                json.dump(self._current_parameter_defaults(), handle, indent=2)
        except Exception as exc:
            self._set_status_error(f"Could not save alignment defaults: {exc}")
            return
        self._set_status_neutral("Saved alignment parameter defaults.")

    def _load_parameter_defaults(self) -> None:
        if not ALIGNMENT_PARAMETER_DEFAULTS_PATH.exists():
            return
        try:
            with ALIGNMENT_PARAMETER_DEFAULTS_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return
        structure = data.get("crystal_structure")
        if structure:
            index = self.crystal_structure.findText(str(structure))
            if index >= 0:
                self.crystal_structure.setCurrentIndex(index)
        reference = data.get("ebsd_reference")
        if reference:
            index = self.ebsd_reference.findText(str(reference))
            if index >= 0:
                self.ebsd_reference.setCurrentIndex(index)
        for widget_name, key in [
            ("lattice_a", "lattice_a"),
            ("lattice_c", "lattice_c"),
            ("slip_tolerance", "slip_tolerance"),
            ("twin_tolerance", "twin_tolerance"),
            ("trace_preview_rows", "trace_preview_rows"),
            ("shape_min_pixels", "shape_min_pixels"),
            ("shape_linearity", "shape_linearity"),
            ("shape_aspect", "shape_aspect"),
            ("shape_density", "shape_density"),
        ]:
            if key in data:
                getattr(self, widget_name).setValue(data[key])
        mode_order = [mode for mode in data.get("mode_order", []) if isinstance(mode, str)]
        active_modes = set(mode for mode in data.get("active_modes", []) if isinstance(mode, str))
        available = [self.mode_list.item(i).text() for i in range(self.mode_list.count())]
        ordered = [mode for mode in mode_order if mode in available] + [mode for mode in available if mode not in mode_order]
        if ordered:
            self.mode_list.blockSignals(True)
            self.mode_list.clear()
            for mode in ordered:
                item = QListWidgetItem(mode)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if mode in active_modes else Qt.CheckState.Unchecked)
                self.mode_list.addItem(item)
            self.mode_list.blockSignals(False)
        self._apply_crystal_setup()

    def _set_status_neutral(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet("color: #374151;")

    def _set_status_error(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet("color: #b91c1c; font-weight: 700;")

    @staticmethod
    def _set_label_neutral(label: QLabel, text: str) -> None:
        label.setText(text)
        label.setStyleSheet("color: #374151;")

    @staticmethod
    def _set_label_error(label: QLabel, text: str) -> None:
        label.setText(text)
        label.setStyleSheet("color: #b91c1c; font-weight: 700;")

    @staticmethod
    def _double_spin(minimum: float, maximum: float, step: float, value: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin
