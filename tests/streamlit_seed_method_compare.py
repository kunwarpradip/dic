from __future__ import annotations

import base64
import csv
from dataclasses import dataclass, replace
from datetime import datetime
from io import BytesIO
import importlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from skimage import measure, morphology
from skimage.color import rgb2lab
from skimage.draw import line as draw_line
from skimage.exposure import equalize_adapthist
from skimage.filters import meijering, threshold_otsu
from skimage.transform import probabilistic_hough_line


def find_project_root() -> Path:
    candidates = [Path(__file__).resolve(), Path.cwd(), *Path.cwd().parents]
    for start in candidates:
        path = start if start.is_dir() else start.parent
        for candidate in [path, *path.parents]:
            if (
                (candidate / "dic_qt").exists()
                and (candidate / "DIC_check_slip_twin_traces_hcp.py").exists()
            ):
                return candidate
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "dic_qt").exists():
            return path
    return Path("/Users/pkunwar/Desktop/DIC_Local_App")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import DIC_check_slip_twin_traces_hcp as hcp_trace_module  # noqa: E402
from DIC_check_slip_traces_bcc import calcSlipTracesBCC110, calcSlipTracesBCC112  # noqa: E402
from dic_qt.core.algorithm import detect_line_from_seed  # noqa: E402
from dic_qt.core.models import DicLine, Point  # noqa: E402

hcp_trace_module = importlib.reload(hcp_trace_module)
calcSlipTracesHCPBasal = hcp_trace_module.calcSlipTracesHCPBasal
calcSlipTracesHCPPrism = hcp_trace_module.calcSlipTracesHCPPrism
calcSlipTracesHCPPyra_I_A = hcp_trace_module.calcSlipTracesHCPPyra_I_A
calcSlipTracesHCPPyra_I_CA = hcp_trace_module.calcSlipTracesHCPPyra_I_CA
calcSlipTracesHCPPyra_II_CA = hcp_trace_module.calcSlipTracesHCPPyra_II_CA
calcTwinTracesHCP = hcp_trace_module.calcTwinTracesHCP


DEFAULT_IMAGE = ROOT / "z_share_DIC_data_for_hv_mvu_pk" / "Ti_Cryo" / "Fused_BlN_step3.tif"
DEFAULT_GRX_IMAGE = (
    ROOT / "z_share_DIC_data_for_hv_mvu_pk" / "Ti_Cryo" / "Fused_GrX_step3_3pxGB_8bit.tif"
)
DEFAULT_EBSD_IMAGE = ROOT / "z_share_EBSD_data_for_hv_mvu_pk" / "example_cropped_distorted_dic_oim_map.tif"
DEFAULT_SLIP_MASK_CSV = ROOT / "z_share_DIC_data_for_hv_mvu_pk" / "Ti_Cryo" / "slip_mask_pixels.csv"
DEFAULT_DETECTED_EVENTS_CSV = ROOT / "tests" / "outputs" / "hough_events" / "test_this_events.csv"
DEFAULT_DETECTED_EVENT_PIXELS_CSV = (
    ROOT / "tests" / "outputs" / "hough_events" / "test_this_event_pixels.csv"
)
DEFAULT_FULL_DETECTED_EVENTS_CSV = (
    ROOT / "tests" / "outputs" / "hough_events" / "full_ti_image_detection_events.csv"
)
DEFAULT_FULL_DETECTED_EVENT_PIXELS_CSV = (
    ROOT / "tests" / "outputs" / "hough_events" / "full_ti_image_detection_event_pixels.csv"
)
DEFAULT_EBSD_BOUNDARY_MAP = ROOT / "z_share_EBSD_DIC_alignment_data" / "check_ebsd_mask_manual_edits.png"
DEFAULT_ALIGNMENT_TRANSFORM_JSON = (
    ROOT / "z_share_DIC_data_for_hv_mvu_pk" / "Ti_Cryo" / "app_data" / "alignment_transformation.json"
)
DEFAULT_ANG_FILE = ROOT / "z_share_EBSD_DIC_alignment_data" / "undistorted_dic_cleaned_approximatecrop_ang.ang"
DEFAULT_EBSD_CUT_EVENTS_CSV = ROOT / "tests" / "outputs" / "hough_events" / "full_ti_image_detection_boundary_cut_events.csv"
DEFAULT_EBSD_CUT_EVENT_PIXELS_CSV = (
    ROOT / "tests" / "outputs" / "hough_events" / "full_ti_image_detection_boundary_cut_event_pixels.csv"
)
HCP_MILLER_INDEX_DIR = ROOT / "hcp_slip_twin_miller_indices"
hcp_trace_module.HCP_MILLER_INDEX_DIR = HCP_MILLER_INDEX_DIR
CROP_SELECTOR_COMPONENT = components.declare_component(
    "dic_crop_selector",
    path=str(ROOT / "tests" / "streamlit_components" / "crop_selector"),
)

EBSD_REFERENCE_FRAMES = [
    "EDAX (default reference frame)",
    "Oxford (default reference frame)",
]

CRYSTAL_MODE_OPTIONS = {
    "HCP (Hexagonal Close Packed)": [
        "Basal - 1x plane trace",
        "Prismatic - 3x plane traces",
        "Pyramidal I - 6x plane traces",
        "Pyramidal II - 6x plane traces",
        "{10-12} Twin (Tension) - 6x plane traces",
        "{11-21} Twin (Tension) - 6x plane traces",
        "{11-22} Twin (Compression) - 6x plane traces",
        "{10-11} Twin (Compression) - 6x plane traces",
        "{11-24} Twin (Compression) - 6x plane traces",
    ],
    "BCC (Body Centered Cubic)": [
        "BCC {110} Slip - 6x plane traces",
        "BCC {112} Slip - 12x plane traces",
    ],
    "FCC (Face Centered Cubic)": [
        "{111}<110> Slip",
        "{111}<112> Twin",
    ],
}

DEFAULT_CRYSTAL_MODES = {
    "HCP (Hexagonal Close Packed)": [
        "Basal - 1x plane trace",
        "Prismatic - 3x plane traces",
    ],
    "BCC (Body Centered Cubic)": [
        "BCC {110} Slip - 6x plane traces",
        "BCC {112} Slip - 12x plane traces",
    ],
    "FCC (Face Centered Cubic)": [
        "{111}<110> Slip",
    ],
}

CRYSTAL_MODE_TRACE_COUNTS = {
    "Basal - 1x plane trace": 1,
    "Prismatic - 3x plane traces": 3,
    "Pyramidal I - 6x plane traces": 6,
    "Pyramidal II - 6x plane traces": 6,
    "{10-12} Twin (Tension) - 6x plane traces": 6,
    "{11-21} Twin (Tension) - 6x plane traces": 6,
    "{11-22} Twin (Compression) - 6x plane traces": 6,
    "{10-11} Twin (Compression) - 6x plane traces": 6,
    "{11-24} Twin (Compression) - 6x plane traces": 6,
    "BCC {110} Slip - 6x plane traces": 6,
    "BCC {112} Slip - 12x plane traces": 12,
    "{111}<110> Slip": 0,
    "{111}<112> Twin": 0,
}

HCP_FILE_BACKED_MODES = {
    "Pyramidal I - 6x plane traces": "hcp_pyra_i_a_miller.csv",
    "Pyramidal II - 6x plane traces": "hcp_pyra_ii_ca_miller.csv",
    "{10-12} Twin (Tension) - 6x plane traces": "hcp_twin_t1_miller.csv",
    "{11-21} Twin (Tension) - 6x plane traces": "hcp_twin_t2_miller.csv",
    "{11-22} Twin (Compression) - 6x plane traces": "hcp_twin_c1_miller.csv",
    "{10-11} Twin (Compression) - 6x plane traces": "hcp_twin_c2_miller.csv",
    "{11-24} Twin (Compression) - 6x plane traces": "hcp_twin_c3_miller.csv",
}


@dataclass(frozen=True)
class PipelineParams:
    image_path: str
    grx_overlay_path: str
    display_min: float
    display_max: float
    use_full_image: bool
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    clahe_clip_limit: float
    ridge_sigma_max: int
    ridge_percentile: float
    threshold_multiplier: float
    min_object_size: int
    closing_radius: int
    use_skeletonize: bool
    hough_threshold: int
    hough_line_length: int
    hough_line_gap: int
    hough_seed_spacing: int
    hough_max_seeds: int
    hough_use_all_seeds: bool
    region_min_pixels: int
    region_seed_spacing: int
    region_max_seeds: int
    region_use_all_seeds: bool
    intensity_difference_tolerance: int
    best_fit_line_tolerance: float
    min_intensity: int
    min_points_threshold: int
    duplicate_overlap: float
    merge_distance_tolerance: float
    merge_angle_tolerance: float
    connect_merged_event_gaps: bool


def main() -> None:
    st.set_page_config(page_title="DIC Seed Method Compare", layout="wide")
    st.title("DIC Seed Method Compare")
    st.caption(
        "Compare Hough-line and regionprops ridge seeds on a crop, then feed each "
        "seed into the Java-style DIC event-growing algorithm."
    )

    base_params = sidebar_params()
    (
        tab_crystal_setup,
        tab_region,
        tab_processing,
        tab_results,
        tab_detection_analysis,
        tab_category_dashboard,
        tab_boundary_cut,
        tab_alignment,
        tab_event_analysis,
    ) = st.tabs(
        [
            "Crystal Setup",
            "Processing Region",
            "Further Processing",
            "Detection Results",
            "Detection Analysis",
            "Category Dashboard",
            "Boundary Cut",
            "Alignment",
            "Cut Event Analysis",
        ]
    )

    with tab_crystal_setup:
        crystal_setup_tab()

    with tab_region:
        params = processing_region_controls(base_params)

    with tab_processing:
        params = further_processing_controls(params)

    with tab_results:
        stored_params = st.session_state.get("last_pipeline_params")
        stored_result = st.session_state.get("last_pipeline_result")
        tab_hough_result, tab_region_result = st.tabs(["Hough Line Detection", "RegionProp Seed Detection"])
        hough_detect_clicked = False
        hough_trace_clicked = False
        auto_run_hough = False
        region_run_clicked = False

        with tab_hough_result:
            st.markdown("**Step 1: Detect Hough Lines And Seeds**")
            params = hough_method_controls(params, stored_result)
            auto_run_hough = st.checkbox(
                "Auto-run Step 1 when controls change",
                value=False,
                key="auto_run_hough_step1",
                help="Runs only Hough line/seed detection automatically. Event tracing still runs only when Step 2 is pressed.",
            )
            hough_detect_clicked = st.button(
                "Run Step 1",
                type="primary",
                key="run_hough_seed_detection",
            )
            hough_step1_result_area = st.container()

            st.divider()
            st.markdown("**Step 2: Trace Events From Hough Seeds**")
            params = hough_event_tracing_controls(params)
            hough_trace_clicked = st.button(
                "Run Step 2",
                key="trace_hough_seed_events",
            )
            hough_step2_result_area = st.container()

            if auto_run_hough or hough_detect_clicked:
                label = "full-image" if params.use_full_image else "crop-based"
                with st.spinner(f"Running {label} Hough line/seed detection..."):
                    stored_result = run_hough_seed_pipeline(params)
                    st.session_state["last_pipeline_params"] = params
                    st.session_state["last_pipeline_result"] = stored_result
                    stored_params = params

            if hough_trace_clicked:
                label = "full-image" if params.use_full_image else "crop-based"
                if stored_result is None or "hough" not in stored_result:
                    with st.spinner(f"Running {label} Hough line/seed detection first..."):
                        stored_result = run_hough_seed_pipeline(params)
                with st.spinner("Tracing events from selected Hough seeds..."):
                    stored_result = trace_hough_events_for_result(stored_result, params)
                    st.session_state["last_pipeline_params"] = params
                    st.session_state["last_pipeline_result"] = stored_result
                    stored_params = params

            with hough_step1_result_area:
                if stored_result is None or "hough" not in stored_result:
                    st.info("Select a processing region first, then run Step 1.")
                else:
                    if not hough_source_matches_result(stored_result, params):
                        st.warning("Step 1 result is from previous Hough/preprocessing settings. Run Step 1 again.")
                    show_detection_mask_metrics(stored_result)
                    show_hough_step1_result(stored_result)

            with hough_step2_result_area:
                if stored_result is None or "hough" not in stored_result:
                    st.info("Run Step 1 before tracing events.")
                elif "hough_events" not in stored_result:
                    st.info("Step 1 is complete. Run Step 2 to trace events from the selected Hough seeds.")
                else:
                    if not params_match_for_result(stored_params, params):
                        st.warning("Step 2 result is from previous tracing settings. Run Step 2 again.")
                    show_hough_step2_result(stored_result)

        with tab_region_result:
            params = regionprops_method_controls(params, stored_result)
            region_run_clicked = st.button(
                "Run / refresh detection",
                type="primary",
                key="run_regionprops_detection",
            )

        if region_run_clicked:
            label = "full-image" if params.use_full_image else "crop-based"
            with st.spinner(f"Running {label} RegionProp detection..."):
                stored_result = run_pipeline(params)
                st.session_state["last_pipeline_params"] = params
                st.session_state["last_pipeline_result"] = stored_result
                stored_params = params

        if stored_result is None:
            with tab_region_result:
                st.info("Select a processing region first, then press Run / refresh detection.")
        else:
            if not params_match_for_result(stored_params, params):
                with tab_region_result:
                    st.warning("Displayed result is from previous settings. Press Run / refresh detection to update.")

            with tab_region_result:
                if "regionprops" in stored_result:
                    show_detection_mask_metrics(stored_result)
                    show_regionprops_method_view(stored_result)
                else:
                    st.info("RegionProp results are not loaded. Press Run / refresh detection in this tab.")

    with tab_detection_analysis:
        detection_analysis_tab(params)

    with tab_category_dashboard:
        category_dashboard_tab(params)

    with tab_boundary_cut:
        boundary_cut_tab()

    with tab_alignment:
        alignment_tab()

    with tab_event_analysis:
        event_analysis_tab()


def crystal_setup_tab() -> None:
    st.subheader("Crystal Setup")
    st.caption(
        "Choose the crystal structure, EBSD reference-frame convention, and the slip/twin modes "
        "expected to be active. These selections are stored for later trace-generation and scoring steps."
    )

    setup_cols = st.columns([1, 1])
    with setup_cols[0]:
        crystal_structure = st.radio(
            "Crystal structure",
            list(CRYSTAL_MODE_OPTIONS.keys()),
            index=0,
            key="crystal_setup_structure",
            help="Controls which slip/twin mode options are shown.",
        )
    with setup_cols[1]:
        ebsd_reference_frame = st.selectbox(
            "EBSD data format / reference frame",
            EBSD_REFERENCE_FRAMES,
            index=0,
            key="crystal_setup_ebsd_reference_frame",
            help="The vendor default reference frame used when converting Euler angles to plane traces.",
        )

    active_modes = ordered_crystal_mode_selector(crystal_structure)
    if ebsd_reference_frame.startswith("Oxford"):
        st.warning(
            "Oxford is available in the UI, but the current parent trace-vector functions only implement EDAX. "
            "Scoring will need an Oxford reference-frame implementation before this option can be used."
        )
    if crystal_structure.startswith("HCP"):
        missing_hcp_mode_files = [
            f"{mode}: {filename}"
            for mode, filename in HCP_FILE_BACKED_MODES.items()
            if mode in active_modes and not (HCP_MILLER_INDEX_DIR / filename).exists()
        ]
        if missing_hcp_mode_files:
            st.warning(
                f"Some selected HCP modes require Miller-index CSV files in `{HCP_MILLER_INDEX_DIR}`:\n\n"
                + "\n".join(f"- {entry}" for entry in missing_hcp_mode_files)
            )

    lattice_a = None
    lattice_c = None
    slip_angle_tolerance = None
    twin_angle_tolerance = None
    if crystal_structure.startswith("HCP"):
        st.markdown("**HCP Lattice Parameters**")
        lattice_cols = st.columns(2)
        with lattice_cols[0]:
            lattice_a = st.number_input(
                "<a> lattice parameter",
                min_value=0.000001,
                value=float(st.session_state.get("crystal_setup_lattice_a", 2.95)),
                step=0.01,
                format="%.6f",
                key="crystal_setup_lattice_a",
                help="Scalar <a> lattice parameter used by pyramidal and twin mode trace calculations.",
            )
        with lattice_cols[1]:
            lattice_c = st.number_input(
                "<c> lattice parameter",
                min_value=0.000001,
                value=float(st.session_state.get("crystal_setup_lattice_c", 4.686)),
                step=0.01,
                format="%.6f",
                key="crystal_setup_lattice_c",
                help="Scalar <c> lattice parameter used by pyramidal and twin mode trace calculations.",
            )
        st.markdown("**HCP Angle Tolerances**")
        tolerance_cols = st.columns(2)
        with tolerance_cols[0]:
            slip_angle_tolerance = st.number_input(
                "Slip angle tolerance (degrees)",
                min_value=0.0,
                max_value=90.0,
                value=float(st.session_state.get("crystal_setup_hcp_slip_angle_tolerance", 5.0)),
                step=0.5,
                format="%.3f",
                key="crystal_setup_hcp_slip_angle_tolerance",
                help="Angle tolerance for HCP slip modes: basal, prismatic, pyramidal I, and pyramidal II.",
            )
        with tolerance_cols[1]:
            twin_angle_tolerance = st.number_input(
                "Twin angle tolerance (degrees)",
                min_value=0.0,
                max_value=90.0,
                value=float(st.session_state.get("crystal_setup_hcp_twin_angle_tolerance", 5.0)),
                step=0.5,
                format="%.3f",
                key="crystal_setup_hcp_twin_angle_tolerance",
                help="Angle tolerance for HCP twin modes.",
            )
    else:
        st.markdown("**Angle Tolerance**")
        slip_angle_tolerance = st.number_input(
            "Slip angle tolerance (degrees)",
            min_value=0.0,
            max_value=90.0,
            value=float(st.session_state.get("crystal_setup_slip_angle_tolerance", 5.0)),
            step=0.5,
            format="%.3f",
            key="crystal_setup_slip_angle_tolerance",
            help="Angle tolerance for selected slip modes.",
        )

    config = {
        "crystal_structure": crystal_structure,
        "ebsd_reference_frame": ebsd_reference_frame,
        "active_modes": active_modes,
        "hcp_lattice_a": lattice_a,
        "hcp_lattice_c": lattice_c,
        "slip_angle_tolerance_deg": slip_angle_tolerance,
        "twin_angle_tolerance_deg": twin_angle_tolerance,
    }
    st.session_state["crystal_setup_config"] = config

    st.markdown("**Current Selection**")
    active_trace_count = sum(CRYSTAL_MODE_TRACE_COUNTS.get(mode, 0) for mode in active_modes)
    summary_rows = [
        {"setting": "Crystal structure", "value": crystal_structure},
        {"setting": "EBSD reference frame", "value": ebsd_reference_frame},
        {"setting": "Active modes", "value": ", ".join(active_modes) if active_modes else "None selected"},
        {"setting": "Actual trace vectors used per event", "value": str(active_trace_count)},
    ]
    if crystal_structure.startswith("HCP"):
        summary_rows.extend(
            [
                {"setting": "<a> lattice parameter", "value": f"{float(lattice_a):.6f}"},
                {"setting": "<c> lattice parameter", "value": f"{float(lattice_c):.6f}"},
                {"setting": "<c>/<a>", "value": f"{float(lattice_c) / float(lattice_a):.6f}"},
                {"setting": "Slip angle tolerance", "value": f"{float(slip_angle_tolerance):.3f} deg"},
                {"setting": "Twin angle tolerance", "value": f"{float(twin_angle_tolerance):.3f} deg"},
            ]
        )
    else:
        summary_rows.append(
            {"setting": "Slip angle tolerance", "value": f"{float(slip_angle_tolerance):.3f} deg"}
        )
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
    if active_modes:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "priority": index + 1,
                        "mode": mode,
                        "trace_vectors": CRYSTAL_MODE_TRACE_COUNTS.get(mode, 0),
                    }
                    for index, mode in enumerate(active_modes)
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    st.info("These selections are used by Alignment scoring. Each selected mode expands into its trace-vector variants.")


def ordered_crystal_mode_selector(crystal_structure: str) -> list[str]:
    mode_options = CRYSTAL_MODE_OPTIONS[crystal_structure]
    default_modes = [
        mode
        for mode in DEFAULT_CRYSTAL_MODES.get(crystal_structure, [])
        if mode in mode_options
    ]
    order_key = f"crystal_setup_mode_order_{crystal_structure}"
    if order_key not in st.session_state:
        st.session_state[order_key] = default_modes.copy()

    current_order = [
        mode
        for mode in st.session_state.get(order_key, [])
        if mode in mode_options
    ]

    st.markdown("**Expected Active Slip/Twin Modes**")
    st.caption(
        "Select the modes to consider, then order them by priority. During scoring, the first "
        "mode in this list should be checked first once we wire this into the backend."
    )

    selected_from_checks = []
    checkbox_cols = st.columns(2)
    for index, mode in enumerate(mode_options):
        checkbox_key = f"crystal_setup_mode_enabled_{crystal_structure}_{index}"
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = mode in current_order
        with checkbox_cols[index % 2]:
            enabled = st.checkbox(
                mode,
                key=checkbox_key,
                help="Enable this slip/twin mode for downstream trace matching.",
            )
        if enabled:
            selected_from_checks.append(mode)

    ordered_selected = [mode for mode in current_order if mode in selected_from_checks]
    for mode in selected_from_checks:
        if mode not in ordered_selected:
            ordered_selected.append(mode)
    st.session_state[order_key] = ordered_selected

    st.markdown("**Mode Priority**")
    if not ordered_selected:
        st.warning("No active modes selected.")
        return []

    for index, mode in enumerate(ordered_selected):
        row = st.columns([0.12, 0.62, 0.13, 0.13])
        row[0].markdown(f"**{index + 1}**")
        row[1].write(mode)
        if row[2].button(
            "Up",
            key=f"crystal_setup_mode_up_{crystal_structure}_{index}",
            disabled=index == 0,
            help="Move this mode earlier in the priority order.",
        ):
            reordered = ordered_selected.copy()
            reordered[index - 1], reordered[index] = reordered[index], reordered[index - 1]
            st.session_state[order_key] = reordered
            st.rerun()
        if row[3].button(
            "Down",
            key=f"crystal_setup_mode_down_{crystal_structure}_{index}",
            disabled=index == len(ordered_selected) - 1,
            help="Move this mode later in the priority order.",
        ):
            reordered = ordered_selected.copy()
            reordered[index + 1], reordered[index] = reordered[index], reordered[index + 1]
            st.session_state[order_key] = reordered
            st.rerun()

    return ordered_selected


def ebsd_cuts_tab(params: PipelineParams) -> None:
    st.subheader("EBSD Cuts")
    st.caption(
        "Uses the exact DIC processing crop coordinates on the EBSD image, then overlays "
        "high EBSD color-gradient pixels in black."
    )

    ebsd_path = st.text_input("EBSD image path", str(DEFAULT_EBSD_IMAGE), key="ebsd_cuts_image_path")
    ebsd_path = str(Path(ebsd_path).expanduser())
    if not Path(ebsd_path).exists():
        st.error(f"EBSD image not found: {ebsd_path}")
        return

    ebsd_w, ebsd_h = image_size(ebsd_path)
    dic_w, dic_h = image_size(params.image_path)
    if (ebsd_w, ebsd_h) != (dic_w, dic_h):
        st.warning(
            f"EBSD image size is {ebsd_w} x {ebsd_h}, while the DIC image size is "
            f"{dic_w} x {dic_h}. The same crop coordinates will be used, so overlays "
            "are only exact if the images are already aligned."
        )

    crop_rect = sanitize_crop_rect(
        {
            "x": params.crop_x,
            "y": params.crop_y,
            "width": params.crop_width,
            "height": params.crop_height,
        },
        ebsd_w,
        ebsd_h,
    )
    if (
        crop_rect["x"] != params.crop_x
        or crop_rect["y"] != params.crop_y
        or crop_rect["width"] != params.crop_width
        or crop_rect["height"] != params.crop_height
    ):
        st.warning(
            "The current DIC crop had to be clipped to fit inside the EBSD image. "
            "Check image alignment before interpreting the overlay."
        )

    cols = st.columns(5)
    cols[0].metric("Crop X", crop_rect["x"])
    cols[1].metric("Crop Y", crop_rect["y"])
    cols[2].metric("Width", crop_rect["width"])
    cols[3].metric("Height", crop_rect["height"])
    cols[4].metric("Mode", "Full" if params.use_full_image else "Crop")

    if params.use_full_image:
        st.warning("Full-image EBSD gradient processing can be slow. Tune on a crop first when possible.")

    st.markdown("**Gradient Map Controls**")
    col1, col2, col3 = st.columns(3)
    with col1:
        color_space = st.selectbox(
            "Gradient color space",
            ["rgb", "lab"],
            index=0,
            key="ebsd_cuts_color_space",
            help=(
                "RGB is faster. Lab can better emphasize visible color differences "
                "between grains because it is closer to perceptual color distance."
            ),
        )
    with col2:
        gradient_sigma = st.slider(
            "Gradient smoothing sigma",
            0.0,
            5.0,
            1.0,
            0.1,
            key="ebsd_cuts_gradient_sigma",
            help="Gaussian smoothing before Sobel gradients. Higher values suppress tiny speckles but blur narrow boundaries.",
        )
    with col3:
        normalized_threshold = st.slider(
            "Normalized gradient threshold",
            0.0,
            1.0,
            0.50,
            0.01,
            key="ebsd_cuts_gradient_threshold",
            help=(
                "Threshold applied after percentile normalization of the gradient map. "
                "Higher values keep only stronger color-gradient points."
            ),
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        normalize_low_pct = st.slider(
            "Gradient normalize low percentile",
            0.0,
            10.0,
            1.0,
            0.1,
            key="ebsd_cuts_norm_low_pct",
            help="Gradient values at or below this percentile map to 0 for thresholding/display.",
        )
    with col2:
        normalize_high_pct = st.slider(
            "Gradient normalize high percentile",
            90.0,
            100.0,
            99.5,
            0.1,
            key="ebsd_cuts_norm_high_pct",
            help="Gradient values at or above this percentile map to 1 for thresholding/display.",
        )
    with col3:
        dilation_radius = st.slider(
            "Overlay dilation radius",
            0,
            10,
            3,
            1,
            key="ebsd_cuts_dilation_radius",
            help="Dilates thresholded gradient points only for display. Default 3 makes thin points visible.",
        )

    if normalize_high_pct <= normalize_low_pct:
        st.error("Gradient normalize high percentile must be greater than the low percentile.")
        return

    signature = (
        ebsd_path,
        crop_rect["x"],
        crop_rect["y"],
        crop_rect["width"],
        crop_rect["height"],
        color_space,
        float(gradient_sigma),
        float(normalize_low_pct),
        float(normalize_high_pct),
        float(normalized_threshold),
        int(dilation_radius),
    )
    stored_signature = st.session_state.get("last_ebsd_cuts_signature")
    stored_result = st.session_state.get("last_ebsd_cuts_result")

    auto_process = st.checkbox(
        "Auto-process EBSD cuts when controls change",
        value=not params.use_full_image,
        key="ebsd_cuts_auto_process",
        help="When enabled, slider changes immediately recompute the EBSD gradient overlay.",
    )
    run_clicked = st.button("Run / refresh EBSD cuts", type="primary", key="run_ebsd_cuts")

    if auto_process or run_clicked:
        with st.spinner("Computing EBSD color-gradient cut overlay..."):
            stored_result = run_ebsd_cuts_gradient_overlay(
                ebsd_path,
                crop_rect["x"],
                crop_rect["y"],
                crop_rect["width"],
                crop_rect["height"],
                color_space,
                float(gradient_sigma),
                float(normalize_low_pct),
                float(normalize_high_pct),
                float(normalized_threshold),
                int(dilation_radius),
            )
            st.session_state["last_ebsd_cuts_signature"] = signature
            st.session_state["last_ebsd_cuts_result"] = stored_result
            stored_signature = signature

    if stored_result is None:
        st.info("Press Run / refresh EBSD cuts to compute the gradient overlay for the current DIC crop.")
        return

    if stored_signature != signature:
        st.warning("Displayed EBSD cuts result is from previous controls. Press Run / refresh EBSD cuts to update.")

    show_ebsd_cuts_result(stored_result)


def ebsd_processing_tab() -> None:
    st.subheader("EBSD Processing")
    st.caption(
        "Tune grain-boundary extraction from EBSD color changes. The boundary mask is built "
        "from high color-gradient pixels, then optionally cleaned, closed, and thinned."
    )

    ebsd_path = st.text_input("EBSD image path", str(DEFAULT_EBSD_IMAGE), key="ebsd_image_path")
    ebsd_path = str(Path(ebsd_path).expanduser())
    if not Path(ebsd_path).exists():
        st.error(f"EBSD image not found: {ebsd_path}")
        return

    img_w, img_h = image_size(ebsd_path)
    st.caption(f"EBSD image size: {img_w} x {img_h} pixels")

    mode = st.radio(
        "EBSD region mode",
        ["Crop", "Full image"],
        horizontal=True,
        key="ebsd_region_mode",
    )
    use_full_image = mode == "Full image"

    if use_full_image:
        st.warning(
            "Full-image EBSD gradient processing can be slow and memory-heavy. "
            "Use a crop while tuning, then run the full image once the parameters look right."
        )
        crop_rect = {"x": 0, "y": 0, "width": img_w, "height": img_h}
    else:
        default_crop_size = min(1000, img_w, img_h)
        default_crop = sanitize_crop_rect(
            {
                "x": max(0, (img_w - default_crop_size) // 2),
                "y": max(0, (img_h - default_crop_size) // 2),
                "width": default_crop_size,
                "height": default_crop_size,
            },
            img_w,
            img_h,
        )
        state_key = f"ebsd_selected_crop:{ebsd_path}"
        current_crop = sanitize_crop_rect(st.session_state.get(state_key, default_crop), img_w, img_h)
        preview = load_region_selector_preview(
            ebsd_path,
            0,
            0,
            img_w,
            img_h,
            max_dim=1400,
        )
        st.caption(
            f"Crop selector uses the full EBSD image downsampled to "
            f"{preview['preview_width']} x {preview['preview_height']}."
        )
        selected = CROP_SELECTOR_COMPONENT(
            image_data_url=preview["image_data_url"],
            preview_width=preview["preview_width"],
            preview_height=preview["preview_height"],
            original_width=preview["original_width"],
            original_height=preview["original_height"],
            view_x=preview["view_x"],
            view_y=preview["view_y"],
            view_width=preview["view_width"],
            view_height=preview["view_height"],
            initial_crop=current_crop,
            default_crop_width=default_crop["width"],
            default_crop_height=default_crop["height"],
            min_crop_size=20,
            key=f"ebsd_crop_selector_{Path(ebsd_path).stem}_full_overview",
            default=current_crop,
        )
        crop_rect = sanitize_crop_rect(selected or current_crop, img_w, img_h)
        st.session_state[state_key] = crop_rect

    cols = st.columns(4)
    cols[0].metric("Left X", crop_rect["x"])
    cols[1].metric("Top Y", crop_rect["y"])
    cols[2].metric("Width", crop_rect["width"])
    cols[3].metric("Height", crop_rect["height"])

    crop_preview = load_crop_by_rect(
        ebsd_path,
        crop_rect["x"],
        crop_rect["y"],
        crop_rect["width"],
        crop_rect["height"],
    )
    st.image(
        to_display(crop_preview["display_rgb"]),
        caption="Selected EBSD region",
        width='stretch',
    )

    st.divider()
    st.markdown("**Gradient Boundary Controls**")
    col1, col2, col3 = st.columns(3)
    with col1:
        color_space = st.selectbox(
            "Gradient color space",
            ["rgb", "lab"],
            index=0,
            key="ebsd_gradient_color_space",
            help=(
                "RGB is faster and uses less memory. Lab measures perceptual color distance, "
                "which can help when grains differ by hue rather than raw channel intensity."
            ),
        )
    with col2:
        gradient_sigma = st.slider(
            "Gradient smoothing sigma",
            0.0,
            5.0,
            1.0,
            0.1,
            key="ebsd_gradient_sigma",
            help="Gaussian smoothing before the gradient. Higher values suppress speckle but can blur narrow boundaries.",
        )
    with col3:
        gradient_percentile = st.slider(
            "Gradient percentile",
            80.0,
            99.9,
            96.5,
            0.1,
            key="ebsd_gradient_percentile",
            help="Keeps pixels whose color-gradient strength is above this percentile. Higher values keep fewer, stronger boundaries.",
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_boundary_object_size = st.slider(
            "Min boundary object pixels",
            0,
            5000,
            40,
            10,
            key="ebsd_min_boundary_object_size",
            help="Removes connected boundary fragments smaller than this many pixels. Increase to clean tiny specks.",
        )
    with col2:
        boundary_closing_radius = st.slider(
            "Boundary closing radius",
            0,
            10,
            1,
            1,
            key="ebsd_boundary_closing_radius",
            help="Morphological closing radius. Higher values bridge small gaps and can join nearby boundary pieces.",
        )
    with col3:
        use_boundary_skeleton = st.checkbox(
            "Thin boundary",
            value=True,
            key="ebsd_use_boundary_skeleton",
            help="Thins the cleaned boundary mask toward one-pixel-wide lines. Disable if it is slow or if you want thicker boundary regions.",
        )
    with col4:
        boundary_display_radius = st.slider(
            "Display dilation radius",
            0,
            6,
            1,
            1,
            key="ebsd_boundary_display_radius",
            help="Dilates only the displayed/saved overlay boundary so thin lines are easier to see. It does not change the core boundary count.",
        )

    overlay_color_name = st.selectbox(
        "Overlay color",
        ["white", "red", "yellow", "cyan", "blue"],
        index=0,
        key="ebsd_overlay_color",
        help="Color used to draw detected boundaries over the EBSD image.",
    )

    ebsd_signature = (
        ebsd_path,
        crop_rect["x"],
        crop_rect["y"],
        crop_rect["width"],
        crop_rect["height"],
        color_space,
        float(gradient_sigma),
        float(gradient_percentile),
        int(min_boundary_object_size),
        int(boundary_closing_radius),
        bool(use_boundary_skeleton),
        int(boundary_display_radius),
        overlay_color_name,
    )
    stored_signature = st.session_state.get("last_ebsd_signature")
    stored_result = st.session_state.get("last_ebsd_result")

    auto_process = st.checkbox(
        "Auto-process EBSD when controls change",
        value=not use_full_image,
        key=f"ebsd_auto_process_{'full' if use_full_image else 'crop'}",
        help="When enabled, every slider change reruns the EBSD gradient boundary extraction.",
    )
    run_clicked = st.button("Run / refresh EBSD processing", type="primary", key="run_ebsd_processing")

    if auto_process or run_clicked:
        with st.spinner("Running EBSD gradient boundary extraction..."):
            stored_result = run_ebsd_gradient_boundary(
                ebsd_path,
                crop_rect["x"],
                crop_rect["y"],
                crop_rect["width"],
                crop_rect["height"],
                color_space,
                float(gradient_sigma),
                float(gradient_percentile),
                int(min_boundary_object_size),
                int(boundary_closing_radius),
                bool(use_boundary_skeleton),
                int(boundary_display_radius),
                overlay_color_name,
            )
            st.session_state["last_ebsd_signature"] = ebsd_signature
            st.session_state["last_ebsd_result"] = stored_result
            stored_signature = ebsd_signature

    if stored_result is None:
        st.info("Choose an EBSD region and press Run / refresh EBSD processing.")
        return

    if stored_signature != ebsd_signature:
        st.warning("Displayed EBSD result is from previous settings. Press Run / refresh EBSD processing to update.")

    show_ebsd_gradient_result(stored_result)

    if st.button("Save EBSD boundary outputs", key="save_ebsd_outputs"):
        saved_paths = save_ebsd_boundary_outputs(stored_result, ebsd_path)
        st.success(
            "Saved EBSD outputs:\n\n"
            + "\n".join(f"- {path}" for path in saved_paths)
        )


def detection_analysis_tab(params: PipelineParams) -> None:
    st.subheader("Detection Analysis")
    st.caption(
        "Compare saved automatic event pixels against slip ground-truth pixels inside the "
        "saved detection crop. Quality colors: true positive green, false positive red, "
        "missed slip pixels blue."
    )

    bln_path = params.image_path
    grx_path = params.grx_overlay_path
    st.caption(f"Using BLN image from sidebar: `{bln_path}`")
    st.caption(f"Using GRX image from sidebar: `{grx_path or 'not set'}`")

    col_a, col_b = st.columns(2)
    with col_a:
        slip_mask_file = st.file_uploader(
            "Ground truth slip_mask_pixels.csv",
            type=["csv"],
            key="analysis_slip_mask_file",
        )
        if st.button("Use default slip mask", key="analysis_use_default_slip_mask"):
            st.session_state["analysis_slip_mask_path"] = str(DEFAULT_SLIP_MASK_CSV)
    with col_b:
        detected_pixels_file = st.file_uploader(
            "Detected event pixels CSV",
            type=["csv"],
            key="analysis_detected_pixels_file",
        )
        if st.button("Use default detected pixels", key="analysis_use_default_detected_pixels"):
            st.session_state["analysis_detected_pixels_path"] = str(DEFAULT_DETECTED_EVENT_PIXELS_CSV)

    if slip_mask_file is not None:
        st.session_state["analysis_slip_mask_path"] = str(
            save_uploaded_streamlit_file(slip_mask_file, "detection_analysis")
        )
    if detected_pixels_file is not None:
        st.session_state["analysis_detected_pixels_path"] = str(
            save_uploaded_streamlit_file(detected_pixels_file, "detection_analysis")
        )
    slip_mask_path = st.session_state.get("analysis_slip_mask_path", str(DEFAULT_SLIP_MASK_CSV))
    detected_pixels_path = st.session_state.get(
        "analysis_detected_pixels_path",
        str(DEFAULT_DETECTED_EVENT_PIXELS_CSV),
    )
    st.caption(f"Ground truth slip pixels: `{slip_mask_path}`")
    st.caption(f"Detected event pixels: `{detected_pixels_path}`")

    st.markdown("**BLN Display Range**")
    st.caption(
        "Fiji-style brightness/contrast for the BLN analysis overlay. Values below the "
        "minimum map to black, above the maximum map to white, and values between are "
        "linearly mapped to 0-255."
    )
    bln_display_min, bln_display_max = st.slider(
        "BLN brightness/contrast range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.001,
        key="analysis_bln_display_range_0_1",
        help="Display window for BLN values in the Detection Analysis tab. Default 0-1.",
    )

    run_clicked = st.button(
        "Run detection analysis",
        type="primary",
        key="run_detection_analysis",
    )

    paths = {
        "BLN image": bln_path,
        "GRX image": grx_path,
        "slip mask CSV": slip_mask_path,
        "detected event pixels CSV": detected_pixels_path,
    }
    missing = [
        label
        for label, path in paths.items()
        if not str(path).strip() or not Path(path).expanduser().exists()
    ]
    if missing:
        for label in missing:
            st.error(f"{label} not found: {paths[label]}")
        return

    analysis_signature = (
        str(Path(bln_path).expanduser()),
        str(Path(grx_path).expanduser()),
        str(Path(slip_mask_path).expanduser()),
        str(Path(detected_pixels_path).expanduser()),
        float(bln_display_min),
        float(bln_display_max),
    )

    if run_clicked:
        try:
            with st.spinner("Running detection analysis..."):
                result = run_detection_analysis(*analysis_signature)
        except Exception as exc:
            st.error(f"Detection analysis failed: {exc}")
            return
        st.session_state["last_detection_analysis_result"] = result
        st.session_state["last_detection_analysis_signature"] = analysis_signature

    result = st.session_state.get("last_detection_analysis_result")
    result_signature = st.session_state.get("last_detection_analysis_signature")
    if result is None:
        st.info("Press Run detection analysis to calculate metrics and overlays.")
        return

    if result_signature != analysis_signature:
        st.warning("Displayed detection analysis is from previous inputs. Press Run detection analysis to update.")

    crop = result["crop"]
    st.markdown("**Detection Crop**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("x", crop["x_min"])
    c2.metric("y", crop["y_min"])
    c3.metric("width", crop["width"])
    c4.metric("height", crop["height"])
    st.caption(
        f"Crop origin from detected events CSV: ({crop['x_min']}, {crop['y_min']}); "
        f"extent from detected event pixels: x={crop['x_min']}..{crop['x_max']}, "
        f"y={crop['y_min']}..{crop['y_max']}."
    )

    st.markdown("**Pixel Counts**")
    count_cols = st.columns(5)
    stats = result["stats"]
    count_cols[0].metric("Detected pixels", f"{stats['automatic_detected_pixels']:,}")
    count_cols[1].metric("Slip GT pixels", f"{stats['ground_truth_pixels']:,}")
    count_cols[2].metric("Overlap / TP", f"{stats['overlap_true_positive']:,}")
    count_cols[3].metric("Missed / FN", f"{stats['missed_ground_truth_false_negative']:,}")
    count_cols[4].metric("Extra / FP", f"{stats['automatic_false_positive']:,}")

    rate_cols = st.columns(4)
    rate_cols[0].metric("Precision", f"{stats['precision_tp_over_detected']:.4f}")
    rate_cols[1].metric("Recall", f"{stats['recall_tp_over_ground_truth']:.4f}")
    rate_cols[2].metric("F1", f"{stats['f1_score']:.4f}")
    rate_cols[3].metric("IoU", f"{stats['iou_jaccard']:.4f}")

    with st.expander("Detailed statistics", expanded=False):
        st.dataframe(pd.DataFrame([stats]), width="stretch")

    st.markdown("**Overlays**")
    st.caption(
        "Quality overlays: green = detected slip overlap, red = automatic false positive, "
        "blue = missed slip ground truth."
    )
    col1, col2 = st.columns(2)
    with col1:
        zoomable_image(result["grx_quality_overlay"], "GRX quality overlay", key="analysis_grx_quality")
    with col2:
        zoomable_image(result["bln_quality_overlay"], "BLN quality overlay", key="analysis_bln_quality")


def category_dashboard_tab(params: PipelineParams) -> None:
    st.subheader("Category Dashboard")
    tab_pixels, tab_angles = st.tabs(["Pixel Categories", "Misorientation Histograms"])
    with tab_pixels:
        category_pixel_dashboard_panel(params)
    with tab_angles:
        misorientation_histogram_panel()


def category_pixel_dashboard_panel(params: PipelineParams) -> None:
    st.caption(
        "Upload ground-truth slip pixels and the categorized event-pixel CSVs saved from Alignment. "
        "The dashboard compares each category directly against the same ground-truth pixel set."
    )

    upload_cols = st.columns(2)
    with upload_cols[0]:
        ground_truth_file = st.file_uploader(
            "Ground truth slip_mask_pixels.csv",
            type=["csv"],
            key="category_dashboard_ground_truth_file",
        )
        matched_file = st.file_uploader(
            "Matched event pixels CSV",
            type=["csv"],
            key="category_dashboard_matched_pixels_file",
        )
        uncertain_file = st.file_uploader(
            "Uncertain event pixels CSV",
            type=["csv"],
            key="category_dashboard_uncertain_pixels_file",
        )
    with upload_cols[1]:
        blob_file = st.file_uploader(
            "Blob event pixels CSV",
            type=["csv"],
            key="category_dashboard_blob_pixels_file",
        )
        noise_file = st.file_uploader(
            "Noise event pixels CSV",
            type=["csv"],
            key="category_dashboard_noise_pixels_file",
        )

    st.caption(f"Overlay BLN image from sidebar: `{params.image_path}`")
    bln_display_min, bln_display_max = st.slider(
        "BLN overlay brightness/contrast range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.001,
        key="category_dashboard_bln_display_range",
        help="Display window for the BLN crop used under the red/green/blue category overlay.",
    )
    overlay_max_dim = st.slider(
        "Overlay max dimension",
        min_value=800,
        max_value=4000,
        value=2200,
        step=200,
        key="category_dashboard_overlay_max_dim",
        help="Maximum width or height for the category quality overlay preview.",
    )
    run_clicked = st.button(
        "Build category dashboard",
        type="primary",
        key="category_dashboard_run",
        disabled=ground_truth_file is None or matched_file is None,
    )
    if ground_truth_file is None or matched_file is None:
        st.info("Upload at least ground truth and matched event pixels to build the dashboard.")
        return

    if run_clicked:
        try:
            with st.spinner("Comparing category pixels with ground truth..."):
                result = run_category_detection_dashboard(
                    ground_truth_file,
                    {
                        "matched": matched_file,
                        "blob": blob_file,
                        "uncertain": uncertain_file,
                        "noise": noise_file,
                    },
                    bln_path=params.image_path,
                    bln_display_min=float(bln_display_min),
                    bln_display_max=float(bln_display_max),
                    overlay_max_dim=int(overlay_max_dim),
                )
        except Exception as exc:
            st.error(f"Category dashboard failed: {exc}")
            return
        st.session_state["category_dashboard_result"] = result

    result = st.session_state.get("category_dashboard_result")
    if result is None:
        return

    overview = result["overview"]
    overview_cols = st.columns(5)
    overview_cols[0].metric("Ground truth pixels", f"{overview['ground_truth_pixels']:,}")
    overview_cols[1].metric("Matched TP", f"{overview['matched_true_positive']:,}")
    overview_cols[2].metric("Matched FP", f"{overview['matched_false_positive']:,}")
    overview_cols[3].metric("Matched FN", f"{overview['matched_false_negative']:,}")
    overview_cols[4].metric("GT found in non-matched", f"{overview['ground_truth_in_nonmatched_categories']:,}")

    st.markdown("**Category Metrics**")
    metrics = result["metrics"].copy()
    for col in ["precision_tp_over_detected", "recall_tp_over_ground_truth", "f1_score", "iou_jaccard"]:
        if col in metrics.columns:
            metrics[col] = metrics[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    st.dataframe(metrics, width="stretch", hide_index=True)

    st.markdown("**Category Plots**")
    plot_cols = st.columns(2)
    with plot_cols[0]:
        st.altair_chart(category_count_chart(result["metrics"]), use_container_width=True)
    with plot_cols[1]:
        st.altair_chart(category_rate_chart(result["metrics"]), use_container_width=True)

    st.markdown("**Missed Ground Truth Breakdown**")
    missed_cols = st.columns(4)
    missed_cols[0].metric("Missed by matched", f"{overview['matched_false_negative']:,}")
    missed_cols[1].metric("Found in blob", f"{overview['matched_missed_found_in_blob']:,}")
    missed_cols[2].metric("Found in uncertain", f"{overview['matched_missed_found_in_uncertain']:,}")
    missed_cols[3].metric("Found in noise", f"{overview['matched_missed_found_in_noise']:,}")
    st.caption(
        f"Ground-truth pixels not found in any uploaded category: "
        f"{overview['ground_truth_missed_by_all_uploaded_categories']:,}"
    )

    st.markdown("**Matched Detection Overlay**")
    st.caption(
        "BLN background uses the sidebar image and the display range above. "
        "Red = original ground truth pixels, green = matched overlap / true positive, blue = matched false positive."
    )
    overlay = result.get("overlay")
    overlay_crop = result.get("overlay_crop", {})
    if overlay is not None:
        st.caption(
            f"Overlay crop: x={overlay_crop.get('x_min')}, y={overlay_crop.get('y_min')}, "
            f"width={overlay_crop.get('width')}, height={overlay_crop.get('height')}"
        )
        zoomable_image(
            overlay,
            image_title_with_dimensions("Matched event quality overlay", overlay),
            key="category_dashboard_quality_overlay",
        )


def misorientation_histogram_panel() -> None:
    st.caption(
        "Upload a trace-comparison CSV saved from Alignment. This analyzes signed and absolute "
        "misorientation between each event direction and its assigned slip trace."
    )
    trace_file = st.file_uploader(
        "Trace comparison CSV",
        type=["csv"],
        key="misorientation_trace_comparison_csv",
    )
    threshold_values = st.multiselect(
        "Thresholds to summarize",
        options=[5.0, 7.0],
        default=[5.0, 7.0],
        key="misorientation_thresholds",
        help="Events with best_angle_error_deg less than or equal to each threshold are included.",
    )
    bin_width = st.slider(
        "Histogram bin width (degrees)",
        min_value=0.25,
        max_value=2.0,
        value=1.0,
        step=0.25,
        key="misorientation_bin_width",
    )
    if trace_file is None:
        st.info("Upload a trace-comparison CSV to plot misorientation distributions.")
        return

    try:
        trace_df = read_uploaded_or_path_csv(trace_file)
        analysis = run_misorientation_analysis(trace_df, threshold_values, float(bin_width))
    except Exception as exc:
        st.error(f"Could not analyze misorientation CSV: {exc}")
        return

    st.markdown("**Summary**")
    summary = analysis["summary"].copy()
    for col in ["signed_mean_deg", "signed_median_deg", "signed_std_deg", "abs_mean_deg", "abs_median_deg", "abs_std_deg"]:
        if col in summary.columns:
            summary[col] = summary[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    st.dataframe(summary, width="stretch", hide_index=True)

    tab_specs = [(float(threshold), f"{threshold:g} deg") for threshold in analysis["thresholds"]]
    tab_specs.append(("all", "All"))
    tab_specs.append(("noise", "Noise"))
    tabs = st.tabs([label for _, label in tab_specs])
    for tab, (threshold, _label) in zip(tabs, tab_specs, strict=False):
        with tab:
            bundle = analysis["histograms"][threshold]
            st.markdown("**Signed misorientation**")
            st.caption("Positive means the event vector is counterclockwise from the selected trace; negative means clockwise.")
            st.altair_chart(colored_histogram_chart(bundle["signed"], "Signed misorientation (deg)"), use_container_width=True)
            st.markdown("**Absolute misorientation**")
            st.altair_chart(colored_histogram_chart(bundle["absolute"], "Absolute misorientation (deg)"), use_container_width=True)


def boundary_cut_tab() -> None:
    st.subheader("Boundary Cut")
    st.caption(
        "Cut saved detected events wherever their pixels cross black EBSD grain-boundary pixels. "
        "Overlays use red for original event pixels, blue for EBSD boundaries, and green for pixels removed by the cut."
    )

    tab_manual, tab_auto, tab_check = st.tabs(["Manual Cut", "Auto Cut", "Check Cuts"])

    with tab_manual:
        manual_boundary_cut_tab()

    with tab_auto:
        auto_boundary_cut_tab()

    with tab_check:
        check_boundary_cut_events_tab()


def alignment_tab() -> None:
    st.subheader("Alignment")
    st.caption("Load EBSD/DIC transformation metadata and EBSD ANG data for the next alignment-processing steps.")

    col1, col2 = st.columns(2)
    with col1:
        transform_file = st.file_uploader(
            "Transformation JSON",
            type=["json"],
            key="alignment_transform_json_file",
            help="JSON saved from the DIC/EBSD alignment tool. It should contain control points and transform coefficients.",
        )
    with col2:
        ang_file = st.file_uploader(
            "ANG file",
            type=["ang", "txt"],
            key="alignment_ang_file",
            help="EBSD orientation data file in .ang format.",
        )

    if st.button("Use default alignment files", key="alignment_use_default_files"):
        st.session_state["alignment_transform_json_path"] = str(DEFAULT_ALIGNMENT_TRANSFORM_JSON)
        st.session_state["alignment_ang_path"] = str(DEFAULT_ANG_FILE)
        st.session_state["alignment_events_path"] = str(DEFAULT_EBSD_CUT_EVENTS_CSV)
        st.session_state["alignment_event_pixels_path"] = str(DEFAULT_EBSD_CUT_EVENT_PIXELS_CSV)

    if transform_file is not None:
        st.session_state["alignment_transform_json_path"] = str(
            save_uploaded_streamlit_file(transform_file, "alignment")
        )
    if ang_file is not None:
        st.session_state["alignment_ang_path"] = str(
            save_uploaded_streamlit_file(ang_file, "alignment")
        )

    transform_path = st.session_state.get("alignment_transform_json_path", "")
    ang_path = st.session_state.get("alignment_ang_path", "")

    loaded = []
    if transform_path:
        loaded.append(f"transformation JSON: `{Path(transform_path).name}`")
    if ang_path:
        loaded.append(f"ANG: `{Path(ang_path).name}`")
    if loaded:
        st.caption("Loaded " + " | ".join(loaded))
    else:
        st.info("Load a transformation JSON and an ANG file to begin.")
        return

    transform = None
    dic_coords = None
    alignment_coordinate_source = "Original transformed ANG points"
    if transform_path and Path(transform_path).exists():
        st.markdown("**Transformation Parameters**")
        try:
            transform = load_alignment_transform_parameters(transform_path)
        except Exception as exc:
            st.error(f"Could not extract transformation parameters: {exc}")
            return

        cols = st.columns(5)
        cols[0].metric("Family", transform["metadata"].get("transform_family", "unknown"))
        cols[1].metric("Order", transform["metadata"].get("polynomial_order", "unknown"))
        cols[2].metric("Matched points", transform["metadata"].get("matched_points", "missing"))
        cols[3].metric("Forward RMSE", transform["metadata"].get("rmse_forward_pixels", "missing"))
        cols[4].metric("Inverse RMSE", transform["metadata"].get("rmse_inverse_pixels", "missing"))

        table1, table2 = st.columns(2)
        with table1:
            st.caption("Forward transform: EBSD pixel coordinates -> DIC pixel coordinates")
            st.dataframe(transform["forward_coefficients"], width="stretch", hide_index=True)
        with table2:
            st.caption("Inverse map: DIC pixel coordinates -> EBSD pixel coordinates")
            st.dataframe(transform["inverse_coefficients"], width="stretch", hide_index=True)
    elif transform_path:
        st.error(f"Transformation JSON not found: {transform_path}")

    if ang_path and Path(ang_path).exists():
        st.markdown("**ANG Core Data**")
        try:
            with st.spinner("Extracting XSTEP, YSTEP, and first five ANG data columns..."):
                ang_core = load_ang_core_data(ang_path)
        except Exception as exc:
            st.error(f"Could not extract ANG core data: {exc}")
            return

        meta = ang_core["metadata"]
        cols = st.columns(5)
        cols[0].metric("XSTEP", meta.get("XSTEP", "missing"))
        cols[1].metric("YSTEP", meta.get("YSTEP", "missing"))
        cols[2].metric("Rows", f"{len(ang_core['data']):,}")
        cols[3].metric("NCOLS", meta.get("NCOLS_ODD", "missing"))
        cols[4].metric("NROWS", meta.get("NROWS", "missing"))

        coord = ang_core["coordinate_normalization"]
        coord_cols = st.columns(6)
        coord_cols[0].metric("x origin", f"{coord['x_origin']:.6g}")
        coord_cols[1].metric("y origin", f"{coord['y_origin']:.6g}")
        coord_cols[2].metric("x offset applied", f"{coord['x_offset_applied']:.6g}")
        coord_cols[3].metric("y offset applied", f"{coord['y_offset_applied']:.6g}")
        coord_cols[4].metric("x grid max", f"{coord['x_grid_max']:.0f}")
        coord_cols[5].metric("y grid max", f"{coord['y_grid_max']:.0f}")

        st.caption(
            "Extracted first five ANG data columns after the header. "
            "x_grid/y_grid are normalized EBSD grid coordinates: "
            "(x - x_origin) / XSTEP and (y - y_origin) / YSTEP."
        )
        st.dataframe(ang_core["data"].head(100), width="stretch", hide_index=True)

        st.markdown("**ANG Coordinate Sanity Check**")
        st.caption(
            "Orientation RGB image built from normalized Euler angles: "
            "red = phi1, green = PHI, blue = phi2."
        )
        try:
            euler_rgb = build_ang_euler_rgb_image(ang_core["data"])
        except Exception as exc:
            st.error(f"Could not build ANG Euler RGB sanity image: {exc}")
        else:
            zoomable_image(
                euler_rgb,
                image_title_with_dimensions("ANG Euler RGB from normalized grid coordinates", euler_rgb),
                key="alignment_ang_euler_rgb",
            )

        if transform is not None:
            st.markdown("**ANG Coordinates In DIC Space**")
            st.caption(
                "Forward polynomial transform applied to normalized EBSD grid coordinates: "
                "x_ebsd/y_ebsd -> x_dic/y_dic."
            )
            try:
                dic_coords = transform_ang_coordinates_to_dic(
                    ang_core["data"],
                    transform["forward_params"],
                )
            except Exception as exc:
                st.error(f"Could not transform ANG coordinates to DIC coordinates: {exc}")
            else:
                dic_meta = transform["metadata"]
                dic_shape = dic_meta.get("dic_image_shape") or []
                dic_height = int(dic_shape[0]) if len(dic_shape) >= 2 else None
                dic_width = int(dic_shape[1]) if len(dic_shape) >= 2 else None
                coord_metrics = dic_coordinate_metrics(dic_coords, dic_width, dic_height)
                metric_cols = st.columns(6)
                metric_cols[0].metric("x DIC min", f"{coord_metrics['x_dic_min']:.1f}")
                metric_cols[1].metric("x DIC max", f"{coord_metrics['x_dic_max']:.1f}")
                metric_cols[2].metric("y DIC min", f"{coord_metrics['y_dic_min']:.1f}")
                metric_cols[3].metric("y DIC max", f"{coord_metrics['y_dic_max']:.1f}")
                metric_cols[4].metric("DIC in-bounds", f"{coord_metrics['in_bounds_count']:,}")
                metric_cols[5].metric("DIC out-of-bounds", f"{coord_metrics['out_of_bounds_count']:,}")
                st.dataframe(dic_coords.head(100), width="stretch", hide_index=True)
                try:
                    dic_euler_rgb = build_dic_space_euler_rgb_image(
                        dic_coords,
                        dic_width=dic_width,
                        dic_height=dic_height,
                    )
                except Exception as exc:
                    st.error(f"Could not build DIC-space Euler RGB sanity image: {exc}")
                else:
                    zoomable_image(
                        dic_euler_rgb,
                        image_title_with_dimensions("ANG Euler RGB transformed into DIC coordinates", dic_euler_rgb),
                        key="alignment_dic_euler_rgb",
                    )
                st.markdown("**Forward-Filled ANG Euler RGB On DIC Canvas**")
                st.caption(
                    "This uses only the forward EBSD->DIC transform parameters. ANG points are transformed into "
                    "DIC space, then empty preview pixels inside the projected EBSD footprint are filled from the "
                    "nearest transformed ANG pixel."
                )
                full_forward_fill = st.checkbox(
                    "Use full DIC resolution",
                    value=False,
                    key="alignment_forward_fill_full_resolution",
                    help="Render the forward-filled ANG image at the full DIC canvas size. This can be slow and memory-heavy.",
                )
                if full_forward_fill:
                    forward_fill_max_dim = max(int(dic_width or 1), int(dic_height or 1))
                    st.warning(
                        f"Full-resolution output will be {int(dic_width)} x {int(dic_height)} pixels. "
                        "This may take a while to compute and can make the browser sluggish."
                    )
                else:
                    forward_fill_max_dim = st.slider(
                        "Forward-filled preview max dimension",
                        800,
                        3600,
                        2200,
                        100,
                        key="alignment_forward_fill_max_dim",
                        help="Maximum width or height for the forward-filled preview. Higher values show more detail but take longer.",
                    )
                try:
                    forward_filled_rgb = build_forward_filled_dic_euler_rgb_preview(
                        dic_coords,
                        dic_width=dic_width,
                        dic_height=dic_height,
                        max_dim=int(forward_fill_max_dim),
                        full_resolution=bool(full_forward_fill),
                    )
                except Exception as exc:
                    st.error(f"Could not build forward-filled DIC sanity image: {exc}")
                else:
                    zoomable_image(
                        forward_filled_rgb,
                        image_title_with_dimensions("Forward-filled ANG Euler RGB on DIC canvas", forward_filled_rgb),
                        key="alignment_forward_filled_euler_rgb",
                    )
                st.markdown("**ANG Coordinate Source**")
                alignment_coordinate_source = st.radio(
                    "Use ANG data as",
                    [
                        "Original transformed ANG points",
                        "Filled full-DIC canvas",
                    ],
                    index=0,
                    horizontal=True,
                    key="alignment_coordinate_source",
                    help=(
                        "Original transformed ANG points keeps one row per ANG measurement. "
                        "Filled full-DIC canvas uses the forward-filled DIC canvas representation for downstream event-pixel lookup."
                    ),
                )
                if alignment_coordinate_source == "Filled full-DIC canvas":
                    st.caption(
                        "Selected filled full-DIC canvas mode. For memory safety, this is used as an image/lookup source "
                        "rather than materializing every DIC pixel as a dataframe row."
                    )
        else:
            st.info("Load transformation JSON to transform ANG coordinates into DIC coordinates.")

        st.markdown("**Slip Trace Vectors**")
        st.caption(
            "Generate a quick basal/prismatic preview from each row's Bunge Euler angles. "
            "Event scoring below uses the active modes chosen in Crystal Setup and calls the parent trace-vector functions directly. "
            "When the transform is loaded, only coordinates are expressed on the DIC canvas; Euler angles and trace directions are not transformed."
        )
        st.caption(f"Current ANG coordinate source: `{alignment_coordinate_source}`")
        if st.button("Generate slip trace vectors", key="alignment_generate_slip_traces"):
            st.session_state["alignment_slip_trace_source"] = ang_path
            st.session_state["alignment_slip_trace_coordinate_source"] = alignment_coordinate_source

        if st.session_state.get("alignment_slip_trace_source") == ang_path:
            selected_trace_source = st.session_state.get(
                "alignment_slip_trace_coordinate_source",
                alignment_coordinate_source,
            )
            try:
                if transform is not None and dic_coords is not None:
                    if selected_trace_source == "Filled full-DIC canvas":
                        st.info(
                            "Filled full-DIC canvas mode is selected. The trace table below still shows the original "
                            "transformed ANG measurement rows; the filled canvas will be used for DIC/event-pixel lookup "
                            "without creating a huge full-pixel dataframe."
                        )
                    with st.spinner("Generating basal/prismatic preview trace vectors for transformed ANG rows..."):
                        trace_data = build_dic_space_trace_vector_data(
                            dic_coords,
                            transform["forward_params"],
                            schema_version=2,
                        )
                elif transform is not None:
                    st.error("Transform is loaded, but DIC-space ANG coordinates are not available yet.")
                    return
                else:
                    with st.spinner("Generating EBSD-space basal/prismatic preview trace vectors for ANG rows..."):
                        trace_data = load_ang_trace_vector_data(ang_path, schema_version=3)
            except Exception as exc:
                st.error(f"Could not generate slip trace vectors: {exc}")
                return

            trace_cols = st.columns(4)
            trace_cols[0].metric("Trace rows", f"{len(trace_data):,}")
            trace_cols[1].metric("Prism systems", "3")
            trace_cols[2].metric("Basal systems", "1")
            trace_cols[3].metric("Angle units", "radians")
            if len(trace_data):
                shape_cols = st.columns(2)
                shape_cols[0].metric("Prism shape", matrix_shape_label(trace_data["prism"].iloc[0]))
                shape_cols[1].metric("Basal shape", matrix_shape_label(trace_data["basal"].iloc[0]))
            st.caption(
                "`prism` stores a 3x3 trace-vector matrix. `basal` stores a 1x3 trace-vector matrix. "
                "The coordinates may be transformed into DIC space, but the trace vectors are generated directly from unchanged Euler angles."
            )
            st.dataframe(
                display_trace_vector_table(trace_data.head(100)),
                width="stretch",
                hide_index=True,
            )
    elif ang_path:
        st.error(f"ANG file not found: {ang_path}")

    st.markdown("**EBSD-Cut Events**")
    event_col1, event_col2 = st.columns(2)
    with event_col1:
        events_file = st.file_uploader(
            "Events CSV after EBSD cut",
            type=["csv"],
            key="alignment_events_file",
            help="Post-cut event-level CSV, usually saved from Boundary Cut.",
        )
    with event_col2:
        event_pixels_file = st.file_uploader(
            "Event pixels CSV after EBSD cut",
            type=["csv"],
            key="alignment_event_pixels_file",
            help="Post-cut event pixel CSV with event_id, pixel_x, pixel_y.",
        )

    if st.button("Use default EBSD-cut event files", key="alignment_use_default_event_files"):
        st.session_state["alignment_events_path"] = str(DEFAULT_EBSD_CUT_EVENTS_CSV)
        st.session_state["alignment_event_pixels_path"] = str(DEFAULT_EBSD_CUT_EVENT_PIXELS_CSV)

    if events_file is not None:
        st.session_state["alignment_events_path"] = str(
            save_uploaded_streamlit_file(events_file, "alignment_events")
        )
    if event_pixels_file is not None:
        st.session_state["alignment_event_pixels_path"] = str(
            save_uploaded_streamlit_file(event_pixels_file, "alignment_events")
        )

    events_path = st.session_state.get("alignment_events_path", "")
    event_pixels_path = st.session_state.get("alignment_event_pixels_path", "")
    event_labels = []
    if events_path:
        event_labels.append(f"events: `{Path(events_path).name}`")
    if event_pixels_path:
        event_labels.append(f"event pixels: `{Path(event_pixels_path).name}`")
    if event_labels:
        st.caption("Loaded " + " | ".join(event_labels))
    else:
        st.info("Load the EBSD-cut events CSV and event-pixels CSV for the next alignment step.")
        return

    if not Path(events_path).exists():
        st.error(f"Events CSV not found: {events_path}")
        return
    if not Path(event_pixels_path).exists():
        st.error(f"Event pixels CSV not found: {event_pixels_path}")
        return

    try:
        events_signature = file_cache_signature(events_path)
        event_pixels_signature = file_cache_signature(event_pixels_path)
        event_info = load_alignment_event_data(
            events_path,
            event_pixels_path,
            events_signature,
            event_pixels_signature,
        )
    except Exception as exc:
        st.error(f"Could not load EBSD-cut event data: {exc}")
        return

    event_metrics = event_info["metrics"]
    metric_cols = st.columns(5)
    metric_cols[0].metric("Events", f"{event_metrics['event_count']:,}")
    metric_cols[1].metric("Event pixels", f"{event_metrics['pixel_count']:,}")
    metric_cols[2].metric("Pixel event IDs", f"{event_metrics['pixel_event_count']:,}")
    metric_cols[3].metric("Min pixels/event", f"{event_metrics['min_pixels_per_event']:,}")
    metric_cols[4].metric("Max pixels/event", f"{event_metrics['max_pixels_per_event']:,}")

    table1, table2 = st.columns(2)
    with table1:
        st.caption("Event details")
        st.dataframe(event_info["events"].head(100), width="stretch", hide_index=True)
    with table2:
        st.caption("Event pixel details")
        st.dataframe(event_info["event_pixels"].head(100), width="stretch", hide_index=True)

    st.markdown("**EBSD-Cut Event Shape Analysis**")
    st.caption(
        "Fits one PCA/SVD best-fit line to each cut event and classifies the shape from linearity, "
        "aspect ratio, pixel density, and connected components."
    )
    shape_cols = st.columns(4)
    with shape_cols[0]:
        morphology_min_pixels = st.slider(
            "Minimum pixels",
            1,
            500,
            40,
            1,
            key="alignment_morphology_min_pixels",
            help="Events below this pixel count are classified as small_noise.",
        )
    with shape_cols[1]:
        morphology_linearity_threshold = st.slider(
            "Linearity threshold",
            0.0,
            1.0,
            0.80,
            0.01,
            key="alignment_morphology_linearity_threshold",
            help="Higher values require the event pixels to lie more strongly along one best-fit line.",
        )
    with shape_cols[2]:
        morphology_aspect_threshold = st.slider(
            "Aspect ratio threshold",
            1.0,
            30.0,
            3.0,
            0.5,
            key="alignment_morphology_aspect_threshold",
            help="PCA major/minor spread ratio. Higher values require longer, thinner events to be called linear.",
        )
    with shape_cols[3]:
        morphology_blob_density_threshold = st.slider(
            "Blob density threshold",
            0.01,
            1.0,
            0.25,
            0.01,
            key="alignment_morphology_blob_density_threshold",
            help="Pixels divided by bounding-box area. Dense non-linear events above this value are classified as blob_like.",
        )

    morphology_signature = (
        str(events_path),
        str(event_pixels_path),
        file_cache_signature(events_path),
        file_cache_signature(event_pixels_path),
        int(morphology_min_pixels),
        float(morphology_linearity_threshold),
        float(morphology_aspect_threshold),
        float(morphology_blob_density_threshold),
    )
    if st.button("Analyze EBSD-cut event shapes", key="alignment_analyze_event_shapes"):
        try:
            with st.spinner("Fitting event lines and classifying event shapes..."):
                morphology_result = compute_event_morphology_features(
                    event_info["events"],
                    event_info["event_pixels"],
                    min_pixels=int(morphology_min_pixels),
                    linearity_threshold=float(morphology_linearity_threshold),
                    aspect_threshold=float(morphology_aspect_threshold),
                    blob_density_threshold=float(morphology_blob_density_threshold),
                )
        except Exception as exc:
            st.error(f"Event shape analysis failed: {exc}")
        else:
            st.session_state["alignment_event_morphology_result"] = morphology_result
            st.session_state["alignment_event_morphology_signature"] = morphology_signature

    morphology_result = st.session_state.get("alignment_event_morphology_result")
    if morphology_result is not None:
        if st.session_state.get("alignment_event_morphology_signature") != morphology_signature:
            st.info("Event-shape analysis needs to be rerun for the current files or thresholds.")
            morphology_result = None

    if morphology_result is not None:
        morphology_counts = morphology_result["event_type"].value_counts().to_dict()
        count_cols = st.columns(5)
        count_cols[0].metric("Linear", f"{morphology_counts.get('linear', 0):,}")
        count_cols[1].metric("Blob-like", f"{morphology_counts.get('blob_like', 0):,}")
        count_cols[2].metric("Irregular", f"{morphology_counts.get('irregular', 0):,}")
        count_cols[3].metric("Fragmented", f"{morphology_counts.get('fragmented', 0):,}")
        count_cols[4].metric("Small noise", f"{morphology_counts.get('small_noise', 0):,}")

        st.dataframe(morphology_result, width="stretch", hide_index=True)

        st.markdown("**Event Shape Overlays**")
        st.caption(
            "Each overlay uses the BLN image as the background and draws only one event type at a time. "
            "Large overlays are rendered as downsampled previews to avoid memory spikes."
        )
        shape_overlay_max_dim = st.slider(
            "Shape overlay preview max dimension",
            min_value=800,
            max_value=3200,
            value=1800,
            step=200,
            key="alignment_shape_overlay_max_dim",
            help="Maximum width or height used for event-shape overlay previews. Lower values load faster and use less memory.",
        )
        morphology_bln_path = infer_event_image_path_from_frame(event_info["events"]) or str(DEFAULT_IMAGE)
        if not Path(morphology_bln_path).exists():
            st.warning(f"BLN image not found for overlays: {morphology_bln_path}")
        else:
            overlay_specs = [
                ("linear", "Linear events"),
                ("blob_like", "Blob-like events"),
                ("irregular", "Irregular events"),
                ("fragmented", "Fragmented events"),
                ("small_noise", "Noise events"),
            ]
            overlay_signature = (
                morphology_signature,
                str(morphology_bln_path),
                str(DEFAULT_EBSD_BOUNDARY_MAP),
            )
            if st.button("Load event shape overlays", key="alignment_load_event_shape_overlays"):
                use_full_shape_overlay = event_shape_overlay_uses_full_image(
                    event_info["events"],
                    event_info["event_pixels"],
                    morphology_bln_path,
                )
                with st.spinner("Building event-shape overlays..."):
                    overlay_bundle = {}
                    for event_type, title in overlay_specs:
                        overlay_bundle[event_type] = {
                            "title": title,
                            "count": int((morphology_result["event_type"].astype(str) == event_type).sum()),
                            "result": build_filtered_event_type_overlay(
                                morphology_bln_path,
                                event_info["event_pixels"],
                                morphology_result,
                                event_type=event_type,
                                margin=0,
                                max_dim=int(shape_overlay_max_dim),
                                full_image=use_full_shape_overlay,
                                boundary_path=str(DEFAULT_EBSD_BOUNDARY_MAP),
                                boundary_threshold=0.20,
                            ),
                        }
                    st.session_state["alignment_event_shape_overlay_bundle"] = overlay_bundle
                    st.session_state["alignment_event_shape_overlay_signature"] = overlay_signature

            overlay_bundle = st.session_state.get("alignment_event_shape_overlay_bundle")
            if overlay_bundle is not None:
                if st.session_state.get("alignment_event_shape_overlay_signature") != overlay_signature:
                    st.info("Event-shape overlays need to be reloaded for the current files or thresholds.")
                else:
                    overlay_tabs = st.tabs([title for _, title in overlay_specs])
                    for tab, (event_type, _title) in zip(overlay_tabs, overlay_specs):
                        with tab:
                            entry = overlay_bundle.get(event_type)
                            if entry is None:
                                st.info("Overlay not loaded.")
                                continue
                            overlay_result = entry["result"]
                            crop = overlay_result["crop"]
                            st.caption(
                                f"Overlay canvas: x={crop['x']}, y={crop['y']}, "
                                f"width={crop['width']}, height={crop['height']}"
                            )
                            zoomable_image(
                                overlay_result["overlay"],
                                image_title_with_dimensions(
                                    f"{entry['title']} ({entry['count']:,})",
                                    overlay_result["overlay"],
                                ),
                                key=f"alignment_morphology_overlay_{event_type}",
                            )

    st.markdown("**Event Reality Scoring**")
    st.caption(
        "Classifies each EBSD-cut event by fitting one PCA/SVD line to the event, finding the nearest transformed "
        "ANG point to the event center, and comparing that line to the active trace modes selected in Crystal Setup."
    )
    score_cols = st.columns(3)
    with score_cols[0]:
        angle_tolerance_deg = st.slider(
            "Angle tolerance",
            1.0,
            45.0,
            5.0,
            1.0,
            key="alignment_score_angle_tolerance",
            help="Fallback angle tolerance if Crystal Setup has not provided mode-specific tolerances.",
        )
    with score_cols[1]:
        max_lookup_distance = st.slider(
            "Max center lookup distance",
            1.0,
            100.0,
            20.0,
            1.0,
            key="alignment_score_center_lookup_distance",
            help="Nearest transformed ANG point distance from event center. Farther events are classified as noise.",
        )
    with score_cols[2]:
        min_linearity = st.slider(
            "Minimum linearity",
            0.0,
            1.0,
            0.0,
            0.05,
            key="alignment_score_min_linearity",
            help="Optional line-quality gate. Set to 0 to classify only by angle and lookup distance.",
        )

    can_score = transform is not None and dic_coords is not None
    if not can_score:
        st.info("Load transformation JSON and ANG data first, then event scoring can use transformed ANG trace vectors.")
    score_signature = (
        str(events_path),
        str(event_pixels_path),
        file_cache_signature(events_path),
        file_cache_signature(event_pixels_path),
        morphology_signature,
        json.dumps(st.session_state.get("crystal_setup_config", {}), sort_keys=True, default=str),
        float(angle_tolerance_deg),
        float(max_lookup_distance),
        float(min_linearity),
    )
    if st.button("Score events", type="primary", key="alignment_score_events", disabled=not can_score):
        try:
            with st.spinner("Scoring events against selected crystal trace directions..."):
                scoring_events = event_info["events"]
                scoring_event_pixels = event_info["event_pixels"]
                excluded_blob_ids: set[str] = set()
                score_diagnostics = {
                    "loaded_event_rows": int(len(event_info["events"])),
                    "loaded_event_pixel_ids": int(event_info["event_pixels"]["event_id"].astype(str).nunique()),
                    "morphology_rows": int(len(morphology_result)) if morphology_result is not None else 0,
                    "morphology_blob_like": 0,
                    "blob_ids_intersecting_loaded_pixels": 0,
                    "blob_ids_missing_from_loaded_pixels": 0,
                    "events_sent_to_scoring": 0,
                }
                if morphology_result is not None and "event_type" in morphology_result.columns:
                    morphology_identity_col = event_identity_column(morphology_result)
                    scoring_events_identity_col = event_identity_column(scoring_events)
                    scoring_pixels_identity_col = event_identity_column(scoring_event_pixels)
                    excluded_blob_ids = set(
                        morphology_result.loc[
                            morphology_result["event_type"].astype(str) == "blob_like",
                            morphology_identity_col,
                        ].astype(str)
                    )
                    loaded_pixel_ids = set(scoring_event_pixels[scoring_pixels_identity_col].astype(str))
                    score_diagnostics["morphology_blob_like"] = int(len(excluded_blob_ids))
                    score_diagnostics["blob_ids_intersecting_loaded_pixels"] = int(len(excluded_blob_ids & loaded_pixel_ids))
                    score_diagnostics["blob_ids_missing_from_loaded_pixels"] = int(len(excluded_blob_ids - loaded_pixel_ids))
                    if excluded_blob_ids:
                        scoring_events = scoring_events[
                            ~scoring_events[scoring_events_identity_col].astype(str).isin(excluded_blob_ids)
                        ].copy()
                        scoring_event_pixels = scoring_event_pixels[
                            ~scoring_event_pixels[scoring_pixels_identity_col].astype(str).isin(excluded_blob_ids)
                        ].copy()
                score_diagnostics["events_sent_to_scoring"] = int(scoring_event_pixels["event_id"].astype(str).nunique())
                score_result = score_events_with_trace_alignment(
                    scoring_events,
                    scoring_event_pixels,
                    dic_coords,
                    transform["forward_params"],
                    angle_tolerance_deg=float(angle_tolerance_deg),
                    max_lookup_distance=float(max_lookup_distance),
                    min_linearity=float(min_linearity),
                    crystal_config=st.session_state.get("crystal_setup_config"),
                )
        except Exception as exc:
            st.error(f"Event scoring failed: {exc}")
        else:
            st.session_state["alignment_event_score_result"] = score_result
            st.session_state["alignment_event_score_excluded_blob_ids"] = sorted(excluded_blob_ids)
            st.session_state["alignment_event_score_diagnostics"] = score_diagnostics
            st.session_state["alignment_event_score_signature"] = score_signature

    score_result = st.session_state.get("alignment_event_score_result")
    if score_result is not None:
        if st.session_state.get("alignment_event_score_signature") != score_signature:
            st.info("Event scoring needs to be rerun for the current files, shape analysis, crystal setup, or scoring thresholds.")
            score_result = None
    if score_result is not None:
        class_counts = score_result["classification"].value_counts().to_dict()
        excluded_blob_ids = set(st.session_state.get("alignment_event_score_excluded_blob_ids", []))
        result_cols = st.columns(5)
        result_cols[0].metric("Scored events", f"{len(score_result):,}")
        result_cols[1].metric("Matched events", f"{class_counts.get('likely_real', 0):,}")
        result_cols[2].metric("Uncertain", f"{class_counts.get('uncertain', 0):,}")
        result_cols[3].metric("Noise events", f"{class_counts.get('likely_noise', 0):,}")
        result_cols[4].metric("Blob-like excluded", f"{len(excluded_blob_ids):,}")
        diagnostics = st.session_state.get("alignment_event_score_diagnostics", {})
        if diagnostics:
            diag_cols = st.columns(6)
            diag_cols[0].metric("Loaded event rows", f"{diagnostics.get('loaded_event_rows', 0):,}")
            diag_cols[1].metric("Loaded pixel IDs", f"{diagnostics.get('loaded_event_pixel_ids', 0):,}")
            diag_cols[2].metric("Shape rows", f"{diagnostics.get('morphology_rows', 0):,}")
            diag_cols[3].metric("Shape blobs", f"{diagnostics.get('morphology_blob_like', 0):,}")
            diag_cols[4].metric("Blob IDs in pixels", f"{diagnostics.get('blob_ids_intersecting_loaded_pixels', 0):,}")
            diag_cols[5].metric("Sent to scoring", f"{diagnostics.get('events_sent_to_scoring', 0):,}")
        st.dataframe(score_result, width="stretch", hide_index=True)
        st.markdown("**Reality Score Overlay**")
        st.caption(
            "BLN background uses brightness/contrast range 0-1. "
            "Green = likely real, red = likely noise, yellow = uncertain, blue = EBSD boundary, "
            "cyan = event PCA direction, magenta = selected trace vector. "
            "In the all-trace plot: teal = slip traces and green = twin traces."
        )
        overlay_boundary = st.checkbox(
            "Overlay EBSD boundary",
            value=True,
            key="alignment_score_overlay_boundary_enabled",
            help="Draw aligned EBSD boundary pixels on top of the BLN/event classification overlay.",
        )
        boundary_path = ""
        boundary_threshold = 0.20
        if overlay_boundary:
            boundary_path = st.text_input(
                "EBSD boundary map",
                str(DEFAULT_EBSD_BOUNDARY_MAP),
                key="alignment_score_overlay_boundary_path",
                help="Black/white aligned EBSD boundary image. Dark pixels are treated as boundaries.",
            )
            boundary_threshold = st.slider(
                "Boundary black threshold",
                0.0,
                1.0,
                0.20,
                0.01,
                key="alignment_score_overlay_boundary_threshold",
            )
        overlay_event_vectors = st.checkbox(
            "Overlay event vectors",
            value=True,
            key="alignment_score_overlay_vectors_enabled",
            help="Draw each event's PCA/SVD main direction as a cyan line through the event centroid.",
        )
        overlay_trace_vectors = st.checkbox(
            "Overlay selected trace vectors",
            value=True,
            key="alignment_score_overlay_trace_vectors_enabled",
            help="Draw the trace vector that produced the assigned angle match for each event.",
        )
        overlay_all_trace_vectors = True
        vector_cols = st.columns(2)
        with vector_cols[0]:
            event_vector_scale = st.slider(
                "Event vector scale",
                0.25,
                2.0,
                1.0,
                0.25,
                key="alignment_score_overlay_vector_scale",
                disabled=not overlay_event_vectors,
                help="Scales the displayed vector length relative to the event's PCA length.",
            )
        with vector_cols[1]:
            event_vector_width = st.slider(
                "Event vector width",
                1,
                9,
                3,
                1,
                key="alignment_score_overlay_vector_width",
                disabled=not overlay_event_vectors,
            )
        full_score_overlay = st.checkbox(
            "Show full BLN image",
            value=False,
            key="alignment_score_overlay_full_image",
            help="Keep disabled for faster crop-based inspection. Enable only for a full-image final check.",
        )
        display_margin = st.slider(
            "Overlay display margin",
            0,
            1000,
            150,
            25,
            key="alignment_score_overlay_margin",
            disabled=full_score_overlay,
        )
        use_full_resolution_overlay = st.checkbox(
            "Use full-resolution overlay",
            value=True,
            key="alignment_score_overlay_full_resolution",
            help="Keeps every pixel in the selected overlay crop. Disable to downsample large overlays.",
        )
        if use_full_resolution_overlay:
            overlay_max_dim = 0
        else:
            overlay_max_dim = st.slider(
                "Overlay max dimension",
                800,
                3600,
                2200,
                100,
                key="alignment_score_overlay_max_dim",
            )
        try:
            bln_path = infer_event_image_path_from_frame(event_info["events"]) or str(DEFAULT_IMAGE)
            score_identity_col = event_identity_column(score_result)
            pixel_identity_col = event_identity_column(event_info["event_pixels"])
            scored_event_pixels = event_info["event_pixels"][
                event_info["event_pixels"][pixel_identity_col].astype(str).isin(
                    set(score_result[score_identity_col].astype(str))
                )
            ]
            summary_overlay = build_event_reality_overlay(
                bln_path,
                scored_event_pixels,
                score_result,
                full_image=bool(full_score_overlay),
                margin=int(display_margin),
                max_dim=int(overlay_max_dim),
                use_full_resolution=bool(use_full_resolution_overlay),
                boundary_path=str(Path(boundary_path).expanduser()) if boundary_path else "",
                boundary_threshold=float(boundary_threshold),
                draw_vectors=False,
                draw_trace_vectors=False,
                vector_scale=float(event_vector_scale),
                vector_width=int(event_vector_width),
            )
            score_overlay = build_event_reality_overlay(
                bln_path,
                scored_event_pixels,
                score_result,
                full_image=bool(full_score_overlay),
                margin=int(display_margin),
                max_dim=int(overlay_max_dim),
                use_full_resolution=bool(use_full_resolution_overlay),
                boundary_path=str(Path(boundary_path).expanduser()) if boundary_path else "",
                boundary_threshold=float(boundary_threshold),
                draw_vectors=bool(overlay_event_vectors),
                draw_trace_vectors=bool(overlay_trace_vectors),
                vector_scale=float(event_vector_scale),
                vector_width=int(event_vector_width),
            )
            all_trace_overlay = build_event_reality_overlay(
                bln_path,
                scored_event_pixels,
                score_result,
                full_image=bool(full_score_overlay),
                margin=int(display_margin),
                max_dim=int(overlay_max_dim),
                use_full_resolution=bool(use_full_resolution_overlay),
                boundary_path=str(Path(boundary_path).expanduser()) if boundary_path else "",
                boundary_threshold=float(boundary_threshold),
                draw_vectors=False,
                draw_trace_vectors=False,
                draw_all_trace_vectors=True,
                vector_scale=float(event_vector_scale),
                vector_width=int(event_vector_width),
            )
        except Exception as exc:
            st.error(f"Could not build reality score overlay: {exc}")
        else:
            zoomable_image(
                summary_overlay["overlay"],
                image_title_with_dimensions("BLN event classification overlay", summary_overlay["overlay"]),
                key="alignment_event_classification_overlay",
            )
            zoomable_image(
                score_overlay["overlay"],
                image_title_with_dimensions("BLN event reality overlay with vectors", score_overlay["overlay"]),
                key="alignment_event_reality_overlay",
            )
            zoomable_image(
                all_trace_overlay["overlay"],
                image_title_with_dimensions("BLN event overlay with all trace vectors", all_trace_overlay["overlay"]),
                key="alignment_event_all_trace_overlay",
            )
            st.markdown("**Save Trace Comparison Table**")
            save_cols = st.columns([2, 1, 1])
            with save_cols[0]:
                trace_summary_prefix = st.text_input(
                    "Save filename prefix",
                    "event_trace_comparison",
                    key="alignment_trace_summary_prefix",
                    help="Used for the trace CSV and the full-resolution overlay PNG files.",
                )
            with save_cols[1]:
                if st.button("Save trace comparison CSV", key="alignment_save_trace_summary"):
                    try:
                        out_path = save_trace_comparison_summary(score_result, trace_summary_prefix)
                    except Exception as exc:
                        st.error(f"Could not save trace comparison CSV: {exc}")
                    else:
                        st.success(f"Saved trace comparison CSV:\n\n{out_path}")
            with save_cols[2]:
                if st.button("Save full-resolution overlays", key="alignment_save_full_resolution_overlays"):
                    try:
                        with st.spinner("Building and saving full-resolution overlay images..."):
                            full_summary_overlay = build_event_reality_overlay(
                                bln_path,
                                scored_event_pixels,
                                score_result,
                                full_image=bool(full_score_overlay),
                                margin=int(display_margin),
                                max_dim=0,
                                use_full_resolution=True,
                                boundary_path=str(Path(boundary_path).expanduser()) if boundary_path else "",
                                boundary_threshold=float(boundary_threshold),
                                draw_vectors=False,
                                draw_trace_vectors=False,
                                vector_scale=float(event_vector_scale),
                                vector_width=int(event_vector_width),
                            )
                            full_score_overlay_result = build_event_reality_overlay(
                                bln_path,
                                scored_event_pixels,
                                score_result,
                                full_image=bool(full_score_overlay),
                                margin=int(display_margin),
                                max_dim=0,
                                use_full_resolution=True,
                                boundary_path=str(Path(boundary_path).expanduser()) if boundary_path else "",
                                boundary_threshold=float(boundary_threshold),
                                draw_vectors=bool(overlay_event_vectors),
                                draw_trace_vectors=bool(overlay_trace_vectors),
                                vector_scale=float(event_vector_scale),
                                vector_width=int(event_vector_width),
                            )
                            full_all_trace_overlay = build_event_reality_overlay(
                                bln_path,
                                scored_event_pixels,
                                score_result,
                                full_image=bool(full_score_overlay),
                                margin=int(display_margin),
                                max_dim=0,
                                use_full_resolution=True,
                                boundary_path=str(Path(boundary_path).expanduser()) if boundary_path else "",
                                boundary_threshold=float(boundary_threshold),
                                draw_vectors=False,
                                draw_trace_vectors=False,
                                draw_all_trace_vectors=True,
                                vector_scale=float(event_vector_scale),
                                vector_width=int(event_vector_width),
                            )
                            saved_paths = save_alignment_overlay_images(
                                full_summary_overlay["overlay"],
                                full_score_overlay_result["overlay"],
                                full_all_trace_overlay["overlay"],
                                trace_summary_prefix,
                            )
                    except Exception as exc:
                        st.error(f"Could not save full-resolution overlays: {exc}")
                    else:
                        st.success("Saved full-resolution overlays:\n\n" + "\n".join(f"- {path}" for path in saved_paths))
            st.markdown("**Save Scored Event Categories**")
            category_cols = st.columns([2, 1])
            with category_cols[0]:
                category_prefix = st.text_input(
                    "Category CSV filename prefix",
                    f"{trace_summary_prefix}_categories",
                    key="alignment_score_category_prefix",
                    help="Creates separate CSV files for matched, uncertain, noise, and blob-like events.",
                )
            with category_cols[1]:
                if st.button("Save category CSVs", key="alignment_save_score_category_csvs"):
                    try:
                        saved_category_paths = save_event_score_category_csvs(
                            score_result,
                            morphology_result,
                            event_info["events"],
                            event_info["event_pixels"],
                            category_prefix,
                        )
                    except Exception as exc:
                        st.error(f"Could not save category CSVs: {exc}")
                    else:
                        st.success(
                            "Saved category CSVs:\n\n"
                            + "\n".join(f"- {path}" for path in saved_category_paths)
                            + "\n\nUse any `*_event_pixels.csv` file as the detected event pixels input in Detection Analysis."
                        )
            st.markdown("**Download Per-Event Trace Plots**")
            if st.button("Prepare per-event trace plot ZIP", key="alignment_prepare_event_trace_plot_zip"):
                try:
                    with st.spinner("Creating one trace diagnostic plot per event..."):
                        zip_bytes, zip_name = build_event_trace_plot_zip(
                            score_result,
                            scored_event_pixels,
                            trace_summary_prefix,
                        )
                except Exception as exc:
                    st.error(f"Could not create per-event trace plots: {exc}")
                else:
                    st.session_state["alignment_event_trace_plot_zip_bytes"] = zip_bytes
                    st.session_state["alignment_event_trace_plot_zip_name"] = zip_name
            if "alignment_event_trace_plot_zip_bytes" in st.session_state:
                st.download_button(
                    "Download per-event trace plot ZIP",
                    data=st.session_state["alignment_event_trace_plot_zip_bytes"],
                    file_name=st.session_state.get("alignment_event_trace_plot_zip_name", "event_trace_plots.zip"),
                    mime="application/zip",
                    key="alignment_download_event_trace_plot_zip",
                )


def event_analysis_tab() -> None:
    st.subheader("Cut Event Analysis")
    st.caption(
        "Inspect how an original detected event was split by the EBSD boundary cut."
    )

    col1, col2 = st.columns(2)
    with col1:
        original_pixels_file = st.file_uploader(
            "Original event pixels CSV",
            type=["csv"],
            key="event_cut_inspector_original_pixels_file",
            help="Pre-boundary-cut event pixels CSV with event_id, pixel_x, pixel_y.",
        )
        cut_events_file = st.file_uploader(
            "Cut events CSV",
            type=["csv"],
            key="event_cut_inspector_cut_events_file",
        )
    with col2:
        cut_pixels_file = st.file_uploader(
            "Cut event pixels CSV",
            type=["csv"],
            key="event_cut_inspector_cut_pixels_file",
        )
        boundary_file = st.file_uploader(
            "EBSD boundary image",
            type=["png", "bmp", "tif", "tiff", "jpg", "jpeg"],
            key="event_cut_inspector_boundary_file",
        )

    if st.button("Use default boundary-cut inspection files", key="event_cut_inspector_use_default_files"):
        st.session_state["event_cut_inspector_original_pixels_path"] = str(DEFAULT_FULL_DETECTED_EVENT_PIXELS_CSV)
        st.session_state["event_cut_inspector_cut_events_path"] = str(DEFAULT_EBSD_CUT_EVENTS_CSV)
        st.session_state["event_cut_inspector_cut_pixels_path"] = str(DEFAULT_EBSD_CUT_EVENT_PIXELS_CSV)
        st.session_state["event_cut_inspector_boundary_path"] = str(DEFAULT_EBSD_BOUNDARY_MAP)

    with st.expander("Default files used by this tab", expanded=False):
        st.code(
            "\n".join(
                [
                    f"Original event pixels CSV: {DEFAULT_FULL_DETECTED_EVENT_PIXELS_CSV}",
                    f"Cut events CSV: {DEFAULT_EBSD_CUT_EVENTS_CSV}",
                    f"Cut event pixels CSV: {DEFAULT_EBSD_CUT_EVENT_PIXELS_CSV}",
                    f"EBSD boundary image: {DEFAULT_EBSD_BOUNDARY_MAP}",
                ]
            ),
            language="text",
        )

    if original_pixels_file is not None:
        st.session_state["event_cut_inspector_original_pixels_path"] = str(
            save_uploaded_streamlit_file(original_pixels_file, "event_cut_inspector")
        )
    if cut_events_file is not None:
        st.session_state["event_cut_inspector_cut_events_path"] = str(
            save_uploaded_streamlit_file(cut_events_file, "event_cut_inspector")
        )
    if cut_pixels_file is not None:
        st.session_state["event_cut_inspector_cut_pixels_path"] = str(
            save_uploaded_streamlit_file(cut_pixels_file, "event_cut_inspector")
        )
    if boundary_file is not None:
        st.session_state["event_cut_inspector_boundary_path"] = str(
            save_uploaded_streamlit_file(boundary_file, "event_cut_inspector")
        )

    original_pixels_path = st.session_state.get("event_cut_inspector_original_pixels_path", "")
    cut_events_path = st.session_state.get("event_cut_inspector_cut_events_path", "")
    cut_pixels_path = st.session_state.get("event_cut_inspector_cut_pixels_path", "")
    boundary_path = st.session_state.get("event_cut_inspector_boundary_path", "")

    paths = {
        "Original event pixels CSV": original_pixels_path,
        "Cut events CSV": cut_events_path,
        "Cut event pixels CSV": cut_pixels_path,
        "EBSD boundary image": boundary_path,
    }
    missing = [label for label, path in paths.items() if not path or not Path(path).exists()]
    if missing:
        st.info("Load all boundary-cut inspection files, or use the default files.")
        if any(paths.values()):
            st.warning("Missing: " + ", ".join(missing))
        return

    original_pixels, cut_events, cut_pixels = load_boundary_cut_inspection_data(
        original_pixels_path,
        cut_events_path,
        cut_pixels_path,
    )

    original_ids = sorted(original_pixels["event_id"].astype(str).unique())
    produced_ids = sorted(cut_events["original_event_id"].dropna().astype(str).unique()) if "original_event_id" in cut_events.columns else []
    selectable_ids = produced_ids or original_ids
    selected_original_id = st.selectbox(
        "Original event",
        selectable_ids,
        key="event_cut_inspector_original_event_id",
    )

    controls = st.columns(4)
    with controls[0]:
        margin = st.slider("Crop margin", 10, 1000, 120, 10, key="event_cut_inspector_margin")
    with controls[1]:
        boundary_threshold = st.slider("Boundary black threshold", 0.0, 1.0, 0.20, 0.01, key="event_cut_inspector_boundary_threshold")
    with controls[2]:
        boundary_dilation = st.slider("Boundary display dilation", 0, 8, 1, 1, key="event_cut_inspector_boundary_dilation")
    with controls[3]:
        show_labels = st.checkbox("Label cut segments", value=True, key="event_cut_inspector_labels")

    try:
        bln_path = infer_event_image_path_from_frame(cut_events) or str(DEFAULT_IMAGE)
        inspection = build_boundary_cut_inspection_overlay(
            bln_path,
            boundary_path,
            original_pixels,
            cut_events,
            cut_pixels,
            selected_original_id,
            margin=int(margin),
            boundary_threshold=float(boundary_threshold),
            boundary_dilation=int(boundary_dilation),
            show_labels=bool(show_labels),
        )
    except Exception as exc:
        st.error(f"Could not build boundary-cut inspection plot: {exc}")
        return

    stats = inspection["stats"]
    stat_cols = st.columns(5)
    stat_cols[0].metric("Original pixels", f"{stats['original_pixels']:,}")
    stat_cols[1].metric("Cut segments", f"{stats['cut_segments']:,}")
    stat_cols[2].metric("Kept pixels", f"{stats['kept_pixels']:,}")
    stat_cols[3].metric("Cut/removed pixels", f"{stats['removed_pixels']:,}")
    stat_cols[4].metric("Boundary pixels in crop", f"{stats['boundary_pixels']:,}")

    st.caption(
        "Gray = original event, red = produced cut segments, purple = pixels removed by boundary cut, "
        "blue = EBSD boundary."
    )
    zoomable_image(
        inspection["overlay"],
        image_title_with_dimensions(f"Boundary cut inspection: {selected_original_id}", inspection["overlay"]),
        key="event_cut_inspector_overlay",
    )
    st.dataframe(inspection["cut_events"], width="stretch", hide_index=True)


def manual_boundary_cut_tab() -> None:
    st.markdown("**Manual Cut From Saved CSVs**")
    col1, col2 = st.columns(2)
    with col1:
        event_pixels_path = st.text_input(
            "Event pixels CSV",
            str(DEFAULT_FULL_DETECTED_EVENT_PIXELS_CSV),
            key="boundary_cut_event_pixels_path",
            help="CSV with event_id, pixel_x, pixel_y columns in global image coordinates.",
        )
        events_path = st.text_input(
            "Events CSV",
            str(DEFAULT_FULL_DETECTED_EVENTS_CSV),
            key="boundary_cut_events_path",
            help="Optional event-level CSV. It is used for reporting and is not required for the pixel cut itself.",
        )
    with col2:
        boundary_path = st.text_input(
            "EBSD boundary map",
            str(DEFAULT_EBSD_BOUNDARY_MAP),
            key="boundary_cut_boundary_path",
            help="Black/white image where black pixels are grain boundaries.",
        )

    event_pixels_path = str(Path(event_pixels_path).expanduser())
    events_path = str(Path(events_path).expanduser())
    boundary_path = str(Path(boundary_path).expanduser())

    required_paths = {
        "Event pixels CSV": event_pixels_path,
        "EBSD boundary map": boundary_path,
    }
    for label, path in required_paths.items():
        if not Path(path).exists():
            st.error(f"{label} not found: {path}")
            return
    if events_path.strip() and not Path(events_path).exists():
        st.warning(f"Events CSV not found. The cut can still run from event pixels only: {events_path}")
    if st.button("Load EBSD boundary image", key="manual_boundary_cut_load_boundary"):
        boundary_w, boundary_h = image_size(boundary_path)
        st.success(f"Loaded EBSD boundary image: {boundary_w} x {boundary_h} pixels")
    inferred_bln_path = infer_event_image_path(events_path)
    if inferred_bln_path:
        st.caption(f"BLN background image inferred from events CSV: `{inferred_bln_path}`")
    else:
        st.caption(f"BLN background image fallback: `{DEFAULT_IMAGE}`")

    st.markdown("**Cut Controls**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        black_threshold = st.slider(
            "Black boundary threshold",
            0.0,
            1.0,
            0.20,
            0.01,
            key="boundary_cut_black_threshold",
            help="Pixels darker than this are treated as grain-boundary pixels.",
        )
    with c2:
        boundary_dilation_radius = st.slider(
            "Boundary dilation radius",
            0,
            8,
            1,
            1,
            key="boundary_cut_dilation_radius",
            help="Expands boundary pixels before cutting. Higher values cut more aggressively near boundaries.",
        )
    with c3:
        min_segment_pixels = st.slider(
            "Minimum segment pixels",
            1,
            500,
            40,
            1,
            key="boundary_cut_min_segment_pixels",
            help="Drops cut fragments smaller than this many pixels.",
        )
    with c4:
        connectivity = st.radio(
            "Connectivity",
            [1, 2],
            index=1,
            horizontal=True,
            key="boundary_cut_connectivity",
            help="1 is 4-connected. 2 is 8-connected and keeps diagonal event pixels connected.",
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        display_margin = st.slider(
            "Display margin",
            0,
            1000,
            150,
            25,
            key="boundary_cut_display_margin",
            help="Extra pixels around the detected-event extent for the before/after overlays.",
        )
    with c2:
        use_full_resolution_overlay = st.checkbox(
            "Use full-resolution overlay",
            value=True,
            key="boundary_cut_use_full_resolution_overlay",
            help="Keeps every image pixel in the zoom viewer. This shows true pixel data but can be slower for the full Ti image.",
        )
    with c3:
        save_prefix = st.text_input(
            "Save prefix",
            "full_ti_image_detection_boundary_cut",
            key="boundary_cut_save_prefix",
            help="Used when saving the cut events and cut event pixels CSV files.",
        )
    with c4:
        save_min_event_pixels = st.slider(
            "Save pixel threshold",
            1,
            5000,
            40,
            10,
            key="boundary_cut_save_min_event_pixels",
            help="Only cut events with at least this many pixels are written to the saved CSVs.",
        )

    if use_full_resolution_overlay:
        display_max_dim = 0
        st.warning(
            "Full-resolution overlay is enabled. The zoom viewer will receive the real pixel-size crop, "
            "which is best for inspection but can take longer to render."
        )
    else:
        display_max_dim = st.slider(
            "Display max dimension",
            800,
            3600,
            2200,
            100,
            key="boundary_cut_display_max_dim",
            help="Downsamples large overlays before sending them to the browser. Higher values show more detail but are slower.",
        )

    run_clicked = st.button("Run boundary cut", type="primary", key="run_boundary_cut")

    signature = (
        event_pixels_path,
        events_path,
        boundary_path,
        float(black_threshold),
        int(boundary_dilation_radius),
        int(min_segment_pixels),
        int(connectivity),
        int(display_margin),
        int(display_max_dim),
        bool(use_full_resolution_overlay),
    )

    if run_clicked:
        try:
            with st.spinner("Cutting events against EBSD boundary map..."):
                result = run_boundary_cut_analysis(*signature)
        except Exception as exc:
            st.error(f"Boundary cut failed: {exc}")
            return
        st.session_state["last_boundary_cut_signature"] = signature
        st.session_state["last_boundary_cut_result"] = result

    result = st.session_state.get("last_boundary_cut_result")
    stored_signature = st.session_state.get("last_boundary_cut_signature")
    if result is None:
        st.info("Press Run boundary cut to split detected events at grain boundaries.")
        return
    if stored_signature != signature:
        st.warning("Displayed boundary-cut result is from previous inputs or parameters. Press Run boundary cut to update.")

    show_boundary_cut_result(result, min_event_pixels=int(save_min_event_pixels))

    if st.button("Save cut events CSVs", key="save_boundary_cut_csvs"):
        safe_prefix = sanitize_file_prefix(save_prefix) or "boundary_cut_events"
        saved = save_boundary_cut_outputs(result, safe_prefix, min_event_pixels=int(save_min_event_pixels))
        st.success("Saved boundary-cut outputs:\n\n" + "\n".join(f"- {path}" for path in saved))


def auto_boundary_cut_tab() -> None:
    st.markdown("**Auto Cut From Current Hough Detection**")
    st.caption(
        "Uses the current Hough-traced events and the exact processing crop from the Detection Results tab. "
        "Run Hough Step 1 and Step 2 first."
    )

    stored_result = st.session_state.get("last_pipeline_result")
    stored_params = st.session_state.get("last_pipeline_params")
    if stored_result is None or "hough_events" not in stored_result:
        st.info("Run Detection Results > Hough Line Detection > Step 2 first, then return here.")
        return

    crop = stored_result["crop"]
    crop_x, crop_y = crop["origin"]
    crop_h, crop_w = crop["display_rgb"].shape[:2]
    cols = st.columns(5)
    cols[0].metric("Crop X", crop_x)
    cols[1].metric("Crop Y", crop_y)
    cols[2].metric("Width", crop_w)
    cols[3].metric("Height", crop_h)
    cols[4].metric("Events", stored_result["hough_events"]["accepted_count"])

    boundary_path = st.text_input(
        "EBSD boundary map",
        str(DEFAULT_EBSD_BOUNDARY_MAP),
        key="auto_boundary_cut_boundary_path",
        help="Black/white image already aligned to the DIC coordinate system.",
    )
    boundary_path = str(Path(boundary_path).expanduser())
    if not Path(boundary_path).exists():
        st.error(f"EBSD boundary map not found: {boundary_path}")
        return
    if st.button("Load EBSD boundary image", key="auto_boundary_cut_load_boundary"):
        boundary_w, boundary_h = image_size(boundary_path)
        st.success(f"Loaded EBSD boundary image: {boundary_w} x {boundary_h} pixels")

    st.markdown("**Cut Controls**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        black_threshold = st.slider(
            "Black boundary threshold",
            0.0,
            1.0,
            0.20,
            0.01,
            key="auto_boundary_cut_black_threshold",
            help="Pixels darker than this are treated as grain-boundary pixels.",
        )
    with c2:
        boundary_dilation_radius = st.slider(
            "Boundary dilation radius",
            0,
            8,
            1,
            1,
            key="auto_boundary_cut_dilation_radius",
            help="Expands boundary pixels before cutting. Higher values cut more aggressively near boundaries.",
        )
    with c3:
        min_segment_pixels = st.slider(
            "Minimum segment pixels",
            1,
            500,
            40,
            1,
            key="auto_boundary_cut_min_segment_pixels",
            help="Drops cut fragments smaller than this many pixels.",
        )
    with c4:
        connectivity = st.radio(
            "Connectivity",
            [1, 2],
            index=1,
            horizontal=True,
            key="auto_boundary_cut_connectivity",
            help="1 is 4-connected. 2 is 8-connected and keeps diagonal event pixels connected.",
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        use_exact_crop = st.checkbox(
            "Display exact detection crop",
            value=True,
            key="auto_boundary_cut_exact_crop",
            help="When enabled, the overlays show the same crop used for detection. Disable to crop tightly around event pixels.",
        )
    with c2:
        use_full_resolution_overlay = st.checkbox(
            "Use full-resolution overlay",
            value=True,
            key="auto_boundary_cut_use_full_resolution_overlay",
            help="Keeps every crop pixel in the zoom viewer.",
        )
    with c3:
        save_prefix = st.text_input(
            "Save prefix",
            "auto_boundary_cut",
            key="auto_boundary_cut_save_prefix",
        )
    with c4:
        save_min_event_pixels = st.slider(
            "Save pixel threshold",
            1,
            5000,
            40,
            10,
            key="auto_boundary_cut_save_min_event_pixels",
            help="Only cut events with at least this many pixels are written to the saved CSVs.",
        )

    if use_full_resolution_overlay:
        display_max_dim = 0
    else:
        display_max_dim = st.slider(
            "Display max dimension",
            800,
            3600,
            2200,
            100,
            key="auto_boundary_cut_display_max_dim",
        )

    run_clicked = st.button("Run auto boundary cut", type="primary", key="run_auto_boundary_cut")
    signature = (
        id(stored_result),
        boundary_path,
        float(black_threshold),
        int(boundary_dilation_radius),
        int(min_segment_pixels),
        int(connectivity),
        bool(use_exact_crop),
        int(display_max_dim),
        bool(use_full_resolution_overlay),
    )

    if run_clicked:
        try:
            with st.spinner("Cutting current Hough events against EBSD boundary crop..."):
                event_pixels, events = hough_result_to_boundary_cut_frames(stored_result, stored_params)
                display_crop = {
                    "x": int(crop_x),
                    "y": int(crop_y),
                    "width": int(crop_w),
                    "height": int(crop_h),
                } if use_exact_crop else None
                result = run_boundary_cut_from_frames(
                    event_pixels,
                    events,
                    boundary_path,
                    float(black_threshold),
                    int(boundary_dilation_radius),
                    int(min_segment_pixels),
                    int(connectivity),
                    display_margin=0,
                    display_max_dim=int(display_max_dim),
                    use_full_resolution_overlay=bool(use_full_resolution_overlay),
                    display_crop_override=display_crop,
                )
        except Exception as exc:
            st.error(f"Auto boundary cut failed: {exc}")
            return
        st.session_state["last_auto_boundary_cut_signature"] = signature
        st.session_state["last_auto_boundary_cut_result"] = result

    result = st.session_state.get("last_auto_boundary_cut_result")
    stored_signature = st.session_state.get("last_auto_boundary_cut_signature")
    if result is None:
        st.info("Press Run auto boundary cut to cut the current Hough-traced events.")
        return
    if stored_signature != signature:
        st.warning("Displayed auto boundary-cut result is from previous inputs or parameters. Press Run auto boundary cut to update.")

    show_boundary_cut_result(result, min_event_pixels=int(save_min_event_pixels))

    if st.button("Save auto cut CSVs", key="save_auto_boundary_cut_csvs"):
        safe_prefix = sanitize_file_prefix(save_prefix) or "auto_boundary_cut"
        saved = save_boundary_cut_outputs(result, safe_prefix, min_event_pixels=int(save_min_event_pixels))
        st.success("Saved boundary-cut outputs:\n\n" + "\n".join(f"- {path}" for path in saved))


def check_boundary_cut_events_tab() -> None:
    st.markdown("**Check Event Sizes On BLN**")
    st.caption(
        "Load saved events, draw them on the BLN image, and highlight small events in purple."
    )

    col1, col2 = st.columns(2)
    with col1:
        event_pixels_file = st.file_uploader(
            "Event pixels CSV",
            type=["csv"],
            key="check_cuts_event_pixels_file",
            help="CSV with event_id, pixel_x, pixel_y columns in global image coordinates.",
        )
        events_file = st.file_uploader(
            "Events CSV",
            type=["csv"],
            key="check_cuts_events_file",
            help="Optional event-level CSV. If it contains image_path, it can help confirm the BLN source.",
        )
    with col2:
        bln_path_input = st.text_input(
            "BLN image path",
            st.session_state.get("check_cuts_bln_loaded_path", str(DEFAULT_IMAGE)),
            key="check_cuts_bln_path_input",
            help=(
                "Local path to the BLN/DIC image used as the background. "
                "Use this instead of upload for large TIFF files."
            ),
        )

    default_clicked = st.button("Use default check-cuts files", key="check_cuts_use_default_files")
    if default_clicked:
        st.session_state["check_cuts_event_pixels_loaded_path"] = str(DEFAULT_FULL_DETECTED_EVENT_PIXELS_CSV)
        st.session_state["check_cuts_events_loaded_path"] = str(DEFAULT_FULL_DETECTED_EVENTS_CSV)
        st.session_state["check_cuts_bln_loaded_path"] = str(DEFAULT_IMAGE)

    if event_pixels_file is not None:
        st.session_state["check_cuts_event_pixels_loaded_path"] = str(
            save_uploaded_streamlit_file(event_pixels_file, "check_cuts")
        )
    if events_file is not None:
        st.session_state["check_cuts_events_loaded_path"] = str(
            save_uploaded_streamlit_file(events_file, "check_cuts")
        )
    if bln_path_input.strip():
        st.session_state["check_cuts_bln_loaded_path"] = str(Path(bln_path_input).expanduser())

    event_pixels_path = st.session_state.get("check_cuts_event_pixels_loaded_path", "")
    events_path = st.session_state.get("check_cuts_events_loaded_path", "")
    bln_path = st.session_state.get("check_cuts_bln_loaded_path", "")

    loaded_labels = []
    if event_pixels_path:
        loaded_labels.append(f"event pixels: `{Path(event_pixels_path).name}`")
    if events_path:
        loaded_labels.append(f"events: `{Path(events_path).name}`")
    if bln_path:
        loaded_labels.append(f"BLN: `{Path(bln_path).name}`")
    if loaded_labels:
        st.caption("Loaded " + " | ".join(loaded_labels))

    threshold = st.slider(
        "Small event pixel threshold",
        1,
        5000,
        100,
        10,
        key="check_cuts_small_event_threshold",
        help="Events with fewer pixels than this value are highlighted in purple.",
    )

    load_clicked = st.button("Load check cuts overlay", type="primary", key="load_check_cuts_overlay")
    signature = (event_pixels_path, events_path, bln_path)

    if load_clicked:
        if not event_pixels_path or not bln_path:
            st.error("Load an event pixels CSV and a BLN image first.")
            return
        missing = [
            label
            for label, path in [
                ("Event pixels CSV", event_pixels_path),
                ("BLN image", bln_path),
            ]
            if not Path(path).exists()
        ]
        if missing:
            st.error("Missing required input: " + ", ".join(missing))
            return
        if events_path.strip() and not Path(events_path).exists():
            st.warning(f"Events CSV not found. Continuing from event pixels only: {events_path}")
        st.session_state["last_check_cuts_signature"] = signature

    stored_signature = st.session_state.get("last_check_cuts_signature")
    if stored_signature != signature:
        st.info("Choose the CSV/image inputs, then press Load check cuts overlay.")
        return

    try:
        with st.spinner("Preparing BLN overlay for event-size check..."):
            result = build_check_cuts_overlay(
                event_pixels_path,
                events_path,
                bln_path,
                int(threshold),
            )
    except Exception as exc:
        st.error(f"Could not build check-cuts overlay: {exc}")
        return

    stats = result["stats"]
    cols = st.columns(5)
    cols[0].metric("Events", f"{stats['event_count']:,}")
    cols[1].metric("Event pixels", f"{stats['pixel_count']:,}")
    cols[2].metric("Small events", f"{stats['small_event_count']:,}")
    cols[3].metric("Small pixels", f"{stats['small_pixel_count']:,}")
    cols[4].metric("Threshold", f"< {stats['threshold']} px")

    st.caption(
        f"BLN image size: {stats['image_width']} x {stats['image_height']} | "
        f"valid plotted pixels: {stats['valid_pixel_count']:,} | "
        "red = all loaded event pixels, purple = events below threshold."
    )

    with st.expander("Small events table", expanded=False):
        st.dataframe(result["small_events"], width="stretch")

    zoomable_image(
        result["overlay"],
        "BLN image with small events highlighted",
        key="check_cuts_bln_overlay",
    )


@st.cache_data(show_spinner=False)
def run_boundary_cut_analysis(
    event_pixels_path: str,
    events_path: str,
    boundary_path: str,
    black_threshold: float,
    boundary_dilation_radius: int,
    min_segment_pixels: int,
    connectivity: int,
    display_margin: int,
    display_max_dim: int,
    use_full_resolution_overlay: bool,
) -> dict:
    event_pixels = pd.read_csv(event_pixels_path)
    if not {"event_id", "pixel_x", "pixel_y"}.issubset(event_pixels.columns):
        raise ValueError("Event pixels CSV must include event_id, pixel_x, and pixel_y columns.")
    events = pd.read_csv(events_path) if events_path.strip() and Path(events_path).exists() else None
    return run_boundary_cut_from_frames(
        event_pixels,
        events,
        boundary_path,
        black_threshold,
        boundary_dilation_radius,
        min_segment_pixels,
        connectivity,
        display_margin,
        display_max_dim,
        use_full_resolution_overlay,
        event_pixels_path=event_pixels_path,
        events_path=events_path,
    )


def run_boundary_cut_from_frames(
    event_pixels: pd.DataFrame,
    events: pd.DataFrame | None,
    boundary_path: str,
    black_threshold: float,
    boundary_dilation_radius: int,
    min_segment_pixels: int,
    connectivity: int,
    display_margin: int,
    display_max_dim: int,
    use_full_resolution_overlay: bool,
    event_pixels_path: str = "",
    events_path: str = "",
    display_crop_override: dict[str, int] | None = None,
) -> dict:
    if not {"event_id", "pixel_x", "pixel_y"}.issubset(event_pixels.columns):
        raise ValueError("Event pixels data must include event_id, pixel_x, and pixel_y columns.")

    bln_path = infer_event_image_path_from_frame(events) or str(DEFAULT_IMAGE)
    if not Path(bln_path).exists():
        raise ValueError(f"BLN image path from events CSV does not exist: {bln_path}")

    event_pixels = event_pixels[["event_id", "pixel_x", "pixel_y"]].copy()
    event_pixels["pixel_x"] = event_pixels["pixel_x"].astype(np.int64)
    event_pixels["pixel_y"] = event_pixels["pixel_y"].astype(np.int64)

    boundary_mask = load_black_boundary_mask(boundary_path, black_threshold)
    if boundary_dilation_radius > 0:
        cut_mask = morphology.binary_dilation(
            boundary_mask,
            morphology.disk(int(boundary_dilation_radius)),
        )
    else:
        cut_mask = boundary_mask

    cut_event_pixels, cut_events, cut_summary = cut_events_by_boundary_mask(
        event_pixels,
        cut_mask,
        min_segment_pixels=int(min_segment_pixels),
        connectivity=int(connectivity),
        boundary_dilation_radius=int(boundary_dilation_radius),
    )

    display_crop = (
        sanitize_crop_rect(display_crop_override, cut_mask.shape[1], cut_mask.shape[0])
        if display_crop_override is not None
        else event_extent_crop(event_pixels, cut_mask.shape, margin=int(display_margin))
    )
    bln_overlay, boundary_overlay = boundary_cut_overlays(
        event_pixels,
        cut_event_pixels,
        bln_path,
        boundary_path,
        cut_mask,
        display_crop,
        max_dim=int(display_max_dim),
        use_full_resolution=bool(use_full_resolution_overlay),
    )

    source_event_count = int(event_pixels["event_id"].nunique())
    event_csv_count = int(events["event_id"].nunique()) if events is not None and "event_id" in events.columns else None
    original_pixel_count = int(len(event_pixels))
    kept_pixel_count = int(len(cut_event_pixels))
    removed_pixel_count = int(cut_summary["removed_boundary_pixels_from_original"].sum()) if "removed_boundary_pixels_from_original" in cut_summary else 0
    split_event_count = int((cut_summary["segments_from_original"] > 1).sum()) if "segments_from_original" in cut_summary else 0
    touched_event_count = int((cut_summary["removed_boundary_pixels_from_original"] > 0).sum()) if "removed_boundary_pixels_from_original" in cut_summary else 0

    return {
        "event_pixels_path": event_pixels_path,
        "events_path": events_path,
        "bln_path": bln_path,
        "boundary_path": boundary_path,
        "black_threshold": float(black_threshold),
        "boundary_dilation_radius": int(boundary_dilation_radius),
        "min_segment_pixels": int(min_segment_pixels),
        "connectivity": int(connectivity),
        "boundary_shape": tuple(int(v) for v in cut_mask.shape),
        "display_crop": display_crop,
        "use_full_resolution_overlay": bool(use_full_resolution_overlay),
        "stats": {
            "events_csv_event_count": event_csv_count,
            "source_event_count": source_event_count,
            "cut_event_count": int(cut_event_pixels["event_id"].nunique()) if len(cut_event_pixels) else 0,
            "source_pixel_count": original_pixel_count,
            "kept_pixel_count": kept_pixel_count,
            "removed_pixel_count": removed_pixel_count,
            "events_touching_boundary": touched_event_count,
            "events_split_into_multiple_segments": split_event_count,
            "discarded_small_segments": int(cut_summary["discarded_small_segments"].sum()) if "discarded_small_segments" in cut_summary else 0,
        },
        "cut_event_pixels": cut_event_pixels,
        "cut_events": cut_events,
        "cut_summary": cut_summary,
        "bln_overlay": bln_overlay,
        "boundary_overlay": boundary_overlay,
    }


def hough_result_to_boundary_cut_frames(
    result: dict,
    params: PipelineParams | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "hough_events" not in result:
        raise ValueError("No Hough events found. Run Hough Step 2 first.")

    crop = result["crop"]
    crop_x, crop_y = crop["origin"]
    image_path = params.image_path if params is not None else str(DEFAULT_IMAGE)
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    pixel_rows = []
    event_rows = []

    for index, line in enumerate(result["hough_events"]["accepted"], start=1):
        event_id = f"auto_hough_{index:04d}"
        points = sorted(line.points, key=lambda point: (point.y, point.x))
        global_points = [(int(crop_x + point.x), int(crop_y + point.y)) for point in points]
        if not global_points:
            continue
        event_rows.append(
            {
                "event_id": event_id,
                "method": "hough",
                "num_pixels": len(global_points),
                "crop_x": int(crop_x),
                "crop_y": int(crop_y),
                "image_path": image_path,
                "created_at": created_at,
            }
        )
        for pixel_x, pixel_y in global_points:
            pixel_rows.append(
                {
                    "event_id": event_id,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                }
            )

    return pd.DataFrame(pixel_rows), pd.DataFrame(event_rows)


def infer_event_image_path(events_path: str) -> str | None:
    path = Path(events_path)
    if not events_path.strip() or not path.exists():
        return None
    try:
        events = pd.read_csv(path, nrows=20)
    except Exception:
        return None
    return infer_event_image_path_from_frame(events)


def infer_event_image_path_from_frame(events: pd.DataFrame | None) -> str | None:
    if events is None or "image_path" not in events.columns:
        return None
    paths = events["image_path"].dropna().astype(str)
    paths = paths[paths.str.strip() != ""]
    if paths.empty:
        return None
    return str(Path(paths.iloc[0]).expanduser())


@st.cache_data(show_spinner=False)
def load_black_boundary_mask(boundary_path: str, black_threshold: float) -> np.ndarray:
    with Image.open(boundary_path) as img:
        arr = np.asarray(img).copy()
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float32)
        gray = rgb.mean(axis=2) / 255.0
    else:
        gray = arr.astype(np.float32)
        if gray.max() > 1:
            gray = gray / 255.0
    return gray < float(black_threshold)


def cut_events_by_boundary_mask(
    event_pixels: pd.DataFrame,
    cut_mask: np.ndarray,
    min_segment_pixels: int,
    connectivity: int,
    boundary_dilation_radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    new_pixel_rows = []
    new_event_rows = []
    summary_rows = []
    height, width = cut_mask.shape

    for original_event_id, group in event_pixels.groupby("event_id", sort=False):
        xs = group["pixel_x"].to_numpy(dtype=np.int64)
        ys = group["pixel_y"].to_numpy(dtype=np.int64)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs = xs[valid]
        ys = ys[valid]
        if len(xs) == 0:
            continue

        pad = int(boundary_dilation_radius) + 3
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, width)
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, height)

        local_event = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        local_event[ys - y0, xs - x0] = True
        local_cut_mask = cut_mask[y0:y1, x0:x1]
        boundary_hit_pixels = local_event & local_cut_mask
        cut_event = local_event & ~local_cut_mask

        labels = measure.label(cut_event, connectivity=connectivity)
        props = measure.regionprops(labels)
        kept_segments = [prop for prop in props if prop.area >= min_segment_pixels]
        discarded_small_segments = len(props) - len(kept_segments)

        for cut_index, segment in enumerate(kept_segments, start=1):
            rr, cc = segment.coords.T
            global_x = cc + x0
            global_y = rr + y0
            new_event_id = f"{original_event_id}_cut_{cut_index:02d}"

            for pixel_x, pixel_y in zip(global_x, global_y):
                new_pixel_rows.append(
                    {
                        "event_id": new_event_id,
                        "original_event_id": original_event_id,
                        "cut_index": cut_index,
                        "pixel_x": int(pixel_x),
                        "pixel_y": int(pixel_y),
                    }
                )

            new_event_rows.append(
                {
                    "event_id": new_event_id,
                    "original_event_id": original_event_id,
                    "cut_index": cut_index,
                    "method": "hough_boundary_cut",
                    "num_pixels": int(len(global_x)),
                    "bbox_min_x": int(global_x.min()),
                    "bbox_max_x": int(global_x.max()),
                    "bbox_min_y": int(global_y.min()),
                    "bbox_max_y": int(global_y.max()),
                    "removed_boundary_pixels_from_original": int(boundary_hit_pixels.sum()),
                    "segments_from_original": int(len(kept_segments)),
                    "discarded_small_segments": int(discarded_small_segments),
                }
            )

        summary_rows.append(
            {
                "original_event_id": original_event_id,
                "original_pixels": int(len(xs)),
                "removed_boundary_pixels_from_original": int(boundary_hit_pixels.sum()),
                "segments_from_original": int(len(kept_segments)),
                "discarded_small_segments": int(discarded_small_segments),
            }
        )

    pixel_columns = ["event_id", "original_event_id", "cut_index", "pixel_x", "pixel_y"]
    event_columns = [
        "event_id",
        "original_event_id",
        "cut_index",
        "method",
        "num_pixels",
        "bbox_min_x",
        "bbox_max_x",
        "bbox_min_y",
        "bbox_max_y",
        "removed_boundary_pixels_from_original",
        "segments_from_original",
        "discarded_small_segments",
    ]
    summary_columns = [
        "original_event_id",
        "original_pixels",
        "removed_boundary_pixels_from_original",
        "segments_from_original",
        "discarded_small_segments",
    ]
    return (
        pd.DataFrame(new_pixel_rows, columns=pixel_columns),
        pd.DataFrame(new_event_rows, columns=event_columns),
        pd.DataFrame(summary_rows, columns=summary_columns),
    )


def event_extent_crop(event_pixels: pd.DataFrame, shape: tuple[int, int], margin: int) -> dict[str, int]:
    height, width = shape
    if event_pixels.empty:
        return {"x": 0, "y": 0, "width": width, "height": height}
    x0 = max(int(event_pixels["pixel_x"].min()) - margin, 0)
    y0 = max(int(event_pixels["pixel_y"].min()) - margin, 0)
    x1 = min(int(event_pixels["pixel_x"].max()) + margin + 1, width)
    y1 = min(int(event_pixels["pixel_y"].max()) + margin + 1, height)
    return {"x": x0, "y": y0, "width": max(1, x1 - x0), "height": max(1, y1 - y0)}


def boundary_cut_overlays(
    original_pixels: pd.DataFrame,
    cut_pixels: pd.DataFrame,
    bln_path: str,
    boundary_path: str,
    cut_mask: np.ndarray,
    crop: dict[str, int],
    max_dim: int,
    use_full_resolution: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x0 = crop["x"]
    y0 = crop["y"]
    w = crop["width"]
    h = crop["height"]
    boundary_crop = cut_mask[y0:y0 + h, x0:x0 + w]
    original_mask = pixels_to_crop_mask(original_pixels, "pixel_x", "pixel_y", x0, y0, (h, w))
    if cut_pixels.empty:
        kept_mask = np.zeros((h, w), dtype=bool)
    else:
        kept_mask = pixels_to_crop_mask(cut_pixels, "pixel_x", "pixel_y", x0, y0, (h, w))
    removed_mask = original_mask & ~kept_mask

    bln_crop = load_crop_by_rect(bln_path, x0, y0, w, h, display_min=0.0, display_max=1.0)
    bln_overlay = bln_crop["display_rgb"].copy()

    boundary_image_crop = load_crop_by_rect(boundary_path, x0, y0, w, h)
    boundary_overlay = boundary_image_crop["display_rgb"].copy()

    bln_overlay[original_mask] = [255, 0, 0]
    bln_overlay[boundary_crop] = [0, 80, 255]
    bln_overlay[removed_mask] = [0, 220, 80]

    boundary_overlay[boundary_crop] = [0, 80, 255]
    boundary_overlay[original_mask] = [255, 0, 0]
    boundary_overlay[removed_mask] = [0, 220, 80]

    if use_full_resolution:
        return bln_overlay, boundary_overlay
    return (
        to_display(bln_overlay, max_dim=max_dim),
        to_display(boundary_overlay, max_dim=max_dim),
    )


@st.cache_data(show_spinner=False)
def build_check_cuts_overlay(
    event_pixels_path: str,
    events_path: str,
    bln_path: str,
    threshold: int,
) -> dict:
    event_pixels = pd.read_csv(event_pixels_path)
    required = {"event_id", "pixel_x", "pixel_y"}
    if not required.issubset(event_pixels.columns):
        raise ValueError("Event pixels CSV must include event_id, pixel_x, and pixel_y columns.")

    event_pixels = event_pixels[["event_id", "pixel_x", "pixel_y"]].copy()
    event_pixels["event_id"] = event_pixels["event_id"].astype(str)
    event_pixels["pixel_x"] = event_pixels["pixel_x"].astype(np.int64)
    event_pixels["pixel_y"] = event_pixels["pixel_y"].astype(np.int64)

    image_width, image_height = image_size(bln_path)
    bln_crop = load_crop_by_rect(
        bln_path,
        0,
        0,
        image_width,
        image_height,
        display_min=0.0,
        display_max=1.0,
    )
    overlay = bln_crop["display_rgb"].copy()

    event_sizes = (
        event_pixels.groupby("event_id", sort=False)
        .size()
        .reset_index(name="num_pixels")
    )
    small_events = event_sizes[event_sizes["num_pixels"] < int(threshold)].copy()
    small_ids = set(small_events["event_id"].astype(str))
    small_event_count = int(len(small_events))

    xs = event_pixels["pixel_x"].to_numpy(dtype=np.int64)
    ys = event_pixels["pixel_y"].to_numpy(dtype=np.int64)
    valid = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)
    valid_pixels = event_pixels.loc[valid].copy()

    all_mask = pixels_to_crop_mask(valid_pixels, "pixel_x", "pixel_y", 0, 0, (image_height, image_width))
    small_pixels = valid_pixels[valid_pixels["event_id"].isin(small_ids)]
    small_mask = pixels_to_crop_mask(small_pixels, "pixel_x", "pixel_y", 0, 0, (image_height, image_width))

    overlay[all_mask] = [255, 0, 0]
    overlay[small_mask] = [190, 0, 255]

    if events_path.strip() and Path(events_path).exists():
        events = pd.read_csv(events_path)
        if "event_id" in events.columns:
            events = events.copy()
            events["event_id"] = events["event_id"].astype(str)
            small_events = small_events.merge(events, on="event_id", how="left", suffixes=("", "_events_csv"))

    return {
        "overlay": overlay,
        "small_events": small_events,
        "stats": {
            "threshold": int(threshold),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "event_count": int(event_sizes["event_id"].nunique()),
            "pixel_count": int(len(event_pixels)),
            "valid_pixel_count": int(len(valid_pixels)),
            "small_event_count": small_event_count,
            "small_pixel_count": int(len(small_pixels)),
        },
    }


def show_boundary_cut_result(result: dict, min_event_pixels: int = 1) -> None:
    stats = result["stats"]
    kept_for_save, rejected_for_save = boundary_cut_save_threshold_counts(result, int(min_event_pixels))
    cols = st.columns(6)
    cols[0].metric("Source events", stats["source_event_count"])
    cols[1].metric("Cut events", stats["cut_event_count"])
    cols[2].metric("Source pixels", f"{stats['source_pixel_count']:,}")
    cols[3].metric("Kept pixels", f"{stats['kept_pixel_count']:,}")
    cols[4].metric("Removed pixels", f"{stats['removed_pixel_count']:,}")
    cols[5].metric("Split events", stats["events_split_into_multiple_segments"])

    crop = result["display_crop"]
    st.caption(
        f"Boundary map size: {result['boundary_shape'][1]} x {result['boundary_shape'][0]} | "
        f"display crop x={crop['x']}, y={crop['y']}, width={crop['width']}, height={crop['height']} | "
        f"overlay mode: {'full resolution' if result.get('use_full_resolution_overlay') else 'downsampled'} | "
        f"BLN background: `{result.get('bln_path', '')}` | "
        f"save threshold: >= {int(min_event_pixels)} pixels "
        f"({kept_for_save} pass, {rejected_for_save} fail) | "
        "red = original events, blue = EBSD boundaries, green = removed cut pixels, purple = cut events below save threshold."
    )

    with st.expander("Boundary-cut summary table", expanded=False):
        st.dataframe(result["cut_summary"], width="stretch")

    zoomable_image(
        boundary_cut_overlay_with_save_threshold(result, int(min_event_pixels)),
        "BLN image with events, EBSD boundaries, and cut pixels",
        key="boundary_cut_bln_overlay",
    )


def save_boundary_cut_outputs(result: dict, prefix: str, min_event_pixels: int = 1) -> list[Path]:
    out_dir = ROOT / "tests" / "outputs" / "hough_events"
    out_dir.mkdir(parents=True, exist_ok=True)
    pixels_path = out_dir / f"{prefix}_event_pixels.csv"
    events_path = out_dir / f"{prefix}_events.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"

    cut_events, cut_event_pixels = filter_cut_events_for_save(result, int(min_event_pixels))
    cut_event_pixels.to_csv(pixels_path, index=False)
    cut_events.to_csv(events_path, index=False)
    result["cut_summary"].to_csv(summary_path, index=False)
    return [events_path, pixels_path, summary_path]


def filter_cut_events_for_save(result: dict, min_event_pixels: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut_events = result["cut_events"].copy()
    cut_event_pixels = result["cut_event_pixels"].copy()
    if cut_events.empty or cut_event_pixels.empty:
        return cut_events, cut_event_pixels

    min_event_pixels = max(1, int(min_event_pixels))
    if "num_pixels" not in cut_events.columns:
        pixel_counts = (
            cut_event_pixels.groupby("event_id", sort=False)
            .size()
            .reset_index(name="num_pixels")
        )
        cut_events = cut_events.merge(pixel_counts, on="event_id", how="left")

    keep_ids = set(
        cut_events.loc[
            cut_events["num_pixels"].fillna(0).astype(int) >= min_event_pixels,
            "event_id",
        ].astype(str)
    )
    cut_events = cut_events[cut_events["event_id"].astype(str).isin(keep_ids)].copy()
    cut_event_pixels = cut_event_pixels[cut_event_pixels["event_id"].astype(str).isin(keep_ids)].copy()
    return cut_events, cut_event_pixels


def boundary_cut_save_threshold_counts(result: dict, min_event_pixels: int) -> tuple[int, int]:
    cut_events = result["cut_events"]
    if cut_events.empty:
        return 0, 0
    if "num_pixels" in cut_events.columns:
        sizes = cut_events["num_pixels"].fillna(0).astype(int)
    else:
        sizes = result["cut_event_pixels"].groupby("event_id").size()
    kept = int((sizes >= max(1, int(min_event_pixels))).sum())
    failed = int(len(sizes) - kept)
    return kept, failed


def boundary_cut_overlay_with_save_threshold(result: dict, min_event_pixels: int) -> np.ndarray:
    overlay = result["bln_overlay"].copy()
    cut_events = result["cut_events"]
    cut_event_pixels = result["cut_event_pixels"]
    if cut_events.empty or cut_event_pixels.empty:
        return overlay

    min_event_pixels = max(1, int(min_event_pixels))
    if "num_pixels" in cut_events.columns:
        rejected_ids = set(
            cut_events.loc[
                cut_events["num_pixels"].fillna(0).astype(int) < min_event_pixels,
                "event_id",
            ].astype(str)
        )
    else:
        sizes = cut_event_pixels.groupby("event_id").size()
        rejected_ids = set(sizes[sizes < min_event_pixels].index.astype(str))
    if not rejected_ids:
        return overlay

    crop = result["display_crop"]
    h, w = overlay.shape[:2]
    rejected_pixels = cut_event_pixels[cut_event_pixels["event_id"].astype(str).isin(rejected_ids)]
    rejected_mask = pixels_to_crop_mask(
        rejected_pixels,
        "pixel_x",
        "pixel_y",
        int(crop["x"]),
        int(crop["y"]),
        (h, w),
    )
    overlay[rejected_mask] = [190, 0, 255]
    return overlay


@st.cache_data(show_spinner=False)
def run_detection_analysis(
    bln_path: str,
    grx_path: str,
    slip_mask_path: str,
    detected_pixels_path: str,
    bln_display_min: float,
    bln_display_max: float,
) -> dict:
    detected_pixels = pd.read_csv(detected_pixels_path)
    slip_pixels = pd.read_csv(slip_mask_path)

    required_pixel_cols = {"event_id", "pixel_x", "pixel_y"}
    required_slip_cols = {"SlipID", "MaskX", "MaskY"}
    if not required_pixel_cols.issubset(detected_pixels.columns):
        raise ValueError(f"Detected event pixels CSV must include columns: {sorted(required_pixel_cols)}")
    if not required_slip_cols.issubset(slip_pixels.columns):
        raise ValueError(f"Slip mask CSV must include columns: {sorted(required_slip_cols)}")

    detected_pixels["pixel_x"] = detected_pixels["pixel_x"].astype(int)
    detected_pixels["pixel_y"] = detected_pixels["pixel_y"].astype(int)
    slip_pixels["MaskX"] = slip_pixels["MaskX"].astype(int)
    slip_pixels["MaskY"] = slip_pixels["MaskY"].astype(int)

    grx_w, grx_h = image_size(grx_path)
    x_min = max(0, int(detected_pixels["pixel_x"].min()))
    y_min = max(0, int(detected_pixels["pixel_y"].min()))
    x_max = min(grx_w - 1, int(detected_pixels["pixel_x"].max()))
    y_max = min(grx_h - 1, int(detected_pixels["pixel_y"].max()))
    crop_width = x_max - x_min + 1
    crop_height = y_max - y_min + 1
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("Detected crop has non-positive width or height.")

    grx_crop = load_display_crop_at(grx_path, x_min, y_min, crop_width, crop_height)
    bln_crop = load_crop_by_rect(
        bln_path,
        x_min,
        y_min,
        crop_width,
        crop_height,
        bln_display_min,
        bln_display_max,
    )
    if grx_crop is None or bln_crop is None:
        raise ValueError("Could not load BLN/GRX crop.")

    grx_rgb = grx_crop["display_rgb"]
    bln_rgb = bln_crop["display_rgb"]
    crop_shape = grx_rgb.shape[:2]

    detected_mask = pixels_to_crop_mask(
        detected_pixels,
        x_col="pixel_x",
        y_col="pixel_y",
        crop_x=x_min,
        crop_y=y_min,
        crop_shape=crop_shape,
    )
    slip_crop_pixels = slip_pixels[
        (slip_pixels["MaskX"] >= x_min)
        & (slip_pixels["MaskX"] <= x_max)
        & (slip_pixels["MaskY"] >= y_min)
        & (slip_pixels["MaskY"] <= y_max)
    ].copy()
    slip_mask = pixels_to_crop_mask(
        slip_crop_pixels,
        x_col="MaskX",
        y_col="MaskY",
        crop_x=x_min,
        crop_y=y_min,
        crop_shape=crop_shape,
    )

    stats = detection_overlap_stats(detected_mask, slip_mask)
    stats.update(
        {
            "detected_pixel_rows": int(len(detected_pixels)),
            "detected_event_ids": int(detected_pixels["event_id"].nunique()),
            "slip_rows_inside_crop": int(len(slip_crop_pixels)),
            "slip_ids_inside_crop": int(slip_crop_pixels["SlipID"].nunique()) if len(slip_crop_pixels) else 0,
        }
    )

    return {
        "crop": {
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max),
            "width": int(crop_width),
            "height": int(crop_height),
        },
        "stats": stats,
        "grx_quality_overlay": draw_detection_quality_overlay(grx_rgb, detected_mask, slip_mask),
        "bln_quality_overlay": draw_detection_quality_overlay(bln_rgb, detected_mask, slip_mask),
    }


def run_category_detection_dashboard(
    ground_truth_file,
    category_files: dict[str, object | None],
    bln_path: str,
    bln_display_min: float,
    bln_display_max: float,
    overlay_max_dim: int,
) -> dict:
    ground_truth = read_uploaded_or_path_csv(ground_truth_file)
    ground_truth_points = slip_ground_truth_points(ground_truth)
    rows = []
    category_points: dict[str, set[tuple[int, int]]] = {}

    for category in ["matched", "blob", "uncertain", "noise"]:
        uploaded = category_files.get(category)
        if uploaded is None:
            points: set[tuple[int, int]] = set()
            event_count = 0
            row_count = 0
            events_with_overlap = 0
        else:
            frame = read_uploaded_or_path_csv(uploaded)
            points = event_pixel_points(frame)
            row_count = int(len(frame))
            event_count = int(frame["event_id"].nunique()) if "event_id" in frame.columns else 0
            events_with_overlap = count_events_with_ground_truth_overlap(frame, ground_truth_points)
        category_points[category] = points
        stats = point_set_overlap_stats(points, ground_truth_points)
        rows.append(
            {
                "category": category,
                "event_count": event_count,
                "events_with_gt_overlap": events_with_overlap,
                "pixel_rows": row_count,
                **stats,
            }
        )

    matched = category_points["matched"]
    nonmatched_union = (
        category_points["blob"]
        | category_points["uncertain"]
        | category_points["noise"]
    )
    all_uploaded = matched | nonmatched_union
    matched_missed = ground_truth_points - matched
    overview = {
        "ground_truth_pixels": len(ground_truth_points),
        "matched_true_positive": len(matched & ground_truth_points),
        "matched_false_positive": len(matched - ground_truth_points),
        "matched_false_negative": len(ground_truth_points - matched),
        "ground_truth_in_nonmatched_categories": len(ground_truth_points & nonmatched_union),
        "matched_missed_found_in_blob": len(matched_missed & category_points["blob"]),
        "matched_missed_found_in_uncertain": len(matched_missed & category_points["uncertain"]),
        "matched_missed_found_in_noise": len(matched_missed & category_points["noise"]),
        "ground_truth_missed_by_all_uploaded_categories": len(ground_truth_points - all_uploaded),
    }
    overlay, overlay_crop = build_category_quality_overlay(
        ground_truth_points,
        matched,
        bln_path=str(bln_path),
        display_min=float(bln_display_min),
        display_max=float(bln_display_max),
        max_dim=int(overlay_max_dim),
    )
    return {
        "overview": overview,
        "metrics": pd.DataFrame(rows),
        "overlay": overlay,
        "overlay_crop": overlay_crop,
    }


def run_misorientation_analysis(
    trace_df: pd.DataFrame,
    threshold_values: list[float],
    bin_width: float,
) -> dict:
    required = {"event_angle_deg", "best_trace_angle_deg", "best_angle_error_deg", "best_mode"}
    if not required.issubset(trace_df.columns):
        raise ValueError(f"Trace comparison CSV must include columns: {sorted(required)}")

    df = trace_df.copy()
    for col in ["event_angle_deg", "best_trace_angle_deg", "best_angle_error_deg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["event_angle_deg", "best_trace_angle_deg", "best_angle_error_deg"]).copy()
    if df.empty:
        raise ValueError("No finite angle rows found in the trace comparison CSV.")

    df["signed_misorientation_deg"] = signed_line_angle_difference(
        df["event_angle_deg"],
        df["best_trace_angle_deg"],
    )
    thresholds = [float(value) for value in threshold_values]
    summary_rows = []
    histograms = {}
    for threshold in thresholds:
        subset = df[df["best_angle_error_deg"] <= threshold].copy()
        signed = subset["signed_misorientation_deg"].dropna()
        absolute = subset["best_angle_error_deg"].dropna()
        mode_counts = subset["best_mode"].astype(str).value_counts().to_dict() if len(subset) else {}
        summary_rows.append(
            {
                "threshold_deg": threshold,
                "matched_events": int(len(subset)),
                "clockwise_negative": int((signed < 0).sum()),
                "counterclockwise_positive": int((signed > 0).sum()),
                "signed_mean_deg": float(signed.mean()) if len(signed) else np.nan,
                "signed_median_deg": float(signed.median()) if len(signed) else np.nan,
                "signed_std_deg": float(signed.std(ddof=1)) if len(signed) > 1 else np.nan,
                "abs_mean_deg": float(absolute.mean()) if len(absolute) else np.nan,
                "abs_median_deg": float(absolute.median()) if len(absolute) else np.nan,
                "abs_std_deg": float(absolute.std(ddof=1)) if len(absolute) > 1 else np.nan,
                "prism1": int(mode_counts.get("prism1", 0)),
                "prism2": int(mode_counts.get("prism2", 0)),
                "prism3": int(mode_counts.get("prism3", 0)),
                "basal": int(mode_counts.get("basal", 0)),
            }
        )
        histograms[threshold] = {
            "signed": histogram_frame(signed, -threshold, threshold, bin_width),
            "absolute": histogram_frame(absolute, 0.0, threshold, bin_width),
        }
    all_signed = df["signed_misorientation_deg"].dropna()
    all_absolute = df["best_angle_error_deg"].dropna()
    all_abs_max = max(float(all_absolute.max()) if len(all_absolute) else 1.0, float(bin_width))
    histograms["all"] = {
        "signed": histogram_frame(all_signed, -90.0, 90.0, bin_width),
        "absolute": histogram_frame(all_absolute, 0.0, all_abs_max, bin_width),
    }
    if "classification" in df.columns:
        noise = df[df["classification"].astype(str) == "likely_noise"].copy()
    else:
        noise = df.iloc[0:0].copy()
    noise_signed = noise["signed_misorientation_deg"].dropna()
    noise_absolute = noise["best_angle_error_deg"].dropna()
    noise_abs_max = max(float(noise_absolute.max()) if len(noise_absolute) else 1.0, float(bin_width))
    histograms["noise"] = {
        "signed": histogram_frame(noise_signed, -90.0, 90.0, bin_width),
        "absolute": histogram_frame(noise_absolute, 0.0, noise_abs_max, bin_width),
    }

    return {
        "thresholds": thresholds,
        "summary": pd.DataFrame(summary_rows),
        "histograms": histograms,
    }


def signed_line_angle_difference(event_angle_deg: pd.Series, trace_angle_deg: pd.Series) -> pd.Series:
    return ((event_angle_deg.astype(float) - trace_angle_deg.astype(float) + 90.0) % 180.0) - 90.0


def histogram_frame(values: pd.Series, low: float, high: float, bin_width: float) -> pd.DataFrame:
    values = pd.Series(values, dtype=float).dropna()
    if high <= low:
        high = low + float(bin_width)
    edges = np.arange(float(low), float(high) + float(bin_width) * 1.001, float(bin_width))
    if len(edges) < 2:
        edges = np.array([float(low), float(high)], dtype=float)
    counts, edges = np.histogram(values.to_numpy(dtype=float), bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    smooth_counts = ndi.gaussian_filter1d(counts.astype(np.float64), sigma=1.0) if len(counts) else counts
    return pd.DataFrame(
        {
            "bin_center_deg": centers,
            "bin_start_deg": edges[:-1],
            "bin_end_deg": edges[1:],
            "count": counts,
            "smooth_count": smooth_counts,
        }
    ).assign(quartile=histogram_quartile_labels(centers, values))


def histogram_quartile_labels(bin_centers: np.ndarray, values: pd.Series) -> list[str]:
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return ["Q1"] * len(bin_centers)
    q1, q2, q3 = values.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
    labels = []
    for center in bin_centers:
        if center <= q1:
            labels.append("Q1")
        elif center <= q2:
            labels.append("Q2")
        elif center <= q3:
            labels.append("Q3")
        else:
            labels.append("Q4")
    return labels


def colored_histogram_chart(frame: pd.DataFrame, x_title: str) -> alt.Chart:
    zoom = alt.selection_interval(bind="scales", encodings=["x", "y"])
    base = alt.Chart(frame)
    bars = (
        base
        .mark_bar()
        .encode(
            x=alt.X("bin_start_deg:Q", title=x_title),
            x2="bin_end_deg:Q",
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color(
                "quartile:N",
                title="Quartile",
                scale=alt.Scale(
                    domain=["Q1", "Q2", "Q3", "Q4"],
                    range=["#2b6cb0", "#38a169", "#dd6b20", "#805ad5"],
                ),
            ),
            tooltip=[
                alt.Tooltip("bin_start_deg:Q", title="Bin start", format=".2f"),
                alt.Tooltip("bin_end_deg:Q", title="Bin end", format=".2f"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("quartile:N", title="Quartile"),
            ],
        )
    )
    curve = (
        base
        .mark_line(color="#111827", strokeWidth=3)
        .encode(
            x=alt.X("bin_center_deg:Q", title=x_title),
            y=alt.Y("smooth_count:Q", title="Count"),
            tooltip=[
                alt.Tooltip("bin_center_deg:Q", title="Bin center", format=".2f"),
                alt.Tooltip("smooth_count:Q", title="Smoothed count", format=".2f"),
            ],
        )
    )
    return (bars + curve).properties(height=280).add_params(zoom)


def category_count_chart(metrics: pd.DataFrame) -> alt.Chart:
    count_cols = [
        "overlap_true_positive",
        "false_positive",
        "false_negative",
    ]
    available = [col for col in count_cols if col in metrics.columns]
    long = metrics.melt(
        id_vars=["category"],
        value_vars=available,
        var_name="metric",
        value_name="pixels",
    )
    labels = {
        "overlap_true_positive": "TP overlap",
        "false_positive": "FP extra",
        "false_negative": "FN missed",
    }
    long["metric"] = long["metric"].map(labels).fillna(long["metric"])
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("category:N", title="Category"),
            y=alt.Y("pixels:Q", title="Pixels"),
            color=alt.Color(
                "metric:N",
                title="Metric",
                scale=alt.Scale(
                    domain=["TP overlap", "FP extra", "FN missed"],
                    range=["#38a169", "#2b6cb0", "#e53e3e"],
                ),
            ),
            xOffset="metric:N",
            tooltip=["category:N", "metric:N", alt.Tooltip("pixels:Q", format=",")],
        )
        .properties(title="Pixel overlap counts", height=320)
    )


def category_rate_chart(metrics: pd.DataFrame) -> alt.Chart:
    rate_cols = [
        "precision_tp_over_detected",
        "recall_tp_over_ground_truth",
        "f1_score",
        "iou_jaccard",
    ]
    available = [col for col in rate_cols if col in metrics.columns]
    long = metrics.melt(
        id_vars=["category"],
        value_vars=available,
        var_name="metric",
        value_name="score",
    )
    labels = {
        "precision_tp_over_detected": "Precision",
        "recall_tp_over_ground_truth": "Recall",
        "f1_score": "F1",
        "iou_jaccard": "IoU",
    }
    long["metric"] = long["metric"].map(labels).fillna(long["metric"])
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("category:N", title="Category"),
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric:N", title="Metric"),
            xOffset="metric:N",
            tooltip=["category:N", "metric:N", alt.Tooltip("score:Q", format=".4f")],
        )
        .properties(title="Category quality rates", height=320)
    )


def build_category_quality_overlay(
    ground_truth_points: set[tuple[int, int]],
    matched_points: set[tuple[int, int]],
    bln_path: str,
    display_min: float,
    display_max: float,
    max_dim: int,
    margin: int = 20,
) -> tuple[np.ndarray, dict]:
    all_points = ground_truth_points | matched_points
    if not all_points:
        return np.ones((1, 1, 3), dtype=np.uint8) * 255, {
            "x_min": 0,
            "y_min": 0,
            "x_max": 0,
            "y_max": 0,
            "width": 1,
            "height": 1,
            "render_step": 1,
        }

    xs = np.array([point[0] for point in all_points], dtype=np.int64)
    ys = np.array([point[1] for point in all_points], dtype=np.int64)
    x_min = int(xs.min()) - int(margin)
    y_min = int(ys.min()) - int(margin)
    x_max = int(xs.max()) + int(margin)
    y_max = int(ys.max()) + int(margin)
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    render_step = max(1, int(np.ceil(max(width, height) / max(1, int(max_dim)))))
    overlay = load_scaled_display_crop(
        bln_path,
        x_min,
        y_min,
        width,
        height,
        render_step=render_step,
        display_min=display_min,
        display_max=display_max,
    )

    false_positive = matched_points - ground_truth_points
    true_positive = matched_points & ground_truth_points
    paint_point_set_on_overlay(overlay, ground_truth_points, x_min, y_min, render_step, (255, 0, 0))
    paint_point_set_on_overlay(overlay, false_positive, x_min, y_min, render_step, (0, 64, 255))
    paint_point_set_on_overlay(overlay, true_positive, x_min, y_min, render_step, (0, 220, 0))

    crop = {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "width": width,
        "height": height,
        "render_step": render_step,
    }
    return overlay, crop


def paint_point_set_on_overlay(
    overlay: np.ndarray,
    points: set[tuple[int, int]],
    crop_x: int,
    crop_y: int,
    render_step: int,
    color: tuple[int, int, int],
) -> None:
    if not points:
        return
    coords = np.array(list(points), dtype=np.int64)
    xs = ((coords[:, 0] - int(crop_x)) // int(render_step)).astype(np.int64)
    ys = ((coords[:, 1] - int(crop_y)) // int(render_step)).astype(np.int64)
    valid = (xs >= 0) & (xs < overlay.shape[1]) & (ys >= 0) & (ys < overlay.shape[0])
    overlay[ys[valid], xs[valid]] = np.array(color, dtype=np.uint8)


def read_uploaded_or_path_csv(source) -> pd.DataFrame:
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source)


def slip_ground_truth_points(frame: pd.DataFrame) -> set[tuple[int, int]]:
    required = {"MaskX", "MaskY"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Ground-truth slip CSV must include columns: {sorted(required)}")
    xs = frame["MaskX"].to_numpy(dtype=np.int64)
    ys = frame["MaskY"].to_numpy(dtype=np.int64)
    return set(zip(xs.tolist(), ys.tolist(), strict=False))


def event_pixel_points(frame: pd.DataFrame) -> set[tuple[int, int]]:
    required = {"event_id", "pixel_x", "pixel_y"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Event pixels CSV must include columns: {sorted(required)}")
    xs = frame["pixel_x"].to_numpy(dtype=np.int64)
    ys = frame["pixel_y"].to_numpy(dtype=np.int64)
    return set(zip(xs.tolist(), ys.tolist(), strict=False))


def count_events_with_ground_truth_overlap(frame: pd.DataFrame, ground_truth_points: set[tuple[int, int]]) -> int:
    required = {"event_id", "pixel_x", "pixel_y"}
    if not required.issubset(frame.columns) or not ground_truth_points:
        return 0
    temp = frame.loc[:, ["event_id", "pixel_x", "pixel_y"]].copy()
    temp["point"] = list(
        zip(
            temp["pixel_x"].astype(np.int64).tolist(),
            temp["pixel_y"].astype(np.int64).tolist(),
            strict=False,
        )
    )
    return int(temp.loc[temp["point"].isin(ground_truth_points), "event_id"].astype(str).nunique())


def point_set_overlap_stats(
    detected_points: set[tuple[int, int]],
    ground_truth_points: set[tuple[int, int]],
) -> dict:
    tp = len(detected_points & ground_truth_points)
    fp = len(detected_points - ground_truth_points)
    fn = len(ground_truth_points - detected_points)
    detected_count = len(detected_points)
    ground_truth_count = len(ground_truth_points)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision + recall) and (precision + recall)
        else float("nan")
    )
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    return {
        "detected_unique_pixels": detected_count,
        "ground_truth_pixels": ground_truth_count,
        "overlap_true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision_tp_over_detected": precision,
        "recall_tp_over_ground_truth": recall,
        "f1_score": f1,
        "iou_jaccard": iou,
    }


def pixels_to_crop_mask(
    pixels: pd.DataFrame,
    x_col: str,
    y_col: str,
    crop_x: int,
    crop_y: int,
    crop_shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(crop_shape, dtype=bool)
    if pixels.empty:
        return mask
    xs = pixels[x_col].to_numpy(dtype=np.int64) - crop_x
    ys = pixels[y_col].to_numpy(dtype=np.int64) - crop_y
    valid = (xs >= 0) & (xs < crop_shape[1]) & (ys >= 0) & (ys < crop_shape[0])
    mask[ys[valid], xs[valid]] = True
    return mask


def detection_overlap_stats(detected_mask: np.ndarray, ground_truth_mask: np.ndarray) -> dict:
    detected_mask = detected_mask.astype(bool, copy=False)
    ground_truth_mask = ground_truth_mask.astype(bool, copy=False)
    true_positive = detected_mask & ground_truth_mask
    false_negative = ground_truth_mask & ~detected_mask
    false_positive = detected_mask & ~ground_truth_mask
    true_negative = ~(detected_mask | ground_truth_mask)

    tp = int(true_positive.sum())
    fn = int(false_negative.sum())
    fp = int(false_positive.sum())
    tn = int(true_negative.sum())
    detected_pixels = int(detected_mask.sum())
    ground_truth_pixels = int(ground_truth_mask.sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision + recall) and (precision + recall)
        else float("nan")
    )
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")

    return {
        "crop_pixels_total": int(detected_mask.size),
        "automatic_detected_pixels": detected_pixels,
        "ground_truth_pixels": ground_truth_pixels,
        "overlap_true_positive": tp,
        "missed_ground_truth_false_negative": fn,
        "automatic_false_positive": fp,
        "true_negative": tn,
        "precision_tp_over_detected": precision,
        "recall_tp_over_ground_truth": recall,
        "f1_score": f1,
        "iou_jaccard": iou,
    }


def draw_detection_quality_overlay(
    rgb: np.ndarray,
    detected_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> np.ndarray:
    out = rgb.copy()
    tp_mask = detected_mask & ground_truth_mask
    fp_mask = detected_mask & ~ground_truth_mask
    fn_mask = ground_truth_mask & ~detected_mask
    out[tp_mask] = [0, 255, 0]
    out[fp_mask] = [255, 0, 0]
    out[fn_mask] = [0, 64, 255]
    return out


def draw_mask_overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.copy()
    out[mask.astype(bool, copy=False)] = color
    return out


def sidebar_params() -> PipelineParams:
    image_path = st.sidebar.text_input("BLN detection image path", str(DEFAULT_IMAGE))
    grx_overlay_path = st.sidebar.text_input("GRX overlay image path", str(DEFAULT_GRX_IMAGE))
    image_path = str(Path(image_path).expanduser())
    grx_overlay_path = str(Path(grx_overlay_path).expanduser()) if grx_overlay_path.strip() else ""
    if not Path(image_path).exists():
        st.sidebar.error(f"Detection image not found: {image_path}")
        st.stop()
    if grx_overlay_path and not Path(grx_overlay_path).exists():
        st.sidebar.warning(f"GRX overlay image not found: {grx_overlay_path}")
        grx_overlay_path = ""
    img_w, img_h = image_size(image_path)
    default_crop_size = min(500, img_w, img_h)

    return PipelineParams(
        image_path=image_path,
        grx_overlay_path=grx_overlay_path,
        display_min=0.0,
        display_max=1.0,
        use_full_image=False,
        crop_x=max(0, (img_w - default_crop_size) // 2),
        crop_y=max(0, (img_h - default_crop_size) // 2),
        crop_width=default_crop_size,
        crop_height=default_crop_size,
        clahe_clip_limit=0.100,
        ridge_sigma_max=4,
        ridge_percentile=88.0,
        threshold_multiplier=0.40,
        min_object_size=231,
        closing_radius=3,
        use_skeletonize=True,
        hough_threshold=20,
        hough_line_length=30,
        hough_line_gap=10,
        hough_seed_spacing=20,
        hough_max_seeds=200,
        hough_use_all_seeds=True,
        region_min_pixels=20,
        region_seed_spacing=20,
        region_max_seeds=200,
        region_use_all_seeds=True,
        intensity_difference_tolerance=75,
        best_fit_line_tolerance=7.0,
        min_intensity=120,
        min_points_threshold=40,
        duplicate_overlap=0.5,
        merge_distance_tolerance=5.0,
        merge_angle_tolerance=12.0,
        connect_merged_event_gaps=False,
    )


def params_match_for_result(stored_params: PipelineParams | None, params: PipelineParams) -> bool:
    if stored_params == params:
        return True
    if stored_params is None:
        return False
    comparison_params = stored_params
    try:
        if getattr(stored_params, "hough_use_all_seeds", False) and params.hough_use_all_seeds:
            comparison_params = replace(comparison_params, hough_max_seeds=params.hough_max_seeds)
        if getattr(stored_params, "region_use_all_seeds", False) and params.region_use_all_seeds:
            comparison_params = replace(comparison_params, region_max_seeds=params.region_max_seeds)
    except TypeError:
        return False
    return comparison_params == params


def processing_region_controls(params: PipelineParams) -> PipelineParams:
    img_w, img_h = image_size(params.image_path)
    st.subheader("Processing Region")
    st.caption(f"Detection image size: {img_w} x {img_h} pixels")
    display_min, display_max = brightness_contrast_controls(params.image_path)
    params = replace(params, display_min=display_min, display_max=display_max)

    mode = st.radio(
        "Region mode",
        ["Crop", "Full image"],
        horizontal=True,
        index=1 if params.use_full_image else 0,
    )
    use_full_image = mode == "Full image"

    if use_full_image:
        st.warning(
            "Full-image processing can be slow and memory-heavy. Use Run / refresh detection "
            "from the Detection Results tab when you are ready."
        )
        st.metric("Selected processing size", f"{img_w} x {img_h}")
        return replace(
            params,
            use_full_image=True,
            crop_x=0,
            crop_y=0,
            crop_width=img_w,
            crop_height=img_h,
        )

    default_crop = sanitize_crop_rect(
        {
            "x": params.crop_x,
            "y": params.crop_y,
            "width": params.crop_width,
            "height": params.crop_height,
        },
        img_w,
        img_h,
    )
    state_key = f"selected_crop:{params.image_path}"
    current_crop = sanitize_crop_rect(st.session_state.get(state_key, default_crop), img_w, img_h)
    preview = load_region_selector_preview(
        params.image_path,
        0,
        0,
        img_w,
        img_h,
        display_min,
        display_max,
    )
    st.caption(
        f"Crop selector uses the full BLN image downsampled to "
        f"{preview['preview_width']} x {preview['preview_height']}."
    )
    selected = CROP_SELECTOR_COMPONENT(
        image_data_url=preview["image_data_url"],
        preview_width=preview["preview_width"],
        preview_height=preview["preview_height"],
        original_width=preview["original_width"],
        original_height=preview["original_height"],
        view_x=preview["view_x"],
        view_y=preview["view_y"],
        view_width=preview["view_width"],
        view_height=preview["view_height"],
        initial_crop=current_crop,
        default_crop_width=default_crop["width"],
        default_crop_height=default_crop["height"],
        min_crop_size=20,
        key=f"crop_selector_{Path(params.image_path).stem}_full_overview",
        default=current_crop,
    )
    crop_rect = sanitize_crop_rect(selected or current_crop, img_w, img_h)
    st.session_state[state_key] = crop_rect

    cols = st.columns(4)
    cols[0].metric("Left X", crop_rect["x"])
    cols[1].metric("Top Y", crop_rect["y"])
    cols[2].metric("Width", crop_rect["width"])
    cols[3].metric("Height", crop_rect["height"])

    crop = load_crop_by_rect(
        params.image_path,
        crop_rect["x"],
        crop_rect["y"],
        crop_rect["width"],
        crop_rect["height"],
        display_min,
        display_max,
    )
    st.caption(
        f"Selected crop: x={crop['origin'][0]}, y={crop['origin'][1]}, "
        f"width={crop['display_rgb'].shape[1]}, height={crop['display_rgb'].shape[0]}"
    )
    st.image(
        to_display(crop["display_rgb"]),
        caption="Cropped BLN region with current brightness/contrast range",
        width='stretch',
    )

    return replace(
        params,
        use_full_image=False,
        crop_x=crop_rect["x"],
        crop_y=crop_rect["y"],
        crop_width=crop_rect["width"],
        crop_height=crop_rect["height"],
    )


def further_processing_controls(params: PipelineParams) -> PipelineParams:
    st.subheader("Further Processing")
    area_label = "full image" if params.use_full_image else "selected crop"
    st.caption(
        f"These steps use the {area_label} from the Processing Region tab, after the "
        "current brightness/contrast range has been mapped to 0-255."
    )

    if params.use_full_image:
        st.warning("Full-image preprocessing can be slow because each slider change recomputes the affected steps.")

    stages = st.session_state.setdefault("preprocessing_stages", {})
    if st.button("Recompute all preprocessing"):
        stages.clear()
        st.session_state.pop("last_preprocess_params", None)
        st.session_state.pop("last_preprocess_result", None)
        st.rerun()

    return show_preprocess_stage_workflow(params, stages)


def show_preprocess_stage_workflow(params: PipelineParams, stages: dict) -> PipelineParams:
    region_sig = region_signature(params)

    st.markdown("**Step 1: Load Processing Region**")
    region_stage = stages.get("region")
    if not region_stage or region_stage.get("signature") != region_sig:
        with st.spinner("Updating Step 1..."):
            region_stage = load_preprocessing_region(params)
            stages["region"] = region_stage
            clear_preprocess_stages_after(stages, "region")
            st.session_state.pop("last_preprocess_params", None)
            st.session_state.pop("last_preprocess_result", None)

    crop = region_stage["crop"]
    stats = crop["raw_stats"]
    st.caption(
        f"Loaded region: origin x,y {crop['origin']} | "
        f"size {crop['display_rgb'].shape[1]} x {crop['display_rgb'].shape[0]}"
    )
    show_step_images(
        "Raw selected region, linear 0-255",
        crop["raw_normalized_rgb"],
        "Brightness/contrast mapped region",
        crop["display_rgb"],
        left_caption_suffix=f"min {stats['min']:.4g}, max {stats['max']:.4g}",
    )

    st.divider()
    st.markdown("**Step 2: Apply CLAHE**")
    clahe_clip_limit = st.slider(
        "CLAHE clip limit",
        0.001,
        1.000,
        float(params.clahe_clip_limit),
        0.001,
        key="pre_clahe_clip_limit",
        help="Limits local contrast amplification in CLAHE. Higher values reveal more local detail but can also amplify noise.",
    )
    params = replace(params, clahe_clip_limit=clahe_clip_limit)
    clahe_sig = clahe_signature(params)
    clahe_stage = stages.get("clahe")
    if not clahe_stage or clahe_stage.get("signature") != clahe_sig:
        with st.spinner("Updating Step 2..."):
            clahe_stage = run_clahe_stage(region_stage, params)
            stages["clahe"] = clahe_stage
            clear_preprocess_stages_after(stages, "clahe")
            st.session_state.pop("last_preprocess_params", None)
            st.session_state.pop("last_preprocess_result", None)
    show_step_images(
        "Input from Step 1",
        crop["display_rgb"],
        "CLAHE enhanced",
        clahe_stage["enhanced"],
        right_cmap="gray",
    )

    st.divider()
    st.markdown("**Step 3: Ridge Filter**")
    ridge_sigma_max = st.slider(
        "Ridge sigma max",
        1,
        12,
        int(params.ridge_sigma_max),
        1,
        key="pre_ridge_sigma_max",
        help="Largest ridge-filter scale in pixels. Higher values detect wider line-like features; lower values favor fine, narrow ridges.",
    )
    params = replace(params, ridge_sigma_max=ridge_sigma_max)
    ridge_sig = ridge_signature(params)
    ridge_stage = stages.get("ridge")
    if not ridge_stage or ridge_stage.get("signature") != ridge_sig:
        with st.spinner("Updating Step 3..."):
            ridge_stage = run_ridge_stage(clahe_stage, params)
            stages["ridge"] = ridge_stage
            clear_preprocess_stages_after(stages, "ridge")
            st.session_state.pop("last_preprocess_params", None)
            st.session_state.pop("last_preprocess_result", None)
    show_step_images(
        "Input from Step 2",
        clahe_stage["enhanced"],
        "Ridge response",
        ridge_stage["ridges"],
        left_cmap="gray",
        right_cmap="magma",
    )

    st.divider()
    st.markdown("**Step 4: Threshold Ridge Response**")
    col1, col2 = st.columns(2)
    with col1:
        ridge_percentile = st.slider(
            "Ridge percentile",
            80.0,
            99.9,
            float(params.ridge_percentile),
            0.1,
            key="pre_ridge_percentile",
            help="Percentile cutoff applied to the ridge response. Higher values keep only the strongest ridge pixels.",
        )
    with col2:
        threshold_multiplier = st.slider(
            "Otsu threshold multiplier",
            0.1,
            2.0,
            float(params.threshold_multiplier),
            0.05,
            key="pre_threshold_multiplier",
            help="Scales the automatic Otsu threshold. Higher values make the ridge mask stricter; lower values include more weak ridges.",
        )
    params = replace(
        params,
        ridge_percentile=ridge_percentile,
        threshold_multiplier=threshold_multiplier,
    )
    threshold_sig = threshold_signature(params)
    threshold_stage = stages.get("threshold")
    if not threshold_stage or threshold_stage.get("signature") != threshold_sig:
        with st.spinner("Updating Step 4..."):
            threshold_stage = run_threshold_stage(ridge_stage, params)
            stages["threshold"] = threshold_stage
            clear_preprocess_stages_after(stages, "threshold")
            st.session_state.pop("last_preprocess_params", None)
            st.session_state.pop("last_preprocess_result", None)
    cols = st.columns(3)
    cols[0].metric("Percentile threshold", f"{threshold_stage['percentile_threshold']:.4f}")
    cols[1].metric("Otsu x multiplier", f"{threshold_stage['otsu_threshold']:.4f}")
    cols[2].metric("Final threshold", f"{threshold_stage['ridge_threshold']:.4f}")
    show_step_images(
        "Input from Step 3",
        ridge_stage["ridges"],
        "Threshold ridge mask",
        threshold_stage["candidate_mask"],
        left_cmap="magma",
        right_cmap="gray",
    )

    st.divider()
    st.markdown("**Step 5: Clean Detection Mask**")
    col1, col2 = st.columns(2)
    with col1:
        min_object_size = st.slider(
            "Min ridge object pixels",
            1,
            500,
            int(params.min_object_size),
            5,
            key="pre_min_object_size",
            help="Removes connected ridge-mask objects smaller than this many pixels. Higher values clean specks but can delete short true events.",
        )
    with col2:
        closing_radius = st.slider(
            "Closing radius",
            0,
            10,
            int(params.closing_radius),
            1,
            key="pre_closing_radius",
            help="Radius of morphological closing on the ridge mask. Higher values bridge small gaps and thicken/merge nearby ridge regions.",
        )
    params = replace(
        params,
        min_object_size=min_object_size,
        closing_radius=closing_radius,
    )
    cleanup_sig = cleanup_signature(params)
    cleanup_stage = stages.get("cleanup")
    if not cleanup_stage or cleanup_stage.get("signature") != cleanup_sig:
        with st.spinner("Updating Step 5..."):
            cleanup_stage = run_cleanup_stage(threshold_stage, params)
            stages["cleanup"] = cleanup_stage
            clear_preprocess_stages_after(stages, "cleanup")
            st.session_state.pop("last_preprocess_params", None)
            st.session_state.pop("last_preprocess_result", None)
    show_step_images(
        "Input from Step 4",
        threshold_stage["candidate_mask"],
        "Clean detection mask",
        cleanup_stage["ridge"]["candidate_clean"],
        left_cmap="gray",
        right_cmap="gray",
    )

    st.divider()
    st.markdown("**Step 6: Skeletonize Mask**")
    use_skeletonize = st.checkbox(
        "Use skeletonized mask for seed detection",
        value=bool(params.use_skeletonize),
        key="pre_use_skeletonize",
        help=(
            "Converts the cleaned ridge mask into one-pixel-wide centerlines before "
            "Hough/RegionProps seed detection. Disable it to use the thicker cleaned mask directly."
        ),
    )
    params = replace(params, use_skeletonize=use_skeletonize)
    skeleton_sig = skeleton_signature(params)
    skeleton_stage = stages.get("skeleton")
    if not skeleton_stage or skeleton_stage.get("signature") != skeleton_sig:
        with st.spinner("Updating Step 6..."):
            skeleton_stage = run_skeletonize_stage(cleanup_stage, params)
            stages["skeleton"] = skeleton_stage
            st.session_state["last_preprocess_params"] = params
            st.session_state["last_preprocess_result"] = skeleton_stage
    final_title = "Skeletonized seed mask" if use_skeletonize else "Clean mask used for seed detection"
    show_step_images(
        "Input from Step 5",
        cleanup_stage["ridge"]["candidate_clean"],
        final_title,
        skeleton_stage["ridge"].get("display_mask", skeleton_stage["ridge"]["detection_mask"]),
        left_cmap="gray",
        right_cmap="gray",
    )
    show_preprocess_metrics(skeleton_stage)

    return params


def show_step_images(
    left_title: str,
    left_image: np.ndarray,
    right_title: str,
    right_image: np.ndarray,
    left_cmap: str | None = None,
    right_cmap: str | None = None,
    left_caption_suffix: str | None = None,
) -> None:
    col1, col2 = st.columns(2)
    left_caption = left_title
    if left_caption_suffix:
        left_caption = f"{left_caption} ({left_caption_suffix})"
    col1.image(to_display(left_image, cmap=left_cmap), caption=left_caption, width='stretch')
    col2.image(to_display(right_image, cmap=right_cmap), caption=right_title, width='stretch')


def region_signature(params: PipelineParams) -> tuple:
    return (
        params.image_path,
        params.display_min,
        params.display_max,
        params.use_full_image,
        params.crop_x,
        params.crop_y,
        params.crop_width,
        params.crop_height,
    )


def clahe_signature(params: PipelineParams) -> tuple:
    return (*region_signature(params), params.clahe_clip_limit)


def ridge_signature(params: PipelineParams) -> tuple:
    return (*clahe_signature(params), params.ridge_sigma_max)


def threshold_signature(params: PipelineParams) -> tuple:
    return (*ridge_signature(params), params.ridge_percentile, params.threshold_multiplier)


def cleanup_signature(params: PipelineParams) -> tuple:
    return (*threshold_signature(params), params.min_object_size, params.closing_radius)


def skeleton_signature(params: PipelineParams) -> tuple:
    return (*cleanup_signature(params), params.use_skeletonize)


def clear_preprocess_stages_after(stages: dict, stage_name: str) -> None:
    order = ["region", "clahe", "ridge", "threshold", "cleanup", "skeleton"]
    if stage_name not in order:
        return
    for later_stage in order[order.index(stage_name) + 1:]:
        stages.pop(later_stage, None)


def load_preprocessing_region(params: PipelineParams) -> dict:
    crop = load_crop_by_rect(
        params.image_path,
        params.crop_x,
        params.crop_y,
        params.crop_width,
        params.crop_height,
        params.display_min,
        params.display_max,
    )
    return {
        "signature": region_signature(params),
        "params": params,
        "crop": crop,
        "gray": detection_rgb_to_unit_gray(crop["detection_rgb"]),
    }


def run_clahe_stage(region_stage: dict, params: PipelineParams) -> dict:
    enhanced = equalize_adapthist(
        region_stage["gray"],
        clip_limit=params.clahe_clip_limit,
    ).astype(np.float32)
    return {
        "signature": clahe_signature(params),
        "params": params,
        "crop": region_stage["crop"],
        "gray": region_stage["gray"],
        "enhanced": enhanced,
    }


def run_ridge_stage(clahe_stage: dict, params: PipelineParams) -> dict:
    ridges = meijering(
        clahe_stage["enhanced"],
        sigmas=tuple(range(1, params.ridge_sigma_max + 1)),
        black_ridges=False,
    ).astype(np.float32)
    return {
        "signature": ridge_signature(params),
        "params": params,
        "crop": clahe_stage["crop"],
        "gray": clahe_stage["gray"],
        "enhanced": clahe_stage["enhanced"],
        "ridges": ridges,
    }


def run_threshold_stage(ridge_stage: dict, params: PipelineParams) -> dict:
    percentile_threshold = float(np.percentile(ridge_stage["ridges"], params.ridge_percentile))
    otsu_threshold = float(threshold_otsu(ridge_stage["ridges"]) * params.threshold_multiplier)
    ridge_threshold = max(percentile_threshold, otsu_threshold)
    candidate_mask = ridge_stage["ridges"] > ridge_threshold
    return {
        "signature": threshold_signature(params),
        "params": params,
        "crop": ridge_stage["crop"],
        "gray": ridge_stage["gray"],
        "enhanced": ridge_stage["enhanced"],
        "ridges": ridge_stage["ridges"],
        "percentile_threshold": percentile_threshold,
        "otsu_threshold": otsu_threshold,
        "ridge_threshold": ridge_threshold,
        "candidate_mask": candidate_mask,
    }


def run_cleanup_stage(threshold_stage: dict, params: PipelineParams) -> dict:
    candidate_clean = remove_small_objects_by_label(
        threshold_stage["candidate_mask"],
        params.min_object_size,
    )
    if params.closing_radius > 0:
        candidate_clean = morphology.closing(candidate_clean, morphology.disk(params.closing_radius))
    grow_gray = (np.clip(threshold_stage["enhanced"], 0.0, 1.0) * 255).astype(np.uint8)
    return {
        "signature": cleanup_signature(params),
        "params": params,
        "crop": threshold_stage["crop"],
        "ridge": {
            "gray": threshold_stage["gray"],
            "enhanced": threshold_stage["enhanced"],
            "ridges": threshold_stage["ridges"],
            "ridge_threshold": threshold_stage["ridge_threshold"],
            "candidate_mask": threshold_stage["candidate_mask"],
            "candidate_clean": candidate_clean,
            "detection_mask": candidate_clean,
            "grow_rgb": gray_to_rgb(grow_gray),
        },
    }


def run_skeletonize_stage(cleanup_stage: dict, params: PipelineParams) -> dict:
    clean_mask = cleanup_stage["ridge"]["candidate_clean"].astype(bool, copy=False)
    if params.use_skeletonize:
        skeleton_mask = morphology.skeletonize(clean_mask)
        detection_mask = skeleton_mask
    else:
        skeleton_mask = None
        detection_mask = clean_mask

    display_mask = mask_for_visibility(detection_mask, dilate=params.use_skeletonize)

    ridge = {
        **cleanup_stage["ridge"],
        "skeleton_mask": skeleton_mask,
        "detection_mask": detection_mask,
        "display_mask": display_mask,
        "use_skeletonize": params.use_skeletonize,
    }
    return {
        "signature": skeleton_signature(params),
        "params": params,
        "crop": cleanup_stage["crop"],
        "ridge": ridge,
    }


def detection_rgb_to_unit_gray(detection_rgb: np.ndarray) -> np.ndarray:
    gray = np.nan_to_num(
        detection_rgb[..., 2].astype(np.float32, copy=False),
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )
    return np.clip(gray / 255.0, 0.0, 1.0)


def brightness_contrast_controls(path_str: str) -> tuple[float, float]:
    summary = image_value_summary(path_str)
    raw_min = summary["min"]
    raw_max = summary["max"]
    if raw_max <= raw_min:
        return raw_min, raw_max

    range_key = f"display_range_values_v2:{path_str}"
    slider_key = f"display_range_slider_v2:{path_str}"
    if range_key not in st.session_state:
        st.session_state[range_key] = default_display_range(raw_min, raw_max)

    st.subheader("Brightness / Contrast")
    st.caption(
        "Fiji-style display range. This remaps raw BLN values to 0-255 before the later "
        "preprocessing steps."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw min", f"{raw_min:.4g}")
    col2.metric("Raw max", f"{raw_max:.4g}")
    col3.metric("Auto low", f"{summary['p0_5']:.4g}")
    col4.metric("Auto high", f"{summary['p99_5']:.4g}")

    button_col1, button_col2 = st.columns(2)
    with button_col1:
        if st.button("Auto range", help="Use the 0.5th and 99.5th percentile values."):
            st.session_state[range_key] = (summary["p0_5"], summary["p99_5"])
            st.session_state[slider_key] = st.session_state[range_key]
            st.rerun()
    with button_col2:
        if st.button("Reset range", help="Use the true minimum and maximum values."):
            st.session_state[range_key] = (raw_min, raw_max)
            st.session_state[slider_key] = st.session_state[range_key]
            st.rerun()

    span = raw_max - raw_min
    step = max(span / 2000.0, 0.0001)
    current = sanitize_display_range(st.session_state[range_key], raw_min, raw_max)
    display_min, display_max = st.slider(
        "Display range",
        min_value=float(raw_min),
        max_value=float(raw_max),
        value=(float(current[0]), float(current[1])),
        step=float(step),
        key=slider_key,
        help="Fiji/ImageJ-style brightness and contrast range. Values below the minimum map to black, above the maximum map to white, and values between map linearly to 0-255.",
    )
    display_min, display_max = sanitize_display_range((display_min, display_max), raw_min, raw_max)
    st.session_state[range_key] = (display_min, display_max)

    midpoint = (display_min + display_max) / 2.0
    width = display_max - display_min
    cols = st.columns(4)
    cols[0].metric("Minimum", f"{display_min:.4g}")
    cols[1].metric("Maximum", f"{display_max:.4g}")
    cols[2].metric("Brightness", f"{midpoint:.4g}")
    cols[3].metric("Contrast width", f"{width:.4g}")
    return display_min, display_max


def sanitize_display_range(value: tuple[float, float] | list[float], raw_min: float, raw_max: float) -> tuple[float, float]:
    display_min, display_max = value
    display_min = max(raw_min, min(float(display_min), raw_max))
    display_max = max(raw_min, min(float(display_max), raw_max))
    if display_max <= display_min:
        display_max = min(raw_max, display_min + max((raw_max - raw_min) / 2000.0, 0.0001))
        if display_max <= display_min:
            display_min = raw_min
            display_max = raw_max
    return display_min, display_max


def default_display_range(raw_min: float, raw_max: float) -> tuple[float, float]:
    if raw_min <= 0.0 and raw_max >= 1.0:
        return 0.0, 1.0
    return raw_min, raw_max


def sanitize_crop_rect(value: dict | None, image_width: int, image_height: int) -> dict[str, int]:
    if not isinstance(value, dict):
        value = {}

    width = int(round(float(value.get("width", min(500, image_width)))))
    height = int(round(float(value.get("height", min(500, image_height)))))
    width = max(1, min(width, image_width))
    height = max(1, min(height, image_height))

    x = int(round(float(value.get("x", max(0, (image_width - width) // 2)))))
    y = int(round(float(value.get("y", max(0, (image_height - height) // 2)))))
    x = max(0, min(x, image_width - width))
    y = max(0, min(y, image_height - height))

    return {"x": x, "y": y, "width": width, "height": height}


@st.cache_data(show_spinner=False)
def image_size(path_str: str) -> tuple[int, int]:
    with Image.open(path_str) as img:
        return img.size


@st.cache_data(show_spinner=False)
def image_value_summary(path_str: str) -> dict[str, float]:
    with Image.open(path_str) as img:
        arr = np.asarray(img).copy()

    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[..., :3]
    values = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    percentiles = np.percentile(values, [0.5, 1.0, 99.0, 99.5])
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "p0_5": float(percentiles[0]),
        "p1": float(percentiles[1]),
        "p99": float(percentiles[2]),
        "p99_5": float(percentiles[3]),
    }


@st.cache_data(show_spinner=False)
def load_region_selector_preview(
    path_str: str,
    view_x: int,
    view_y: int,
    view_width: int,
    view_height: int,
    display_min: float | None = None,
    display_max: float | None = None,
    max_dim: int = 1400,
) -> dict:
    with Image.open(path_str) as img:
        original_width, original_height = img.size
        view_width = max(1, min(int(view_width), original_width))
        view_height = max(1, min(int(view_height), original_height))
        view_x = max(0, min(int(view_x), original_width - view_width))
        view_y = max(0, min(int(view_y), original_height - view_height))
        preview_img = img.crop((view_x, view_y, view_x + view_width, view_y + view_height))
        preview_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        arr = np.asarray(preview_img).copy()

    preview_rgb = display_crop_rgb(arr, display_min, display_max)
    return {
        "image_data_url": image_to_png_data_url(preview_rgb),
        "preview_width": int(preview_rgb.shape[1]),
        "preview_height": int(preview_rgb.shape[0]),
        "original_width": int(original_width),
        "original_height": int(original_height),
        "view_x": int(view_x),
        "view_y": int(view_y),
        "view_width": int(view_width),
        "view_height": int(view_height),
    }


@st.cache_data(show_spinner=False)
def load_crop_by_rect(
    path_str: str,
    crop_x: int,
    crop_y: int,
    crop_width: int,
    crop_height: int,
    display_min: float | None = None,
    display_max: float | None = None,
) -> dict:
    with Image.open(path_str) as img:
        img_w, img_h = img.size
        crop_width = max(1, min(int(crop_width), img_w))
        crop_height = max(1, min(int(crop_height), img_h))
        x0 = max(0, min(int(crop_x), img_w - crop_width))
        y0 = max(0, min(int(crop_y), img_h - crop_height))
        x1 = x0 + crop_width
        y1 = y0 + crop_height
        arr = np.asarray(img.crop((x0, y0, x1, y1))).copy()

    is_full_image = crop_width == img_w and crop_height == img_h and x0 == 0 and y0 == 0
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = to_uint8_rgb(arr)
        raw_note = "Full RGB image" if is_full_image else "RGB crop"
        return {
            "raw": arr,
            "raw_direct_rgb": direct_to_uint8_rgb(arr),
            "raw_normalized_rgb": normalize_to_uint8_rgb(arr),
            "raw_stats": array_stats(arr),
            "display_rgb": rgb,
            "detection_rgb": rgb,
            "origin": (x0, y0),
            "note": raw_note,
        }

    loaded_rgb = display_crop_rgb(arr, display_min, display_max)
    raw_note = (
        "Full scalar image with brightness/contrast display and detection buffers"
        if is_full_image
        else "Scalar crop with brightness/contrast display and detection buffers"
    )
    return {
        "raw": arr,
        "raw_direct_rgb": direct_to_uint8_rgb(arr),
        "raw_normalized_rgb": normalize_to_uint8_rgb(arr),
        "raw_stats": array_stats(arr),
        "display_rgb": loaded_rgb,
        "detection_rgb": loaded_rgb,
        "origin": (x0, y0),
        "note": raw_note,
    }


@st.cache_data(show_spinner=False)
def load_display_crop_at(path_str: str, x0: int, y0: int, width: int, height: int) -> dict | None:
    if not path_str.strip():
        return None

    with Image.open(path_str) as img:
        img_w, img_h = img.size
        crop_x0 = max(0, min(x0, img_w - 1))
        crop_y0 = max(0, min(y0, img_h - 1))
        crop_x1 = min(img_w, crop_x0 + width)
        crop_y1 = min(img_h, crop_y0 + height)
        arr = np.asarray(img.crop((crop_x0, crop_y0, crop_x1, crop_y1))).copy()

    return {
        "raw": arr,
        "display_rgb": display_crop_rgb(arr),
        "raw_stats": array_stats(arr),
        "origin": (crop_x0, crop_y0),
        "note": f"Overlay crop from {Path(path_str).name}",
    }


@st.cache_data(show_spinner=False)
def run_ebsd_gradient_boundary(
    path_str: str,
    crop_x: int,
    crop_y: int,
    crop_width: int,
    crop_height: int,
    color_space: str,
    gradient_sigma: float,
    gradient_percentile: float,
    min_boundary_object_size: int,
    boundary_closing_radius: int,
    use_boundary_skeleton: bool,
    boundary_display_radius: int,
    overlay_color_name: str,
) -> dict:
    crop = load_crop_by_rect(path_str, crop_x, crop_y, crop_width, crop_height)
    rgb = crop["display_rgb"]
    color_gradient = color_gradient_magnitude(rgb, color_space=color_space, sigma=gradient_sigma)
    gradient_threshold = float(np.percentile(color_gradient, gradient_percentile))

    raw_boundary_mask = color_gradient >= gradient_threshold
    clean_boundary_mask = raw_boundary_mask
    if min_boundary_object_size > 1:
        clean_boundary_mask = remove_small_objects_by_label(clean_boundary_mask, min_boundary_object_size)
    if boundary_closing_radius > 0:
        clean_boundary_mask = morphology.closing(
            clean_boundary_mask,
            morphology.disk(boundary_closing_radius),
        )

    if use_boundary_skeleton:
        core_boundary_mask = morphology.thin(clean_boundary_mask)
    else:
        core_boundary_mask = clean_boundary_mask.astype(bool, copy=False)

    display_boundary_mask = core_boundary_mask
    if boundary_display_radius > 0:
        display_boundary_mask = ndi.binary_dilation(
            display_boundary_mask,
            iterations=boundary_display_radius,
        )

    overlay = draw_boundary_overlay(
        rgb,
        display_boundary_mask,
        boundary_color_rgb(overlay_color_name),
    )
    labeled_boundaries = measure.label(core_boundary_mask)
    return {
        "crop": crop,
        "color_space": color_space,
        "gradient_sigma": gradient_sigma,
        "gradient_percentile": gradient_percentile,
        "gradient_threshold": gradient_threshold,
        "color_gradient": color_gradient,
        "raw_boundary_mask": raw_boundary_mask,
        "clean_boundary_mask": clean_boundary_mask,
        "core_boundary_mask": core_boundary_mask,
        "display_boundary_mask": display_boundary_mask,
        "overlay": overlay,
        "boundary_fragment_count": int(labeled_boundaries.max()),
        "overlay_color_name": overlay_color_name,
    }


@st.cache_data(show_spinner=False)
def run_ebsd_cuts_gradient_overlay(
    path_str: str,
    crop_x: int,
    crop_y: int,
    crop_width: int,
    crop_height: int,
    color_space: str,
    gradient_sigma: float,
    normalize_low_pct: float,
    normalize_high_pct: float,
    normalized_threshold: float,
    dilation_radius: int,
) -> dict:
    crop = load_crop_by_rect(path_str, crop_x, crop_y, crop_width, crop_height)
    rgb = crop["display_rgb"]
    color_gradient = color_gradient_magnitude(rgb, color_space=color_space, sigma=gradient_sigma)
    gradient_norm, norm_low, norm_high = normalize_with_percentile_range(
        color_gradient,
        normalize_low_pct,
        normalize_high_pct,
    )
    threshold_raw = float(norm_low + normalized_threshold * (norm_high - norm_low))
    threshold_mask = gradient_norm >= normalized_threshold

    display_mask = threshold_mask.astype(bool, copy=True)
    if dilation_radius > 0:
        display_mask = ndi.binary_dilation(
            display_mask,
            structure=morphology.disk(dilation_radius),
        )

    overlay = draw_boundary_overlay(rgb, display_mask, (0, 0, 0))
    labeled = measure.label(threshold_mask)
    return {
        "crop": crop,
        "color_space": color_space,
        "gradient_sigma": gradient_sigma,
        "normalize_low_pct": normalize_low_pct,
        "normalize_high_pct": normalize_high_pct,
        "normalized_threshold": normalized_threshold,
        "threshold_raw": threshold_raw,
        "norm_low": norm_low,
        "norm_high": norm_high,
        "color_gradient": color_gradient,
        "gradient_norm": gradient_norm,
        "threshold_mask": threshold_mask,
        "display_mask": display_mask,
        "overlay": overlay,
        "fragment_count": int(labeled.max()),
    }


def color_gradient_magnitude(rgb: np.ndarray, color_space: str, sigma: float) -> np.ndarray:
    rgb = rgb[..., :3]
    if color_space.lower() == "lab":
        work = rgb2lab(rgb).astype(np.float32, copy=False)
    else:
        work = rgb.astype(np.float32, copy=False) / 255.0

    sigma = max(0.0, float(sigma))
    if sigma > 0:
        work = ndi.gaussian_filter(work, sigma=(sigma, sigma, 0), mode="reflect")

    gradient_sq = np.zeros(work.shape[:2], dtype=np.float32)
    for channel in range(work.shape[2]):
        channel_values = work[..., channel]
        grad_x = ndi.sobel(channel_values, axis=1, mode="reflect")
        grad_y = ndi.sobel(channel_values, axis=0, mode="reflect")
        gradient_sq += grad_x * grad_x + grad_y * grad_y
    return np.sqrt(gradient_sq).astype(np.float32, copy=False)


def normalize_with_percentile_range(
    values: np.ndarray,
    low_pct: float,
    high_pct: float,
) -> tuple[np.ndarray, float, float]:
    values = np.nan_to_num(values.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(values, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32), float(lo), float(hi)
    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return normalized, float(lo), float(hi)


def show_ebsd_cuts_result(result: dict) -> None:
    crop = result["crop"]
    rgb = crop["display_rgb"]
    height, width = rgb.shape[:2]
    threshold_pixels = int(result["threshold_mask"].sum())
    display_pixels = int(result["display_mask"].sum())
    coverage = threshold_pixels / max(1, height * width) * 100.0

    cols = st.columns(6)
    cols[0].metric("Processing size", f"{width} x {height}")
    cols[1].metric("Gradient threshold", f"{result['threshold_raw']:.4g}")
    cols[2].metric("Norm threshold", f"{result['normalized_threshold']:.2f}")
    cols[3].metric("Threshold pixels", f"{threshold_pixels:,}")
    cols[4].metric("Coverage", f"{coverage:.2f}%")
    cols[5].metric("Fragments", result["fragment_count"])

    st.caption(
        f"EBSD crop origin x,y {crop['origin']} | {crop['note']} | "
        f"{result['color_space'].upper()} gradient, sigma {result['gradient_sigma']} | "
        f"normalization raw range {result['norm_low']:.4g} to {result['norm_high']:.4g} | "
        f"display pixels after dilation {display_pixels:,}"
    )

    col1, col2 = st.columns(2)
    col1.image(to_display(rgb), caption="Original EBSD crop matching current DIC crop", width='stretch')
    col2.image(
        to_display(result["gradient_norm"], cmap="magma"),
        caption="EBSD color-gradient magnitude, normalized",
        width='stretch',
    )

    col1, col2 = st.columns(2)
    col1.image(
        to_display(result["display_mask"], cmap="gray"),
        caption="Thresholded gradient points, dilated for display",
        width='stretch',
    )
    with col2:
        zoomable_image(
            result["overlay"],
            "EBSD crop with thresholded gradient points in black",
            key="ebsd_cuts_black_overlay",
        )


def show_ebsd_gradient_result(result: dict) -> None:
    crop = result["crop"]
    rgb = crop["display_rgb"]
    core_boundary_mask = result["core_boundary_mask"]
    display_boundary_mask = result["display_boundary_mask"]
    height, width = rgb.shape[:2]

    cols = st.columns(6)
    cols[0].metric("Processing size", f"{width} x {height}")
    cols[1].metric("Gradient threshold", f"{result['gradient_threshold']:.4g}")
    cols[2].metric("Raw boundary pixels", int(result["raw_boundary_mask"].sum()))
    cols[3].metric("Clean/core pixels", int(core_boundary_mask.sum()))
    cols[4].metric("Display pixels", int(display_boundary_mask.sum()))
    cols[5].metric("Fragments", result["boundary_fragment_count"])
    st.caption(
        f"Region origin x,y {crop['origin']} | {crop['note']} | "
        f"{result['color_space'].upper()} gradient, sigma {result['gradient_sigma']}, "
        f"percentile {result['gradient_percentile']}"
    )

    col1, col2 = st.columns(2)
    col1.image(to_display(rgb), caption="Selected EBSD region", width='stretch')
    col2.image(
        to_display(result["color_gradient"], cmap="magma"),
        caption="Color-gradient magnitude",
        width='stretch',
    )

    col1, col2 = st.columns(2)
    col1.image(
        to_display(display_boundary_mask, cmap="gray"),
        caption="Detected gradient boundary mask",
        width='stretch',
    )
    with col2:
        zoomable_image(
            to_display(result["overlay"], max_dim=2400),
            "EBSD gradient boundary overlay",
            key="ebsd_gradient_boundary_overlay",
        )


def save_ebsd_boundary_outputs(result: dict, source_path: str) -> list[Path]:
    out_dir = ROOT / "tests" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop = result["crop"]
    origin_x, origin_y = crop["origin"]
    height, width = crop["display_rgb"].shape[:2]
    stem = Path(source_path).stem
    suffix = f"x{origin_x}_y{origin_y}_w{width}_h{height}"

    boundary_path = out_dir / f"{stem}_gradient_boundaries_{suffix}.png"
    overlay_path = out_dir / f"{stem}_gradient_boundary_overlay_{suffix}.png"

    boundary_image = (result["display_boundary_mask"].astype(np.uint8) * 255)
    Image.fromarray(boundary_image).save(boundary_path)
    Image.fromarray(result["overlay"]).save(overlay_path)
    return [boundary_path, overlay_path]


def draw_boundary_overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.copy()
    out[mask.astype(bool, copy=False)] = color
    return out


def boundary_color_rgb(name: str) -> tuple[int, int, int]:
    colors = {
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "yellow": (255, 230, 0),
        "cyan": (0, 220, 255),
        "blue": (0, 80, 255),
    }
    return colors.get(name, colors["white"])


@st.cache_data(show_spinner=False)
def compute_ridges(
    detection_rgb: np.ndarray,
    clahe_clip_limit: float,
    ridge_sigma_max: int,
    ridge_percentile: float,
    threshold_multiplier: float,
    min_object_size: int,
    closing_radius: int,
    use_skeletonize: bool = False,
) -> dict:
    gray = detection_rgb_to_unit_gray(detection_rgb)
    enhanced = equalize_adapthist(gray, clip_limit=clahe_clip_limit).astype(np.float32)
    ridges = meijering(
        enhanced,
        sigmas=tuple(range(1, ridge_sigma_max + 1)),
        black_ridges=False,
    ).astype(np.float32)

    percentile_threshold = float(np.percentile(ridges, ridge_percentile))
    otsu_threshold = float(threshold_otsu(ridges) * threshold_multiplier)
    ridge_threshold = max(percentile_threshold, otsu_threshold)

    candidate_mask = ridges > ridge_threshold
    candidate_clean = remove_small_objects_by_label(candidate_mask, min_object_size)
    if closing_radius > 0:
        candidate_clean = morphology.closing(candidate_clean, morphology.disk(closing_radius))
    skeleton_mask = morphology.skeletonize(candidate_clean) if use_skeletonize else None
    detection_mask = skeleton_mask if use_skeletonize else candidate_clean
    display_mask = mask_for_visibility(detection_mask, dilate=use_skeletonize)

    grow_gray = (np.clip(enhanced, 0.0, 1.0) * 255).astype(np.uint8)
    return {
        "gray": gray,
        "enhanced": enhanced,
        "ridges": ridges,
        "ridge_threshold": ridge_threshold,
        "candidate_mask": candidate_mask,
        "candidate_clean": candidate_clean,
        "skeleton_mask": skeleton_mask,
        "detection_mask": detection_mask,
        "display_mask": display_mask,
        "use_skeletonize": use_skeletonize,
        "grow_rgb": gray_to_rgb(grow_gray),
    }


def run_preprocessing(params: PipelineParams) -> dict:
    region_stage = load_preprocessing_region(params)
    clahe_stage = run_clahe_stage(region_stage, params)
    ridge_stage = run_ridge_stage(clahe_stage, params)
    threshold_stage = run_threshold_stage(ridge_stage, params)
    cleanup_stage = run_cleanup_stage(threshold_stage, params)
    return run_skeletonize_stage(cleanup_stage, params)


def hough_source_signature(params: PipelineParams) -> tuple:
    return (
        params.image_path,
        params.grx_overlay_path,
        params.display_min,
        params.display_max,
        params.use_full_image,
        params.crop_x,
        params.crop_y,
        params.crop_width,
        params.crop_height,
        params.clahe_clip_limit,
        params.ridge_sigma_max,
        params.ridge_percentile,
        params.threshold_multiplier,
        params.min_object_size,
        params.closing_radius,
        params.use_skeletonize,
        params.hough_threshold,
        params.hough_line_length,
        params.hough_line_gap,
        params.hough_seed_spacing,
        params.hough_max_seeds,
        params.hough_use_all_seeds,
    )


def hough_source_matches_result(result: dict | None, params: PipelineParams) -> bool:
    if not result or "hough" not in result:
        return False
    result_params = result.get("params")
    if not isinstance(result_params, PipelineParams):
        return False
    return hough_source_signature(result_params) == hough_source_signature(params)


def run_hough_seed_pipeline(params: PipelineParams) -> dict:
    preprocessing = run_preprocessing(params)
    crop = preprocessing["crop"]
    ridge_result = preprocessing["ridge"]
    crop_height, crop_width = crop["display_rgb"].shape[:2]
    grx_overlay = load_display_crop_at(
        params.grx_overlay_path,
        crop["origin"][0],
        crop["origin"][1],
        crop_width,
        crop_height,
    )

    hough = hough_seed_method(
        ridge_result["detection_mask"],
        ridge_result["enhanced"],
        params.hough_threshold,
        params.hough_line_length,
        params.hough_line_gap,
        params.hough_seed_spacing,
        None if params.hough_use_all_seeds else params.hough_max_seeds,
    )
    return {
        **preprocessing,
        "grx_overlay": grx_overlay,
        "hough": hough,
    }


def trace_hough_events_for_result(result: dict, params: PipelineParams) -> dict:
    if not hough_source_matches_result(result, params):
        result = run_hough_seed_pipeline(params)

    hough_events = grow_events_from_seeds(
        result["hough"]["seeds"],
        result["ridge"]["grow_rgb"],
        params,
    )
    return {
        **result,
        "params": params,
        "hough_events": hough_events,
    }


def run_pipeline(params: PipelineParams) -> dict:
    preprocessing = run_hough_seed_pipeline(params)
    crop = preprocessing["crop"]
    ridge_result = preprocessing["ridge"]
    regionprops = regionprops_seed_method(
        ridge_result["detection_mask"],
        ridge_result["enhanced"],
        params.region_min_pixels,
        params.region_seed_spacing,
        None if params.region_use_all_seeds else params.region_max_seeds,
    )

    hough_events = grow_events_from_seeds(preprocessing["hough"]["seeds"], ridge_result["grow_rgb"], params)
    region_events = grow_events_from_seeds(regionprops["seeds"], ridge_result["grow_rgb"], params)

    return {
        **preprocessing,
        "regionprops": regionprops,
        "hough_events": hough_events,
        "regionprops_events": region_events,
    }


def hough_seed_method(
    detection_mask: np.ndarray,
    enhanced: np.ndarray,
    threshold: int,
    line_length: int,
    line_gap: int,
    seed_spacing: int,
    max_seeds: int | None,
) -> dict:
    hough_lines = probabilistic_hough_line(
        detection_mask,
        threshold=threshold,
        line_length=line_length,
        line_gap=line_gap,
    )
    seed_candidates = []
    for p0, p1 in hough_lines:
        x0, y0 = p0
        x1, y1 = p1
        rr, cc = draw_line(y0, x0, y1, x1)
        valid = (rr >= 0) & (rr < enhanced.shape[0]) & (cc >= 0) & (cc < enhanced.shape[1])
        rr = rr[valid]
        cc = cc[valid]
        if len(rr) == 0:
            continue
        best = int(np.argmax(enhanced[rr, cc]))
        seed_candidates.append((int(rr[best]), int(cc[best]), float(enhanced[rr[best], cc[best]])))

    effective_max_seeds = len(seed_candidates) if max_seeds is None else max_seeds
    seeds = select_spaced_seeds(seed_candidates, enhanced.shape, seed_spacing, effective_max_seeds)
    return {
        "lines": hough_lines,
        "raw_count": len(hough_lines),
        "candidate_seed_count": len(seed_candidates),
        "max_seeds": effective_max_seeds,
        "use_all_seeds": max_seeds is None,
        "seeds": seeds,
    }


def regionprops_seed_method(
    detection_mask: np.ndarray,
    enhanced: np.ndarray,
    min_pixels: int,
    seed_spacing: int,
    max_seeds: int | None,
) -> dict:
    labels = measure.label(detection_mask)
    props = [prop for prop in measure.regionprops(labels) if prop.area >= min_pixels]
    seed_candidates = []
    for prop in props:
        rr, cc = prop.coords.T
        best = int(np.argmax(enhanced[rr, cc]))
        seed_candidates.append((int(rr[best]), int(cc[best]), float(enhanced[rr[best], cc[best]])))

    effective_max_seeds = len(seed_candidates) if max_seeds is None else max_seeds
    seeds = select_spaced_seeds(seed_candidates, enhanced.shape, seed_spacing, effective_max_seeds)
    return {
        "props": props,
        "raw_count": len(props),
        "candidate_seed_count": len(seed_candidates),
        "max_seeds": effective_max_seeds,
        "use_all_seeds": max_seeds is None,
        "seeds": seeds,
    }


def grow_events_from_seeds(seeds: list[Point], image_rgb: np.ndarray, params: PipelineParams) -> dict:
    accepted_entries: list[dict] = []
    rejected = 0
    duplicates = 0
    merged = 0
    covered = 0

    for seed in seeds:
        result = detect_line_from_seed(
            seed.x,
            seed.y,
            image_rgb,
            intensity_difference_tolerance=params.intensity_difference_tolerance,
            bfl_tolerance=params.best_fit_line_tolerance,
            min_intensity=params.min_intensity,
            min_points_threshold=params.min_points_threshold,
        )
        if result.line is None:
            rejected += 1
            continue
        merge_indices = [
            index
            for index, entry in enumerate(accepted_entries)
            if should_merge_events(result.line, entry["line"], params)
        ]
        if merge_indices:
            merged_line = result.line
            merged_seeds = {seed}
            for index in sorted(merge_indices, reverse=True):
                entry = accepted_entries.pop(index)
                merged_line = merge_event_lines(
                    entry["line"],
                    merged_line,
                    connect_gaps=params.connect_merged_event_gaps,
                )
                merged_seeds.update(entry["seeds"])
            accepted_entries.append({"line": merged_line, "seeds": merged_seeds, "was_merged": True})
            merged += 1
        else:
            accepted_entries.append({"line": result.line, "seeds": {seed}, "was_merged": False})

    accepted = [entry["line"] for entry in accepted_entries]
    standalone_seeds = [
        seed
        for entry in accepted_entries
        if not entry["was_merged"]
        for seed in sorted(entry["seeds"])
    ]
    merged_seeds = [
        seed
        for entry in accepted_entries
        if entry["was_merged"]
        for seed in sorted(entry["seeds"])
    ]
    merged_groups = [
        {
            "line": entry["line"],
            "seeds": sorted(entry["seeds"]),
        }
        for entry in accepted_entries
        if entry["was_merged"]
    ]

    return {
        "accepted": accepted,
        "accepted_count": len(accepted),
        "rejected": rejected,
        "duplicates": duplicates,
        "merged": merged,
        "covered": covered,
        "standalone_seeds": standalone_seeds,
        "merged_seeds": merged_seeds,
        "merged_groups": merged_groups,
    }


def show_summary(result: dict) -> None:
    st.subheader("Counts")
    cols = st.columns(4)
    cols[0].metric("Hough lines", result["hough"]["raw_count"])
    cols[1].metric("Hough seeds", len(result["hough"]["seeds"]))
    cols[2].metric("Hough events", result["hough_events"]["accepted_count"])
    cols[3].metric("Hough rejected", result["hough_events"]["rejected"])

    cols = st.columns(4)
    cols[0].metric("RegionProp ridges", result["regionprops"]["raw_count"])
    cols[1].metric("RegionProp seeds", len(result["regionprops"]["seeds"]))
    cols[2].metric("RegionProp events", result["regionprops_events"]["accepted_count"])
    cols[3].metric("RegionProp rejected", result["regionprops_events"]["rejected"])

    cols = st.columns(5)
    cols[0].metric("Candidate ridge pixels", int(result["ridge"]["candidate_mask"].sum()))
    cols[1].metric("Clean mask pixels", int(result["ridge"]["candidate_clean"].sum()))
    cols[2].metric("Seed mask pixels", int(result["ridge"]["detection_mask"].sum()))
    cols[3].metric("Seed mask coverage", f"{result['ridge']['detection_mask'].mean() * 100:.2f}%")
    cols[4].metric("Ridge threshold", f"{result['ridge']['ridge_threshold']:.4f}")

    params = result["params"]
    crop_height, crop_width = result["crop"]["display_rgb"].shape[:2]
    area_label = "Full image" if params.use_full_image else "Crop"
    st.caption(
        f"{area_label}: origin x,y {result['crop']['origin']} | "
        f"size {crop_width} x {crop_height} | {result['crop']['note']}"
    )


def show_preprocess_metrics(result: dict) -> None:
    ridge = result["ridge"]
    crop_height, crop_width = result["crop"]["display_rgb"].shape[:2]
    cols = st.columns(6)
    cols[0].metric("Processing size", f"{crop_width} x {crop_height}")
    cols[1].metric("Candidate ridge pixels", int(ridge["candidate_mask"].sum()))
    cols[2].metric("Clean mask pixels", int(ridge["candidate_clean"].sum()))
    cols[3].metric("Seed mask pixels", int(ridge["detection_mask"].sum()))
    cols[4].metric("Seed mask coverage", f"{ridge['detection_mask'].mean() * 100:.2f}%")
    cols[5].metric("Ridge threshold", f"{ridge['ridge_threshold']:.4f}")


def hough_method_controls(params: PipelineParams, stored_result: dict | None) -> PipelineParams:
    st.markdown("**Hough Line Controls**")
    col1, col2, col3 = st.columns(3)
    with col1:
        hough_threshold = st.slider(
            "Hough threshold",
            1,
            100,
            int(params.hough_threshold),
            1,
            key="hough_threshold",
            help="Minimum vote strength needed for a Hough line. Higher values keep only stronger, more obvious line candidates.",
        )
    with col2:
        hough_line_length = st.slider(
            "Hough line length",
            5,
            300,
            int(params.hough_line_length),
            5,
            key="hough_line_length",
            help="Minimum accepted Hough segment length in pixels. Higher values ignore short ridge fragments.",
        )
    with col3:
        hough_line_gap = st.slider(
            "Hough line gap",
            0,
            80,
            int(params.hough_line_gap),
            1,
            key="hough_line_gap",
            help="Maximum gap bridged while forming a Hough line. Higher values merge broken pieces into longer segments.",
        )
    latest_candidate_count = hough_candidate_count_from_result(stored_result)
    slider_max = latest_candidate_count if latest_candidate_count is not None else 10000
    slider_max = max(0, int(slider_max))

    st.markdown("**Hough Seed Selection**")
    col1, col2, col3 = st.columns(3)
    with col1:
        hough_seed_spacing = st.slider(
            "Hough seed spacing",
            1,
            200,
            int(params.hough_seed_spacing),
            1,
            key="hough_seed_spacing",
            help="Minimum spacing between selected Hough seed pixels. Higher values reduce duplicate seeds on the same event.",
        )
    with col2:
        hough_use_all_seeds = st.checkbox(
            "Use all Hough seed candidates",
            value=bool(params.hough_use_all_seeds),
            key="hough_use_all_seeds",
            help=(
                "Keeps the Hough seed cap at the latest detected candidate count. "
                "This is the default so Hough seeds are not accidentally capped."
            ),
        )
    with col3:
        hough_max_seeds = hough_max_seed_slider(
            slider_max=slider_max,
            current_value=params.hough_max_seeds,
            use_all=hough_use_all_seeds,
        )
    if latest_candidate_count is None:
        st.caption("Run Hough detection once to size the Max Hough seeds slider from the detected candidate count.")
    else:
        st.caption(f"Latest Hough candidate seed count: {latest_candidate_count}")
    return replace(
        params,
        hough_threshold=hough_threshold,
        hough_line_length=hough_line_length,
        hough_line_gap=hough_line_gap,
        hough_seed_spacing=hough_seed_spacing,
        hough_max_seeds=hough_max_seeds,
        hough_use_all_seeds=hough_use_all_seeds,
    )


def hough_event_tracing_controls(params: PipelineParams) -> PipelineParams:
    st.markdown("**Event Growth Controls**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        intensity_difference_tolerance = st.slider(
            "Intensity difference tolerance",
            0,
            255,
            int(params.intensity_difference_tolerance),
            1,
            key="hough_trace_intensity_difference_tolerance",
            help="Allowed intensity difference while growing an event from a seed. Higher values let events grow through more varied pixel intensities.",
        )
    with col2:
        best_fit_line_tolerance = st.slider(
            "Best fit line tolerance",
            0.5,
            50.0,
            float(params.best_fit_line_tolerance),
            0.5,
            key="hough_trace_best_fit_line_tolerance",
            help="Maximum distance from growing event pixels to the fitted line. Higher values allow wider or more curved events.",
        )
    with col3:
        min_intensity = st.slider(
            "Minimum intensity",
            0,
            255,
            int(params.min_intensity),
            1,
            key="hough_trace_min_intensity",
            help="Minimum processed intensity required for event growth. Higher values keep only brighter ridge pixels.",
        )
    with col4:
        min_points_threshold = st.slider(
            "Minimum event pixels",
            1,
            200,
            int(params.min_points_threshold),
            1,
            key="hough_trace_min_points_threshold",
            help="Minimum number of pixels needed for a grown line to be accepted as an event.",
        )

    st.markdown("**Event Merge Controls**")
    col1, col2, col3 = st.columns(3)
    with col1:
        duplicate_overlap = st.slider(
            "Merge overlap fraction",
            0.1,
            1.0,
            float(params.duplicate_overlap),
            0.05,
            key="hough_trace_duplicate_overlap",
            help="Overlap fraction above which a newly grown event is merged into an accepted event.",
        )
    with col2:
        merge_distance_tolerance = st.slider(
            "Merge distance tolerance",
            0.0,
            30.0,
            float(params.merge_distance_tolerance),
            0.5,
            key="hough_trace_merge_distance_tolerance",
            help="Maximum nearest-pixel distance for merging two grown events when their orientations are similar.",
        )
    with col3:
        merge_angle_tolerance = st.slider(
            "Merge angle tolerance",
            0.0,
            45.0,
            float(params.merge_angle_tolerance),
            1.0,
            key="hough_trace_merge_angle_tolerance",
            help="Maximum angle difference, in degrees, for merging nearby grown events.",
        )
    connect_merged_event_gaps = st.checkbox(
        "Connect merged event gaps with endpoint bridges",
        value=bool(params.connect_merged_event_gaps),
        key="hough_trace_connect_merged_event_gaps",
        help=(
            "When merged events contain disconnected components, add the thinnest "
            "endpoint-to-endpoint bridge pixels needed to make the merged event connected."
        ),
    )

    return replace(
        params,
        intensity_difference_tolerance=intensity_difference_tolerance,
        best_fit_line_tolerance=best_fit_line_tolerance,
        min_intensity=min_intensity,
        min_points_threshold=min_points_threshold,
        duplicate_overlap=duplicate_overlap,
        merge_distance_tolerance=merge_distance_tolerance,
        merge_angle_tolerance=merge_angle_tolerance,
        connect_merged_event_gaps=connect_merged_event_gaps,
    )


def hough_candidate_count_from_result(result: dict | None) -> int | None:
    if not result:
        return None
    hough = result.get("hough")
    if not isinstance(hough, dict):
        return None
    count = hough.get("candidate_seed_count")
    if count is None:
        return None
    return max(0, int(count))


def hough_max_seed_slider(slider_max: int, current_value: int, use_all: bool) -> int:
    if slider_max <= 0:
        st.slider(
            "Max Hough seeds",
            0,
            1,
            0,
            1,
            key="hough_max_seeds_empty",
            disabled=True,
            help="No Hough seed candidates were found in the latest result.",
        )
        return 0

    if use_all:
        st.slider(
            "Max Hough seeds",
            1,
            slider_max,
            slider_max,
            1,
            key=f"hough_max_seeds_auto_{slider_max}",
            disabled=True,
            help="Automatically set to the latest detected Hough candidate seed count.",
        )
        return slider_max

    manual_key = "hough_max_seeds"
    if manual_key in st.session_state:
        st.session_state[manual_key] = max(1, min(int(st.session_state[manual_key]), slider_max))
    else:
        st.session_state[manual_key] = max(1, min(int(current_value), slider_max))
    return st.slider(
        "Max Hough seeds",
        1,
        slider_max,
        int(st.session_state[manual_key]),
        1,
        key=manual_key,
        help="Upper limit on how many Hough seeds are sent into the event-growing algorithm.",
    )


def regionprops_candidate_count_from_result(result: dict | None) -> int | None:
    if not result:
        return None
    regionprops = result.get("regionprops")
    if not isinstance(regionprops, dict):
        return None
    count = regionprops.get("candidate_seed_count")
    if count is None:
        return None
    return max(0, int(count))


def regionprops_max_seed_slider(slider_max: int, current_value: int, use_all: bool) -> int:
    if slider_max <= 0:
        st.slider(
            "Max RegionProp seeds",
            0,
            1,
            0,
            1,
            key="region_max_seeds_empty",
            disabled=True,
            help="No RegionProp seed candidates were found in the latest result.",
        )
        return 0

    if use_all:
        st.slider(
            "Max RegionProp seeds",
            1,
            slider_max,
            slider_max,
            1,
            key=f"region_max_seeds_auto_{slider_max}",
            disabled=True,
            help="Automatically set to the latest detected RegionProp candidate seed count.",
        )
        return slider_max

    manual_key = "region_max_seeds"
    if manual_key in st.session_state:
        st.session_state[manual_key] = max(1, min(int(st.session_state[manual_key]), slider_max))
    else:
        st.session_state[manual_key] = max(1, min(int(current_value), slider_max))
    return st.slider(
        "Max RegionProp seeds",
        1,
        slider_max,
        int(st.session_state[manual_key]),
        1,
        key=manual_key,
        help="Upper limit on how many RegionProp seeds are sent into the event-growing algorithm.",
    )


def regionprops_method_controls(params: PipelineParams, stored_result: dict | None) -> PipelineParams:
    st.markdown("**RegionProp Controls**")
    latest_candidate_count = regionprops_candidate_count_from_result(stored_result)
    slider_max = latest_candidate_count if latest_candidate_count is not None else 10000
    slider_max = max(0, int(slider_max))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        region_min_pixels = st.slider(
            "Region min ridge pixels",
            1,
            500,
            int(params.region_min_pixels),
            5,
            key="region_min_pixels",
            help="Minimum connected ridge-mask area used as a RegionProp seed candidate. Higher values remove small isolated blobs.",
        )
    with col2:
        region_seed_spacing = st.slider(
            "RegionProp seed spacing",
            1,
            200,
            int(params.region_seed_spacing),
            1,
            key="region_seed_spacing",
            help="Minimum spacing between selected RegionProp seed pixels. Higher values reduce duplicate seeds on the same event.",
        )
    with col3:
        region_use_all_seeds = st.checkbox(
            "Use all RegionProp seed candidates",
            value=bool(params.region_use_all_seeds),
            key="region_use_all_seeds",
            help=(
                "Keeps the RegionProp seed cap at the latest detected candidate count. "
                "This is the default so RegionProp seeds are not accidentally capped."
            ),
        )
    with col4:
        region_max_seeds = regionprops_max_seed_slider(
            slider_max=slider_max,
            current_value=params.region_max_seeds,
            use_all=region_use_all_seeds,
        )
    if latest_candidate_count is None:
        st.caption("Run RegionProp detection once to size the Max RegionProp seeds slider from the detected candidate count.")
    else:
        st.caption(f"Latest RegionProp candidate seed count: {latest_candidate_count}")
    return replace(
        params,
        region_min_pixels=region_min_pixels,
        region_seed_spacing=region_seed_spacing,
        region_max_seeds=region_max_seeds,
        region_use_all_seeds=region_use_all_seeds,
    )


def show_detection_mask_metrics(result: dict) -> None:
    ridge = result["ridge"]
    params = result["params"]
    crop_height, crop_width = result["crop"]["display_rgb"].shape[:2]
    cols = st.columns(5)
    cols[0].metric("Candidate ridge pixels", int(ridge["candidate_mask"].sum()))
    cols[1].metric("Clean mask pixels", int(ridge["candidate_clean"].sum()))
    cols[2].metric("Seed mask pixels", int(ridge["detection_mask"].sum()))
    cols[3].metric("Seed mask coverage", f"{ridge['detection_mask'].mean() * 100:.2f}%")
    cols[4].metric("Ridge threshold", f"{ridge['ridge_threshold']:.4f}")

    area_label = "Full image" if params.use_full_image else "Crop"
    st.caption(
        f"{area_label}: origin x,y {result['crop']['origin']} | "
        f"size {crop_width} x {crop_height} | {result['crop']['note']}"
    )


def show_visualizations(result: dict) -> None:
    tab_hough, tab_region = st.tabs(["Hough Line Detection", "RegionProp Seed Detection"])
    with tab_hough:
        show_hough_method_view(result)
    with tab_region:
        show_regionprops_method_view(result)


def show_preprocess_view(result: dict) -> None:
    crop = result["crop"]
    ridge = result["ridge"]
    raw_normalized_rgb = crop["raw_normalized_rgb"]
    display_rgb = crop["display_rgb"]
    stats = crop["raw_stats"]

    col1, col2, col3 = st.columns(3)
    col1.image(
        to_display(raw_normalized_rgb),
        caption=(
            "Raw linear 0-255 normalized "
            f"(min {stats['min']:.4g}, max {stats['max']:.4g})"
        ),
        width='stretch',
    )
    col2.image(to_display(display_rgb), caption="Loaded BLN crop, current brightness/contrast", width='stretch')
    col3.image(to_display(ridge["enhanced"], cmap="gray"), caption="CLAHE enhanced", width='stretch')

    col1, col2, col3, col4 = st.columns(4)
    col1.image(to_display(ridge["ridges"], cmap="magma"), caption="Ridge response", width='stretch')
    col2.image(to_display(ridge["candidate_mask"], cmap="gray"), caption="Threshold ridge mask", width='stretch')
    col3.image(to_display(ridge["candidate_clean"], cmap="gray"), caption="Clean detection mask", width='stretch')
    final_caption = "Skeletonized seed mask" if ridge.get("use_skeletonize") else "Clean mask used for seed detection"
    col4.image(
        to_display(ridge.get("display_mask", ridge["detection_mask"]), cmap="gray"),
        caption=final_caption,
        width='stretch',
    )


def show_hough_method_view(result: dict) -> None:
    show_hough_step1_result(result)
    if "hough_events" not in result:
        st.info("Step 1 is complete. Press Step 2 to trace events from these selected Hough seeds.")
        return
    st.divider()
    show_hough_step2_result(result)


def show_hough_step1_result(result: dict) -> None:
    display_rgb = result["crop"]["display_rgb"]
    st.markdown("**Step 1 Result: Hough Lines And Seeds**")
    cols = st.columns(4)
    cols[0].metric("Hough lines", result["hough"]["raw_count"])
    cols[1].metric("Candidate seeds", result["hough"]["candidate_seed_count"])
    cols[2].metric("Selected seeds", len(result["hough"]["seeds"]))
    cols[3].metric("Max seeds", result["hough"]["max_seeds"])

    seed_overlay = draw_hough_overlay(display_rgb, result["hough"]["lines"], result["hough"]["seeds"])
    zoomable_image(seed_overlay, "Hough lines with selected seeds", key="hough_seed_lines")



def show_hough_step2_result(result: dict) -> None:
    display_rgb = result["crop"]["display_rgb"]
    hough_events = result["hough_events"]
    st.markdown("**Step 2 Result: Events Traced From Hough Seeds**")
    event_pixel_count = sum(line.size for line in hough_events["accepted"])
    cols = st.columns(4)
    cols[0].metric("Detected events", hough_events["accepted_count"])
    cols[1].metric("Event pixels", event_pixel_count)
    cols[2].metric("Rejected seeds", hough_events["rejected"])
    cols[3].metric("Merged candidates", hough_events.get("merged", 0))

    event_overlay = draw_event_overlay(
        display_rgb,
        hough_events["accepted"],
        hough_events.get("standalone_seeds", result["hough"]["seeds"]),
        hough_events.get("merged_seeds", []),
        hough_events.get("merged_groups", []),
    )

    zoomable_image(event_overlay, "BLN events grown from Hough seeds", key="hough_events")

    show_grx_overlay_view(
        result=result,
        event_result=hough_events,
        title="BLN Hough events over GRX",
        key="grx_hough_events",
    )
    show_hough_event_save_controls(result)


def show_regionprops_method_view(result: dict) -> None:
    display_rgb = result["crop"]["display_rgb"]
    show_method_counts(
        candidate_label="RegionProp ridges",
        candidate_count=result["regionprops"]["raw_count"],
        candidate_seed_count=result["regionprops"]["candidate_seed_count"],
        seed_count=len(result["regionprops"]["seeds"]),
        event_result=result["regionprops_events"],
    )

    seed_overlay = draw_seed_overlay(
        display_rgb,
        result["ridge"]["detection_mask"],
        result["regionprops"]["seeds"],
    )
    event_overlay = draw_event_overlay(
        display_rgb,
        result["regionprops_events"]["accepted"],
        result["regionprops_events"].get("standalone_seeds", result["regionprops"]["seeds"]),
        result["regionprops_events"].get("merged_seeds", []),
        result["regionprops_events"].get("merged_groups", []),
    )

    col1, col2 = st.columns(2)
    with col1:
        zoomable_image(seed_overlay, "RegionProp ridge mask with selected seeds", key="region_seed_mask")
    with col2:
        zoomable_image(event_overlay, "BLN events grown from RegionProp seeds", key="region_events")

    show_grx_overlay_view(
        result=result,
        event_result=result["regionprops_events"],
        title="BLN RegionProp events over GRX",
        key="grx_region_events",
    )


def show_method_counts(
    candidate_label: str,
    candidate_count: int,
    candidate_seed_count: int,
    seed_count: int,
    event_result: dict,
) -> None:
    cols = st.columns(6)
    cols[0].metric(candidate_label, candidate_count)
    cols[1].metric("Candidate seeds", candidate_seed_count)
    cols[2].metric("Selected seeds", seed_count)
    cols[3].metric("Detected events", event_result["accepted_count"])
    cols[4].metric("Rejected seeds", event_result["rejected"])
    cols[5].metric("Merged candidates", event_result.get("merged", 0))

    if event_result["covered"]:
        st.caption(f"{event_result['covered']} seeds were skipped because they were already covered by an accepted event.")


def show_hough_event_save_controls(result: dict) -> None:
    st.divider()
    st.markdown("**Save Hough Events**")

    hough_events = result["hough_events"]["accepted"]
    if not hough_events:
        st.info("No grown Hough events are available to save yet.")
        return

    default_prefix = default_hough_event_prefix(result)
    prefix = st.text_input(
        "Output file prefix",
        value=default_prefix,
        key="hough_event_save_prefix",
        help=(
            "Used to create <prefix>_events.csv and <prefix>_event_pixels.csv "
            "under tests/outputs/hough_events."
        ),
    )
    st.caption(
        "The event pixel file stores global image coordinates, so saved events can be "
        "reloaded later for overlays or curvature analysis."
    )

    if st.button("Save Hough events", key="save_hough_events"):
        try:
            save_result = save_hough_events_csv(result, prefix)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success(
            f"Saved {save_result['event_count']} Hough events and "
            f"{save_result['pixel_count']} unique event pixels."
        )
        if save_result["duplicate_pixel_count"]:
            st.caption(
                f"Removed {save_result['duplicate_pixel_count']} duplicate pixel rows that were shared "
                "by multiple events. The first saved event kept each shared pixel."
            )
        if save_result["skipped_event_count"]:
            st.caption(
                f"Skipped {save_result['skipped_event_count']} events because all of their pixels "
                "were already assigned to earlier events."
            )
        st.code(
            "\n".join(str(path) for path in save_result["paths"]),
            language="text",
        )


def default_hough_event_prefix(result: dict) -> str:
    params = result["params"]
    crop = result["crop"]
    origin_x, origin_y = crop["origin"]
    height, width = crop["display_rgb"].shape[:2]
    image_stem = Path(params.image_path).stem
    return f"{image_stem}_hough_x{origin_x}_y{origin_y}_w{width}_h{height}"


def save_hough_events_csv(result: dict, prefix: str) -> dict:
    safe_prefix = sanitize_file_prefix(prefix)
    if not safe_prefix:
        raise ValueError("Please provide a file prefix before saving.")

    out_dir = ROOT / "tests" / "outputs" / "hough_events"
    out_dir.mkdir(parents=True, exist_ok=True)

    events_path = out_dir / f"{safe_prefix}_events.csv"
    pixels_path = out_dir / f"{safe_prefix}_event_pixels.csv"
    crop = result["crop"]
    params = result["params"]
    crop_x, crop_y = crop["origin"]
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

    event_rows = []
    pixel_rows = []
    assigned_pixels: set[tuple[int, int]] = set()
    original_pixel_count = 0
    duplicate_pixel_count = 0
    skipped_event_count = 0
    saved_event_index = 0

    for line in result["hough_events"]["accepted"]:
        points = sorted(line.points, key=lambda point: (point.y, point.x))
        original_pixel_count += len(points)
        unique_points = []
        for point in points:
            global_pixel = (crop_x + int(point.x), crop_y + int(point.y))
            if global_pixel in assigned_pixels:
                duplicate_pixel_count += 1
                continue
            assigned_pixels.add(global_pixel)
            unique_points.append(global_pixel)

        if not unique_points:
            skipped_event_count += 1
            continue

        saved_event_index += 1
        event_id = f"{safe_prefix}_{saved_event_index:04d}"
        event_rows.append(
            {
                "event_id": event_id,
                "method": "hough",
                "num_pixels": len(unique_points),
                "crop_x": crop_x,
                "crop_y": crop_y,
                "image_path": params.image_path,
                "created_at": created_at,
            }
        )
        for pixel_x, pixel_y in unique_points:
            pixel_rows.append(
                {
                    "event_id": event_id,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                }
            )

    with events_path.open("w", newline="") as events_file:
        writer = csv.DictWriter(
            events_file,
            fieldnames=[
                "event_id",
                "method",
                "num_pixels",
                "crop_x",
                "crop_y",
                "image_path",
                "created_at",
            ],
        )
        writer.writeheader()
        writer.writerows(event_rows)

    with pixels_path.open("w", newline="") as pixels_file:
        writer = csv.DictWriter(
            pixels_file,
            fieldnames=["event_id", "pixel_x", "pixel_y"],
        )
        writer.writeheader()
        writer.writerows(pixel_rows)

    return {
        "event_count": len(event_rows),
        "pixel_count": len(pixel_rows),
        "original_pixel_count": original_pixel_count,
        "duplicate_pixel_count": duplicate_pixel_count,
        "skipped_event_count": skipped_event_count,
        "paths": [events_path, pixels_path],
    }


def sanitize_file_prefix(prefix: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix.strip())
    safe = safe.strip("._-")
    return safe


def save_uploaded_streamlit_file(uploaded_file, folder_name: str) -> Path:
    out_dir = ROOT / "tests" / "outputs" / "streamlit_uploads" / sanitize_file_prefix(folder_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_file_prefix(Path(uploaded_file.name).stem) or "uploaded"
    suffix = Path(uploaded_file.name).suffix.lower()
    out_path = out_dir / f"{safe_name}{suffix}"
    uploaded_bytes = uploaded_file.getbuffer()
    if out_path.exists() and out_path.stat().st_size == len(uploaded_bytes):
        with out_path.open("rb") as f:
            if f.read() == uploaded_bytes:
                return out_path
    with out_path.open("wb") as f:
        f.write(uploaded_bytes)
    return out_path


def save_trace_comparison_summary(score_result: pd.DataFrame, prefix: str) -> Path:
    out_dir = ROOT / "tests" / "outputs" / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = sanitize_file_prefix(prefix) or "event_trace_comparison"
    out_path = out_dir / f"{safe_prefix}.csv"
    summary = trace_comparison_summary_frame(score_result)
    summary.to_csv(out_path, index=False)
    return out_path


def save_event_score_category_csvs(
    score_result: pd.DataFrame,
    morphology_result: pd.DataFrame | None,
    events: pd.DataFrame,
    event_pixels: pd.DataFrame,
    prefix: str,
) -> list[Path]:
    out_dir = ROOT / "tests" / "outputs" / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = sanitize_file_prefix(prefix) or "event_score_categories"
    saved_paths = []
    category_specs = [
        ("matched", "likely_real"),
        ("uncertain", "uncertain"),
        ("noise", "likely_noise"),
    ]
    for category, classification in category_specs:
        frame = score_result[score_result["classification"].astype(str) == classification]
        category_ids = set(frame["event_id"].astype(str)) if "event_id" in frame.columns else set()
        saved_paths.extend(
            save_event_category_source_files(
                events,
                event_pixels,
                category_ids,
                out_dir,
                safe_prefix,
                category,
            )
        )

    if morphology_result is not None and "event_type" in morphology_result.columns:
        blob_frame = morphology_result[morphology_result["event_type"].astype(str) == "blob_like"].copy()
    else:
        blob_frame = pd.DataFrame(columns=["event_id", "event_type"])
    blob_ids = set(blob_frame["event_id"].astype(str)) if "event_id" in blob_frame.columns else set()
    saved_paths.extend(
        save_event_category_source_files(
            events,
            event_pixels,
            blob_ids,
            out_dir,
            safe_prefix,
            "blob",
        )
    )
    return saved_paths


def save_event_category_source_files(
    events: pd.DataFrame,
    event_pixels: pd.DataFrame,
    event_ids: set[str],
    out_dir: Path,
    prefix: str,
    category: str,
) -> list[Path]:
    events_identity_col = event_identity_column(events)
    pixels_identity_col = event_identity_column(event_pixels)
    event_mask = events[events_identity_col].astype(str).isin(event_ids)
    pixel_mask = event_pixels[pixels_identity_col].astype(str).isin(event_ids)
    category_events = events.loc[event_mask].copy()
    category_pixels = event_pixels.loc[pixel_mask].copy()
    category_pixels = order_detection_analysis_pixel_columns(category_pixels)

    events_path = out_dir / f"{prefix}_{category}_events.csv"
    pixels_path = out_dir / f"{prefix}_{category}_event_pixels.csv"
    category_events.to_csv(events_path, index=False)
    category_pixels.to_csv(pixels_path, index=False)
    return [events_path, pixels_path]


def order_detection_analysis_pixel_columns(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["event_id", "pixel_x", "pixel_y"]
    ordered = [col for col in required if col in frame.columns]
    ordered.extend([col for col in frame.columns if col not in ordered])
    return frame.loc[:, ordered].copy()


def save_alignment_overlay_images(
    classification_overlay: np.ndarray,
    vector_overlay: np.ndarray,
    all_trace_overlay: np.ndarray,
    prefix: str,
) -> list[Path]:
    out_dir = ROOT / "tests" / "outputs" / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = sanitize_file_prefix(prefix) or "event_trace_comparison"
    outputs = [
        (out_dir / f"{safe_prefix}_classification_overlay.png", classification_overlay),
        (out_dir / f"{safe_prefix}_vector_overlay.png", vector_overlay),
        (out_dir / f"{safe_prefix}_all_trace_vectors_overlay.png", all_trace_overlay),
    ]
    saved = []
    for path, image in outputs:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
        saved.append(path)
    return saved


def build_event_trace_plot_zip(
    score_result: pd.DataFrame,
    event_pixels: pd.DataFrame,
    prefix: str,
) -> tuple[bytes, str]:
    safe_prefix = sanitize_file_prefix(prefix) or "event_trace_comparison"
    buffer = BytesIO()
    score_identity_col = event_identity_column(score_result)
    pixel_identity_col = event_identity_column(event_pixels)
    pixels = event_pixels.copy()
    pixels[pixel_identity_col] = pixels[pixel_identity_col].astype(str)

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for _, row in score_result.iterrows():
            event_id = str(row[score_identity_col])
            event_subset = pixels[pixels[pixel_identity_col] == event_id]
            if event_subset.empty:
                continue
            image = build_single_event_trace_plot(row, event_subset)
            image_buffer = BytesIO()
            image.save(image_buffer, format="PNG")
            zf.writestr(f"{sanitize_file_prefix(event_id) or 'event'}.png", image_buffer.getvalue())

    return buffer.getvalue(), f"{safe_prefix}_event_trace_plots.zip"


def build_single_event_trace_plot(row: pd.Series, event_pixels: pd.DataFrame) -> Image.Image:
    xs = event_pixels["pixel_x"].to_numpy(dtype=np.int64)
    ys = event_pixels["pixel_y"].to_numpy(dtype=np.int64)
    margin = 50
    x0 = int(xs.min()) - margin
    y0 = int(ys.min()) - margin
    x1 = int(xs.max()) + margin + 1
    y1 = int(ys.max()) + margin + 1
    width = max(520, x1 - x0)
    height = max(160, y1 - y0)
    if width == 520:
        x0 = int(round(float(row.get("event_center_x", xs.mean())))) - width // 2
    if height == 160:
        y0 = int(round(float(row.get("event_center_y", ys.mean())))) - height // 2

    text_panel_height = 275
    image = Image.new("RGB", (width, height + text_panel_height), "white")
    draw = ImageDraw.Draw(image)
    plot_h = height

    for pixel_x, pixel_y in zip(xs, ys, strict=False):
        lx = int(pixel_x - x0)
        ly = int(pixel_y - y0)
        if 0 <= lx < width and 0 <= ly < plot_h:
            draw.rectangle([lx - 1, ly - 1, lx + 1, ly + 1], fill=(220, 0, 0))

    center = np.array(
        [
            float(row.get("event_center_x", np.mean(xs))),
            float(row.get("event_center_y", np.mean(ys))),
        ],
        dtype=np.float64,
    )
    event_length = float(row.get("event_length_pixels", max(width, height) / 2))
    half_length = max(20.0, 0.65 * event_length)

    draw_vector_line(draw, center, vector_from_row(row, "event_vector_dx", "event_vector_dy"), x0, y0, half_length, (0, 210, 230), 4, width, plot_h)
    for spec in trace_vectors_from_row(row):
        draw_vector_line(
            draw,
            center,
            spec["direction"],
            x0,
            y0,
            half_length,
            trace_vector_color(spec),
            2,
            width,
            plot_h,
        )
    draw_vector_line(draw, center, vector_from_row(row, "best_trace_dx", "best_trace_dy"), x0, y0, half_length, (255, 0, 255), 3, width, plot_h)

    cx = float(center[0] - x0)
    cy = float(center[1] - y0)
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(0, 0, 0))

    draw.line([(0, height), (width, height)], fill=(180, 180, 180), width=1)
    label_y = height + 8
    text_lines = [
        f"event_id: {row.get('event_id', '')}",
        f"classification: {row.get('classification', '')} | best: {row.get('best_mode', '')} | score: {row.get('score', '')}",
        f"event center: ({row.get('event_center_x', '')}, {row.get('event_center_y', '')})",
        f"event vector: {format_row_vector(row, 'event_vector_dx', 'event_vector_dy')} | event angle: {row.get('event_angle_deg', '')} deg",
        f"Euler angles: phi1={row.get('nearest_phi1', '')}, Phi={row.get('nearest_Phi', '')}, phi2={row.get('nearest_phi2', '')}",
        f"nearest ANG EBSD: ({row.get('nearest_ang_x_ebsd', '')}, {row.get('nearest_ang_y_ebsd', '')})",
        f"best trace mode: {row.get('best_trace_mode', '')} | kind: {row.get('best_trace_kind', '')}",
        f"all trace labels: {row.get('all_trace_labels', '')}",
        f"all trace vectors: {row.get('all_trace_vectors', '')}",
        f"all angle errors deg: {row.get('all_trace_angle_errors_deg', '')}",
        "vectors/angles are Cartesian: +x right, +y up, positive angle counterclockwise",
        "colors: red pixels = event | cyan = fitted line | teal = slip | green = twin/basal | magenta = selected trace",
    ]
    for line in text_lines:
        for wrapped in wrap_text_for_image(line, max_chars=max(48, width // 7)):
            draw.text((8, label_y), wrapped, fill=(0, 0, 0))
            label_y += 17

    return image


def format_row_vector(row: pd.Series, dx_col: str, dy_col: str) -> str:
    vector = vector_from_row(row, dx_col, dy_col)
    if vector is None:
        return ""
    return f"[{vector[0]:.6f}, {vector[1]:.6f}]"


def wrap_text_for_image(text: str, max_chars: int) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def vector_from_row(row: pd.Series, dx_col: str, dy_col: str) -> np.ndarray | None:
    if dx_col not in row or dy_col not in row:
        return None
    try:
        vector = np.array([float(row[dx_col]), float(row[dy_col])], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(vector).all():
        return None
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return None
    return vector / norm


def trace_vectors_from_row(row: pd.Series) -> list[dict]:
    vectors = parse_json_cell(row.get("all_trace_vectors", ""))
    labels = parse_json_cell(row.get("all_trace_labels", ""))
    modes = parse_json_cell(row.get("all_trace_modes", ""))
    kinds = parse_json_cell(row.get("all_trace_kinds", ""))
    if isinstance(vectors, list) and vectors:
        specs = []
        for index, vector in enumerate(vectors):
            try:
                direction = np.asarray(vector, dtype=np.float64)[:2]
            except (TypeError, ValueError):
                continue
            norm = np.linalg.norm(direction)
            if norm <= 1e-12 or not np.isfinite(direction).all():
                continue
            specs.append(
                {
                    "label": str(labels[index]) if isinstance(labels, list) and index < len(labels) else f"trace_{index + 1}",
                    "mode": str(modes[index]) if isinstance(modes, list) and index < len(modes) else "",
                    "kind": str(kinds[index]) if isinstance(kinds, list) and index < len(kinds) else "slip",
                    "direction": direction / norm,
                }
            )
        if specs:
            return specs

    fallback_specs = []
    for label, dx_col, dy_col, kind in [
        ("prism1", "prism1_dx", "prism1_dy", "slip"),
        ("prism2", "prism2_dx", "prism2_dy", "slip"),
        ("prism3", "prism3_dx", "prism3_dy", "slip"),
        ("basal", "basal_dx", "basal_dy", "slip"),
    ]:
        direction = vector_from_row(row, dx_col, dy_col)
        if direction is not None:
            fallback_specs.append({"label": label, "mode": label, "kind": kind, "direction": direction})
    return fallback_specs


def parse_json_cell(value):
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def trace_vector_color(spec: dict) -> tuple[int, int, int]:
    kind = str(spec.get("kind", "")).lower()
    label = str(spec.get("label", "")).lower()
    if kind == "twin" or "twin" in label:
        return (0, 210, 70)
    if "basal" in label:
        return (0, 230, 70)
    return (0, 190, 190)


def draw_vector_line(
    draw: ImageDraw.ImageDraw,
    center: np.ndarray,
    direction: np.ndarray | None,
    x0: int,
    y0: int,
    half_length: float,
    color: tuple[int, int, int],
    width_px: int,
    image_width: int,
    image_height: int,
) -> None:
    if direction is None:
        return
    # Saved/scored vectors use Cartesian +y-up convention; PIL/image drawing
    # uses pixel +y-down convention, so flip dy only for rendering.
    image_direction = cartesian_vector_to_image(direction)
    start = center - image_direction * half_length
    end = center + image_direction * half_length
    x_start = float(start[0] - x0)
    y_start = float(start[1] - y0)
    x_end = float(end[0] - x0)
    y_end = float(end[1] - y0)
    if (
        max(x_start, x_end) < 0
        or min(x_start, x_end) >= image_width
        or max(y_start, y_end) < 0
        or min(y_start, y_end) >= image_height
    ):
        return
    draw.line([(x_start, y_start), (x_end, y_end)], fill=color, width=max(1, int(width_px)))


def trace_comparison_summary_frame(score_result: pd.DataFrame) -> pd.DataFrame:
    summary = score_result.copy()
    vector_pairs = {
        "event_vector": ("event_vector_dx", "event_vector_dy"),
        "best_trace_vector": ("best_trace_dx", "best_trace_dy"),
        "prism1_vector": ("prism1_dx", "prism1_dy"),
        "prism2_vector": ("prism2_dx", "prism2_dy"),
        "prism3_vector": ("prism3_dx", "prism3_dy"),
        "basal_vector": ("basal_dx", "basal_dy"),
    }
    for out_col, (dx_col, dy_col) in vector_pairs.items():
        if {dx_col, dy_col}.issubset(summary.columns):
            summary[out_col] = [
                format_vector_pair(dx, dy)
                for dx, dy in zip(summary[dx_col], summary[dy_col], strict=False)
            ]

    preferred_columns = [
        "event_id",
        "original_event_id",
        "cut_index",
        "classification",
        "score",
        "num_pixels",
        "event_center_x",
        "event_center_y",
        "event_angle_deg",
        "event_vector",
        "nearest_phi1",
        "nearest_Phi",
        "nearest_phi2",
        "best_mode",
        "best_trace_mode",
        "best_trace_kind",
        "best_trace_angle_deg",
        "best_trace_vector",
        "all_trace_labels",
        "all_trace_modes",
        "all_trace_kinds",
        "all_trace_vectors",
        "all_trace_angle_errors_deg",
        "all_trace_tolerances_deg",
        "prism1_vector",
        "prism2_vector",
        "prism3_vector",
        "basal_vector",
        "nearest_ang_x_dic",
        "nearest_ang_y_dic",
        "nearest_ang_x_ebsd",
        "nearest_ang_y_ebsd",
        "center_lookup_distance",
        "best_angle_error_deg",
        "angle_to_prism1_deg",
        "angle_to_prism2_deg",
        "angle_to_prism3_deg",
        "angle_to_basal_deg",
        "passes_angle",
        "passes_lookup",
        "passes_linearity",
        "linearity",
        "event_length_pixels",
    ]
    component_columns = {
        "event_vector_dx",
        "event_vector_dy",
        "best_trace_dx",
        "best_trace_dy",
        "prism1_dx",
        "prism1_dy",
        "prism2_dx",
        "prism2_dy",
        "prism3_dx",
        "prism3_dy",
        "basal_dx",
        "basal_dy",
    }
    available_columns = [col for col in preferred_columns if col in summary.columns]
    remaining_columns = [
        col
        for col in summary.columns
        if col not in available_columns and col not in component_columns
    ]
    return summary[available_columns + remaining_columns].copy()


def format_vector_pair(dx, dy) -> str:
    try:
        dx_val = float(dx)
        dy_val = float(dy)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite([dx_val, dy_val]).all():
        return ""
    return f"[{dx_val:.6f}, {dy_val:.6f}]"


@st.cache_data(show_spinner=False)
def load_alignment_transform_parameters(path_str: str) -> dict:
    with Path(path_str).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Transformation JSON must contain an object at the top level.")

    forward_params = np.asarray(data.get("forward_transform_ebsd_to_dic", {}).get("params", []), dtype=float)
    inverse_params = np.asarray(data.get("inverse_map_dic_to_ebsd", {}).get("params", []), dtype=float)
    if forward_params.shape[0] != 2 or inverse_params.shape[0] != 2:
        raise ValueError("Transformation JSON does not contain valid forward and inverse parameter matrices.")

    terms = data.get("forward_polynomial_ebsd_to_dic", {}).get("terms") or polynomial_terms_from_count(forward_params.shape[1])
    return {
        "metadata": {
            "file": str(path_str),
            "transform_family": data.get("transform_family"),
            "polynomial_order": data.get("polynomial_order"),
            "matched_points": len(data.get("matched_point_ids") or []),
            "rmse_forward_pixels": format_optional_float(data.get("rmse_forward_pixels")),
            "rmse_inverse_pixels": format_optional_float(data.get("rmse_inverse_pixels")),
            "dic_image_shape": data.get("dic_image_shape"),
            "ebsd_image_shape": data.get("ebsd_image_shape"),
        },
        "forward_params": forward_params,
        "inverse_params": inverse_params,
        "terms": terms,
        "forward_coefficients": transform_coefficients_table(terms, forward_params, "x_dic", "y_dic"),
        "inverse_coefficients": transform_coefficients_table(terms, inverse_params, "x_ebsd", "y_ebsd"),
    }


def polynomial_terms_from_count(count: int) -> list[str]:
    if count == 6:
        return ["1", "x", "y", "x^2", "x*y", "y^2"]
    return [f"term_{index}" for index in range(count)]


def transform_coefficients_table(
    terms: list[str],
    params: np.ndarray,
    x_output_name: str,
    y_output_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "term": terms,
            x_output_name: params[0, : len(terms)],
            y_output_name: params[1, : len(terms)],
        }
    )


def format_optional_float(value) -> str:
    if value is None:
        return "missing"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def file_cache_signature(path_str: str) -> tuple[int, int]:
    stat = Path(path_str).stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


@st.cache_data(show_spinner=False)
def load_alignment_event_data(
    events_path: str,
    event_pixels_path: str,
    events_signature: tuple[int, int] | None = None,
    event_pixels_signature: tuple[int, int] | None = None,
) -> dict:
    _ = (events_signature, event_pixels_signature)
    events = pd.read_csv(events_path)
    event_pixels = pd.read_csv(event_pixels_path)
    if "event_id" not in events.columns:
        raise ValueError("Events CSV must include an event_id column.")
    required_pixels = {"event_id", "pixel_x", "pixel_y"}
    if not required_pixels.issubset(event_pixels.columns):
        raise ValueError("Event pixels CSV must include event_id, pixel_x, and pixel_y columns.")

    events = events.copy()
    event_pixels = event_pixels.copy()
    events["event_id"] = events["event_id"].astype(str)
    event_pixels["event_id"] = event_pixels["event_id"].astype(str)
    event_pixels["pixel_x"] = event_pixels["pixel_x"].astype(np.int64)
    event_pixels["pixel_y"] = event_pixels["pixel_y"].astype(np.int64)

    pixel_counts = event_pixels.groupby("event_id", sort=False).size()
    if "num_pixels" not in events.columns:
        events = events.merge(pixel_counts.rename("num_pixels"), on="event_id", how="left")
    events["num_pixels"] = events["num_pixels"].fillna(0).astype(np.int64)

    return {
        "events": events,
        "event_pixels": event_pixels,
        "metrics": {
            "event_count": int(events["event_id"].nunique()),
            "pixel_count": int(len(event_pixels)),
            "pixel_event_count": int(event_pixels["event_id"].nunique()),
            "min_pixels_per_event": int(pixel_counts.min()) if len(pixel_counts) else 0,
            "max_pixels_per_event": int(pixel_counts.max()) if len(pixel_counts) else 0,
        },
    }


def event_identity_column(frame: pd.DataFrame) -> str:
    return "event_id"


def score_events_with_trace_alignment(
    events: pd.DataFrame,
    event_pixels: pd.DataFrame,
    dic_coords: pd.DataFrame,
    forward_params: np.ndarray,
    angle_tolerance_deg: float,
    max_lookup_distance: float,
    min_linearity: float,
    crystal_config: dict | None = None,
) -> pd.DataFrame:
    _ = forward_params
    trace_points = dic_coords[
        ["x_dic", "y_dic", "x_ebsd", "y_ebsd", "phi1", "PHI", "phi2"]
    ].copy()
    finite = np.isfinite(trace_points[["x_dic", "y_dic", "x_ebsd", "y_ebsd"]].to_numpy(dtype=np.float64)).all(axis=1)
    trace_points = trace_points.loc[finite].reset_index(drop=True)
    if trace_points.empty:
        raise ValueError("No finite transformed ANG/DIC coordinates are available for scoring.")

    tree = cKDTree(trace_points[["x_dic", "y_dic"]].to_numpy(dtype=np.float64))
    event_col = event_identity_column(event_pixels)
    events_by_id = (
        events.set_index(events[event_identity_column(events)].astype(str), drop=False)
        if "event_id" in events.columns
        else pd.DataFrame()
    )
    rows = []

    for event_id, group in event_pixels.groupby(event_col, sort=False):
        segment_id = str(event_id)
        coords = group[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float64)
        num_pixels = int(len(coords))
        event_direction, linearity, event_length = event_direction_from_pixels(coords)

        if num_pixels == 0 or event_direction is None:
            rows.append(empty_event_score_row(segment_id, num_pixels, "too_few_pixels"))
            continue

        # Event pixels are image coordinates (+y is down). Slip trace vectors are
        # interpreted in a Cartesian analysis frame (+y is up), so flip dy before
        # angle comparisons and before saving vector/angle columns.
        event_direction_cart = image_vector_to_cartesian(event_direction)
        center = coords.mean(axis=0)
        center_distance, nearest_index = tree.query(center.reshape(1, -1), k=1)
        center_distance = float(np.asarray(center_distance)[0])
        nearest = trace_points.iloc[[int(np.asarray(nearest_index)[0])]]
        trace_info = trace_candidates_from_crystal_config(
            nearest[["phi1", "PHI", "phi2"]].to_numpy(dtype=np.float64).reshape(3),
            crystal_config,
            fallback_angle_tolerance=float(angle_tolerance_deg),
        )
        trace_candidates = normalize_2d_vectors(trace_info["vectors"][:, :2])
        angle_values = angular_mismatch_values(event_direction_cart, trace_candidates)
        best_index = select_ordered_trace_assignment_index(
            angle_values,
            trace_info["modes"],
            trace_info["tolerances"],
            trace_info["mode_priority"],
        )
        best_angle = float(angle_values[best_index])
        best_mode = str(trace_info["labels"][best_index])
        best_trace_mode = str(trace_info["modes"][best_index])
        best_trace_kind = str(trace_info["kinds"][best_index])
        assigned_tolerance = float(trace_info["tolerances"][best_index])
        best_trace = trace_candidates[best_index].astype(np.float64)
        best_trace_norm = np.linalg.norm(best_trace)
        if best_trace_norm > 1e-12:
            best_trace = best_trace / best_trace_norm
        else:
            best_trace = np.array([np.nan, np.nan], dtype=np.float64)

        passes_angle = best_angle <= assigned_tolerance
        passes_lookup = center_distance <= float(max_lookup_distance)
        passes_linearity = linearity >= float(min_linearity)
        classification = "likely_real" if passes_angle and passes_lookup and passes_linearity else "likely_noise"
        score = float(np.clip(1.0 - (best_angle / max(assigned_tolerance, 1e-9)), 0.0, 1.0))

        row = {
            "event_id": str(group["event_id"].iloc[0]),
            "segment_id": segment_id,
            "classification": classification,
            "score": round(score, 4),
            "best_mode": best_mode,
            "best_trace_mode": best_trace_mode,
            "best_trace_kind": best_trace_kind,
            "num_pixels": num_pixels,
            "event_center_x": round(float(center[0]), 3),
            "event_center_y": round(float(center[1]), 3),
            "nearest_ang_x_dic": round(float(nearest["x_dic"].iloc[0]), 3),
            "nearest_ang_y_dic": round(float(nearest["y_dic"].iloc[0]), 3),
            "nearest_ang_x_ebsd": round(float(nearest["x_ebsd"].iloc[0]), 3),
            "nearest_ang_y_ebsd": round(float(nearest["y_ebsd"].iloc[0]), 3),
            "nearest_phi1": round(float(nearest["phi1"].iloc[0]), 8),
            "nearest_Phi": round(float(nearest["PHI"].iloc[0]), 8),
            "nearest_phi2": round(float(nearest["phi2"].iloc[0]), 8),
            "center_lookup_distance": round(center_distance, 3),
            "event_angle_deg": round(vector_angle_degrees(event_direction_cart), 3),
            "event_vector_dx": round(float(event_direction_cart[0]), 6),
            "event_vector_dy": round(float(event_direction_cart[1]), 6),
            "best_trace_dx": round(float(best_trace[0]), 6),
            "best_trace_dy": round(float(best_trace[1]), 6),
            "best_trace_angle_deg": round(vector_angle_degrees(best_trace), 3) if np.isfinite(best_trace).all() else np.nan,
            "linearity": round(linearity, 4),
            "event_length_pixels": round(event_length, 3),
            "best_angle_error_deg": round(best_angle, 3),
            "angle_tolerance_deg": round(assigned_tolerance, 3),
            "passes_angle": bool(passes_angle),
            "passes_lookup": bool(passes_lookup),
            "passes_linearity": bool(passes_linearity),
        }
        row.update(trace_collection_columns(trace_info, trace_candidates, angle_values))
        row.update(legacy_trace_columns(trace_info["labels"], trace_candidates, angle_values))
        if not events_by_id.empty and segment_id in events_by_id.index:
            source_row = events_by_id.loc[segment_id]
            if isinstance(source_row, pd.DataFrame):
                source_row = source_row.iloc[0]
            for col in ["original_event_id", "cut_index", "method"]:
                if col in source_row:
                    row[col] = source_row[col]
        rows.append(row)

    result = pd.DataFrame(rows)
    if "score" in result.columns:
        result = result.sort_values(["classification", "best_angle_error_deg"], ascending=[True, True]).reset_index(drop=True)
    return result


def event_direction_from_pixels(coords: np.ndarray) -> tuple[np.ndarray | None, float, float]:
    if len(coords) < 2:
        return None, 0.0, 0.0
    centered = coords - coords.mean(axis=0)
    try:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, 0.0, 0.0
    if len(singular_values) == 0 or singular_values[0] <= 1e-12:
        return None, 0.0, 0.0
    direction = vh[0, :2].astype(np.float64)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return None, 0.0, 0.0
    direction = direction / norm
    second = singular_values[1] if len(singular_values) > 1 else 0.0
    linearity = 1.0 - float(second / singular_values[0])
    event_length = float(2.0 * singular_values[0] / max(1.0, np.sqrt(len(coords))))
    return direction, float(np.clip(linearity, 0.0, 1.0)), event_length


def image_vector_to_cartesian(vector: np.ndarray) -> np.ndarray:
    converted = np.asarray(vector, dtype=np.float64).copy()
    converted[..., 1] *= -1.0
    return converted


def cartesian_vector_to_image(vector: np.ndarray) -> np.ndarray:
    converted = np.asarray(vector, dtype=np.float64).copy()
    converted[..., 1] *= -1.0
    return converted


def trace_candidates_from_crystal_config(
    euler_radians: np.ndarray,
    crystal_config: dict | None,
    fallback_angle_tolerance: float,
) -> dict:
    config = normalized_crystal_config(crystal_config, fallback_angle_tolerance)
    ebsd_convention = config["ebsd_convention"]
    if ebsd_convention != "EDAX":
        raise ValueError(
            "Oxford reference frame is available in the UI but is not implemented in the parent trace functions yet."
        )

    vectors = []
    labels = []
    modes = []
    kinds = []
    tolerances = []
    for mode in config["active_modes"]:
        mode_vectors, kind = call_trace_function_for_mode(
            mode,
            euler_radians,
            config,
        )
        for index, vector in enumerate(np.asarray(mode_vectors, dtype=np.float64)):
            vectors.append(vector)
            labels.append(f"{short_mode_label(mode)}_{index + 1}")
            modes.append(mode)
            kinds.append(kind)
            tolerances.append(config["twin_tolerance"] if kind == "twin" else config["slip_tolerance"])

    if not vectors:
        raise ValueError("No active crystal slip/twin modes are selected.")

    return {
        "vectors": np.asarray(vectors, dtype=np.float64),
        "labels": np.asarray(labels, dtype=object),
        "modes": np.asarray(modes, dtype=object),
        "kinds": np.asarray(kinds, dtype=object),
        "tolerances": np.asarray(tolerances, dtype=np.float64),
        "mode_priority": list(config["active_modes"]),
    }


def normalized_crystal_config(crystal_config: dict | None, fallback_angle_tolerance: float) -> dict:
    if not crystal_config:
        crystal_config = {
            "crystal_structure": "HCP (Hexagonal Close Packed)",
            "ebsd_reference_frame": "EDAX (default reference frame)",
            "active_modes": DEFAULT_CRYSTAL_MODES["HCP (Hexagonal Close Packed)"],
            "hcp_lattice_a": 2.95,
            "hcp_lattice_c": 4.686,
        }
    ebsd_reference = str(crystal_config.get("ebsd_reference_frame", EBSD_REFERENCE_FRAMES[0]))
    ebsd_convention = "EDAX" if ebsd_reference.startswith("EDAX") else "Oxford"
    slip_tolerance = float(crystal_config.get("slip_angle_tolerance_deg") or fallback_angle_tolerance)
    twin_tolerance = float(crystal_config.get("twin_angle_tolerance_deg") or slip_tolerance)
    active_modes = [
        mode
        for mode in crystal_config.get("active_modes", [])
        if isinstance(mode, str)
    ]
    return {
        "crystal_structure": str(crystal_config.get("crystal_structure", "HCP (Hexagonal Close Packed)")),
        "ebsd_convention": ebsd_convention,
        "active_modes": active_modes,
        "hcp_lattice_a": float(crystal_config.get("hcp_lattice_a") or 2.95),
        "hcp_lattice_c": float(crystal_config.get("hcp_lattice_c") or 4.686),
        "slip_tolerance": slip_tolerance,
        "twin_tolerance": twin_tolerance,
    }


def call_trace_function_for_mode(mode: str, euler_radians: np.ndarray, config: dict) -> tuple[np.ndarray, str]:
    if mode.startswith("Basal"):
        return calcSlipTracesHCPBasal(euler_radians, "radians", config["ebsd_convention"]), "slip"
    if mode.startswith("Prismatic"):
        return calcSlipTracesHCPPrism(euler_radians, "radians", config["ebsd_convention"]), "slip"
    if mode.startswith("Pyramidal I"):
        return calcSlipTracesHCPPyra_I_A(
            euler_radians,
            "radians",
            config["hcp_lattice_a"],
            config["hcp_lattice_c"],
            config["ebsd_convention"],
        ), "slip"
    if mode.startswith("Pyramidal II"):
        return calcSlipTracesHCPPyra_II_CA(
            euler_radians,
            "radians",
            config["hcp_lattice_a"],
            config["hcp_lattice_c"],
            config["ebsd_convention"],
        ), "slip"
    if mode.startswith("{10-12}"):
        return calcTwinTracesHCP(euler_radians, "t1", "radians", config["hcp_lattice_a"], config["hcp_lattice_c"], config["ebsd_convention"]), "twin"
    if mode.startswith("{11-21}"):
        return calcTwinTracesHCP(euler_radians, "t2", "radians", config["hcp_lattice_a"], config["hcp_lattice_c"], config["ebsd_convention"]), "twin"
    if mode.startswith("{11-22}"):
        return calcTwinTracesHCP(euler_radians, "c1", "radians", config["hcp_lattice_a"], config["hcp_lattice_c"], config["ebsd_convention"]), "twin"
    if mode.startswith("{10-11}"):
        return calcTwinTracesHCP(euler_radians, "c2", "radians", config["hcp_lattice_a"], config["hcp_lattice_c"], config["ebsd_convention"]), "twin"
    if mode.startswith("{11-24}"):
        return calcTwinTracesHCP(euler_radians, "c3", "radians", config["hcp_lattice_a"], config["hcp_lattice_c"], config["ebsd_convention"]), "twin"
    if mode.startswith("BCC {110}"):
        return calcSlipTracesBCC110(euler_radians, "radians"), "slip"
    if mode.startswith("BCC {112}"):
        return calcSlipTracesBCC112(euler_radians, "radians"), "slip"
    raise ValueError(f"No trace function is wired for selected mode: {mode}")


def short_mode_label(mode: str) -> str:
    replacements = {
        "Basal - 1x plane trace": "basal",
        "Prismatic - 3x plane traces": "prism",
        "Pyramidal I - 6x plane traces": "pyra_i",
        "Pyramidal II - 6x plane traces": "pyra_ii",
        "{10-12} Twin (Tension) - 6x plane traces": "twin_10_12",
        "{11-21} Twin (Tension) - 6x plane traces": "twin_11_21",
        "{11-22} Twin (Compression) - 6x plane traces": "twin_11_22",
        "{10-11} Twin (Compression) - 6x plane traces": "twin_10_11",
        "{11-24} Twin (Compression) - 6x plane traces": "twin_11_24",
        "BCC {110} Slip - 6x plane traces": "bcc_110",
        "BCC {112} Slip - 12x plane traces": "bcc_112",
    }
    return replacements.get(mode, re.sub(r"[^A-Za-z0-9]+", "_", mode).strip("_").lower())


def normalize_2d_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    out = np.zeros_like(vectors, dtype=np.float64)
    safe = norms > 1e-12
    out[safe] = vectors[safe] / norms[safe, None]
    return out


def angular_mismatch_values(event_direction: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    dots = np.abs(candidates @ event_direction[:2])
    dots = np.clip(dots, 0.0, 1.0)
    return np.degrees(np.arccos(dots))


def select_ordered_trace_assignment_index(
    angle_values: np.ndarray,
    modes: np.ndarray,
    tolerances: np.ndarray,
    mode_priority: list[str],
) -> int:
    angles = np.asarray(angle_values, dtype=np.float64)
    for mode in mode_priority:
        indices = np.flatnonzero(np.asarray(modes, dtype=object) == mode)
        passing = indices[angles[indices] <= np.asarray(tolerances, dtype=np.float64)[indices]]
        if len(passing):
            return int(passing[np.argmin(angles[passing])])
    return int(np.argmin(angles))


def trace_collection_columns(trace_info: dict, vectors: np.ndarray, angles: np.ndarray) -> dict:
    labels = [str(value) for value in trace_info["labels"]]
    modes = [str(value) for value in trace_info["modes"]]
    kinds = [str(value) for value in trace_info["kinds"]]
    tolerances = [round(float(value), 6) for value in trace_info["tolerances"]]
    vector_values = [
        [round(float(vector[0]), 6), round(float(vector[1]), 6)]
        for vector in np.asarray(vectors, dtype=np.float64)
    ]
    angle_values = [round(float(value), 6) for value in np.asarray(angles, dtype=np.float64)]
    return {
        "all_trace_labels": json.dumps(labels),
        "all_trace_modes": json.dumps(modes),
        "all_trace_kinds": json.dumps(kinds),
        "all_trace_vectors": json.dumps(vector_values),
        "all_trace_angle_errors_deg": json.dumps(angle_values),
        "all_trace_tolerances_deg": json.dumps(tolerances),
    }


def legacy_trace_columns(labels: np.ndarray, vectors: np.ndarray, angles: np.ndarray) -> dict:
    row = {}
    legacy_slots = {
        "prism_1": "prism1",
        "prism_2": "prism2",
        "prism_3": "prism3",
        "basal_1": "basal",
    }
    for label, prefix in legacy_slots.items():
        matches = np.flatnonzero(np.asarray(labels, dtype=object) == label)
        if len(matches):
            idx = int(matches[0])
            row[f"{prefix}_dx"] = round(float(vectors[idx, 0]), 6)
            row[f"{prefix}_dy"] = round(float(vectors[idx, 1]), 6)
            row[f"angle_to_{prefix}_deg"] = round(float(angles[idx]), 3)
        else:
            row[f"{prefix}_dx"] = np.nan
            row[f"{prefix}_dy"] = np.nan
            row[f"angle_to_{prefix}_deg"] = np.nan
    return row


def compute_event_morphology_features(
    events: pd.DataFrame,
    event_pixels: pd.DataFrame,
    min_pixels: int,
    linearity_threshold: float,
    aspect_threshold: float,
    blob_density_threshold: float,
) -> pd.DataFrame:
    event_col = event_identity_column(event_pixels)
    event_lookup = (
        events.set_index(events[event_identity_column(events)].astype(str), drop=False)
        if "event_id" in events.columns
        else pd.DataFrame()
    )
    rows = []
    for event_id, group in event_pixels.groupby(event_col, sort=False):
        segment_id = str(event_id)
        coords = group[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float64)
        num_pixels = int(len(coords))
        direction, linearity, event_length = event_direction_from_pixels(coords)
        xs = coords[:, 0] if len(coords) else np.array([0.0])
        ys = coords[:, 1] if len(coords) else np.array([0.0])
        bbox_w = int(xs.max() - xs.min() + 1) if len(coords) else 0
        bbox_h = int(ys.max() - ys.min() + 1) if len(coords) else 0
        bbox_area = max(1, bbox_w * bbox_h)
        density = float(num_pixels / bbox_area)
        aspect_ratio = morphology_aspect_ratio(coords)
        component_count = event_connected_component_count(coords)
        event_type = classify_event_morphology(
            num_pixels,
            linearity,
            aspect_ratio,
            density,
            component_count,
            min_pixels,
            linearity_threshold,
            aspect_threshold,
            blob_density_threshold,
        )
        row = {
            "event_id": str(group["event_id"].iloc[0]),
            "segment_id": segment_id,
            "event_type": event_type,
            "num_pixels": num_pixels,
            "linearity": round(linearity, 4),
            "aspect_ratio": round(aspect_ratio, 4),
            "density": round(density, 4),
            "component_count": int(component_count),
            "event_length_pixels": round(event_length, 3),
            "bbox_width": bbox_w,
            "bbox_height": bbox_h,
            "event_angle_deg": round(vector_angle_degrees(direction), 3) if direction is not None else np.nan,
        }
        if not event_lookup.empty and segment_id in event_lookup.index:
            source = event_lookup.loc[segment_id]
            if isinstance(source, pd.DataFrame):
                source = source.iloc[0]
            for col in ["original_event_id", "cut_index", "method"]:
                if col in source:
                    row[col] = source[col]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["event_type", "num_pixels"], ascending=[True, False]).reset_index(drop=True)


def morphology_aspect_ratio(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    centered = coords - coords.mean(axis=0)
    try:
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    if len(singular_values) < 2 or singular_values[1] <= 1e-12:
        return float("inf") if singular_values[0] > 0 else 0.0
    return float(singular_values[0] / singular_values[1])


def event_connected_component_count(coords: np.ndarray) -> int:
    if len(coords) == 0:
        return 0
    xs = coords[:, 0].astype(np.int64)
    ys = coords[:, 1].astype(np.int64)
    x0, y0 = int(xs.min()), int(ys.min())
    mask = np.zeros((int(ys.max() - y0 + 1), int(xs.max() - x0 + 1)), dtype=bool)
    mask[ys - y0, xs - x0] = True
    labels = measure.label(mask, connectivity=2)
    return int(labels.max())


def classify_event_morphology(
    num_pixels: int,
    linearity: float,
    aspect_ratio: float,
    density: float,
    component_count: int,
    min_pixels: int,
    linearity_threshold: float,
    aspect_threshold: float,
    blob_density_threshold: float,
) -> str:
    if num_pixels < int(min_pixels):
        return "small_noise"
    if component_count > 1:
        return "fragmented"
    if linearity >= float(linearity_threshold) and aspect_ratio >= float(aspect_threshold):
        return "linear"
    if density >= float(blob_density_threshold):
        return "blob_like"
    return "irregular"


def evenly_sample_coords(coords: np.ndarray, max_points: int) -> np.ndarray:
    max_points = max(1, int(max_points))
    if len(coords) <= max_points:
        return coords
    indices = np.linspace(0, len(coords) - 1, max_points).round().astype(np.int64)
    return coords[indices]


def angular_mismatch_matrix(event_direction: np.ndarray, prism_dic: np.ndarray, basal_dic: np.ndarray) -> np.ndarray:
    candidates = np.concatenate([prism_dic[:, :, :2], basal_dic[:, None, :2]], axis=1)
    normalized = normalize_trace_candidates(candidates)
    dots = np.abs(np.einsum("nkd,d->nk", normalized, event_direction[:2]))
    dots = np.clip(dots, 0.0, 1.0)
    return np.degrees(np.arccos(dots))


def select_trace_assignment_index(angle_values: np.ndarray, angle_tolerance_deg: float) -> int:
    angles = np.asarray(angle_values, dtype=np.float64)
    within_threshold = angles <= float(angle_tolerance_deg)

    # If multiple traces are acceptable, prefer prismatic slip over basal slip.
    # Within the prismatic family, assign the individual trace with the lowest
    # angle error. If nothing passes, keep the nearest trace for diagnostics.
    prism_indices = np.arange(3)
    passing_prisms = prism_indices[within_threshold[:3]]
    if len(passing_prisms):
        return int(passing_prisms[np.argmin(angles[passing_prisms])])
    if len(angles) > 3 and bool(within_threshold[3]):
        return 3
    return int(np.argmin(angles))


def normalize_trace_candidates(candidates: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(candidates, axis=2)
    safe = norms > 1e-12
    normalized = np.zeros_like(candidates, dtype=np.float64)
    normalized[safe] = candidates[safe] / norms[safe, None]
    return normalized


def dominant_trace_mode(best_indices: np.ndarray) -> str:
    labels = np.array(["prism1", "prism2", "prism3", "basal"])
    if len(best_indices) == 0:
        return "unknown"
    counts = np.bincount(best_indices, minlength=len(labels))
    return str(labels[int(np.argmax(counts))])


def vector_angle_degrees(vector: np.ndarray) -> float:
    angle = np.degrees(np.arctan2(float(vector[1]), float(vector[0])))
    if angle < 0:
        angle += 180.0
    if angle >= 180.0:
        angle -= 180.0
    return float(angle)


def classify_event_score(score: float, likely_real_threshold: float, uncertain_threshold: float) -> str:
    if score >= float(likely_real_threshold):
        return "likely_real"
    if score >= float(uncertain_threshold):
        return "uncertain"
    return "likely_noise"


def empty_event_score_row(event_id: str, num_pixels: int, reason: str) -> dict:
    return {
        "event_id": event_id,
        "segment_id": event_id,
        "classification": "likely_noise",
        "score": 0.0,
        "best_mode": "unknown",
        "num_pixels": int(num_pixels),
        "event_center_x": np.nan,
        "event_center_y": np.nan,
        "nearest_ang_x_dic": np.nan,
        "nearest_ang_y_dic": np.nan,
        "center_lookup_distance": np.nan,
        "event_angle_deg": np.nan,
        "linearity": 0.0,
        "event_length_pixels": 0.0,
        "best_angle_error_deg": np.nan,
        "angle_tolerance_deg": np.nan,
        "passes_angle": False,
        "passes_lookup": False,
        "passes_linearity": False,
        "angle_to_prism1_deg": np.nan,
        "angle_to_prism2_deg": np.nan,
        "angle_to_prism3_deg": np.nan,
        "angle_to_basal_deg": np.nan,
        "reason": reason,
    }


@st.cache_data(show_spinner=False)
def load_boundary_cut_inspection_data(
    original_pixels_path: str,
    cut_events_path: str,
    cut_pixels_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_pixels = pd.read_csv(original_pixels_path)
    cut_events = pd.read_csv(cut_events_path)
    cut_pixels = pd.read_csv(cut_pixels_path)
    required_pixels = {"event_id", "pixel_x", "pixel_y"}
    if not required_pixels.issubset(original_pixels.columns):
        raise ValueError("Original event pixels CSV must include event_id, pixel_x, pixel_y.")
    if not required_pixels.issubset(cut_pixels.columns):
        raise ValueError("Cut event pixels CSV must include event_id, pixel_x, pixel_y.")
    if "event_id" not in cut_events.columns:
        raise ValueError("Cut events CSV must include event_id.")

    original_pixels = original_pixels.copy()
    cut_events = cut_events.copy()
    cut_pixels = cut_pixels.copy()
    for df in [original_pixels, cut_pixels]:
        df["event_id"] = df["event_id"].astype(str)
        df["pixel_x"] = df["pixel_x"].astype(np.int64)
        df["pixel_y"] = df["pixel_y"].astype(np.int64)
    cut_events["event_id"] = cut_events["event_id"].astype(str)
    if "original_event_id" in cut_events.columns:
        cut_events["original_event_id"] = cut_events["original_event_id"].astype(str)
    if "original_event_id" in cut_pixels.columns:
        cut_pixels["original_event_id"] = cut_pixels["original_event_id"].astype(str)
    return original_pixels, cut_events, cut_pixels


def build_boundary_cut_inspection_overlay(
    bln_path: str,
    boundary_path: str,
    original_pixels: pd.DataFrame,
    cut_events: pd.DataFrame,
    cut_pixels: pd.DataFrame,
    original_event_id: str,
    margin: int,
    boundary_threshold: float,
    boundary_dilation: int,
    show_labels: bool,
) -> dict:
    original = original_pixels[original_pixels["event_id"].astype(str) == str(original_event_id)].copy()
    if original.empty:
        raise ValueError(f"No original pixels found for event {original_event_id}.")

    if "original_event_id" in cut_pixels.columns:
        cut_subset = cut_pixels[cut_pixels["original_event_id"].astype(str) == str(original_event_id)].copy()
    elif "original_event_id" in cut_events.columns:
        cut_ids = cut_events.loc[
            cut_events["original_event_id"].astype(str) == str(original_event_id),
            "event_id",
        ].astype(str)
        cut_subset = cut_pixels[cut_pixels["event_id"].astype(str).isin(set(cut_ids))].copy()
    else:
        cut_subset = cut_pixels[cut_pixels["event_id"].astype(str).str.startswith(str(original_event_id))].copy()

    width, height = image_size(bln_path)
    crop_source = pd.concat([original[["event_id", "pixel_x", "pixel_y"]], cut_subset[["event_id", "pixel_x", "pixel_y"]]], ignore_index=True)
    crop = event_extent_crop(crop_source, (height, width), margin=int(margin))
    x0 = crop["x"]
    y0 = crop["y"]
    w = crop["width"]
    h = crop["height"]
    bln_crop = load_crop_by_rect(bln_path, x0, y0, w, h, display_min=0.0, display_max=1.0)
    overlay = bln_crop["display_rgb"].copy()

    original_mask = pixels_to_crop_mask(original, "pixel_x", "pixel_y", x0, y0, (h, w))
    kept_mask = pixels_to_crop_mask(cut_subset, "pixel_x", "pixel_y", x0, y0, (h, w)) if not cut_subset.empty else np.zeros((h, w), dtype=bool)
    removed_mask = original_mask & ~kept_mask
    overlay[original_mask] = [150, 150, 150]

    boundary_mask = load_black_boundary_mask(boundary_path, float(boundary_threshold))
    boundary_crop = boundary_mask[y0:y0 + h, x0:x0 + w]
    if int(boundary_dilation) > 0:
        boundary_crop = morphology.binary_dilation(boundary_crop, morphology.disk(int(boundary_dilation)))
    overlay[boundary_crop] = [0, 80, 255]

    for _cut_event_id, group in cut_subset.groupby("event_id", sort=False):
        mask = pixels_to_crop_mask(group, "pixel_x", "pixel_y", x0, y0, (h, w))
        overlay[mask] = [255, 0, 0]
    overlay[removed_mask] = [170, 0, 220]

    if show_labels and not cut_subset.empty:
        event_ids = list(dict.fromkeys(cut_subset["event_id"].astype(str).tolist()))
        label_event_centroids(overlay, cut_subset, event_ids, x0, y0)

    if "original_event_id" in cut_events.columns:
        selected_cut_events = cut_events[cut_events["original_event_id"].astype(str) == str(original_event_id)].copy()
    else:
        selected_cut_events = cut_events[cut_events["event_id"].astype(str).isin(cut_subset["event_id"].astype(str).unique())].copy()

    return {
        "overlay": overlay,
        "crop": crop,
        "cut_events": selected_cut_events,
        "stats": {
            "original_pixels": int(len(original)),
            "cut_segments": int(cut_subset["event_id"].nunique()) if not cut_subset.empty else 0,
            "kept_pixels": int(len(cut_subset)),
            "removed_pixels": int(removed_mask.sum()),
            "boundary_pixels": int(boundary_crop.sum()),
        },
    }


def build_event_reality_overlay(
    bln_path: str,
    event_pixels: pd.DataFrame,
    score_result: pd.DataFrame,
    full_image: bool,
    margin: int,
    max_dim: int,
    use_full_resolution: bool,
    boundary_path: str = "",
    boundary_threshold: float = 0.20,
    draw_vectors: bool = True,
    draw_trace_vectors: bool = True,
    draw_all_trace_vectors: bool = False,
    vector_scale: float = 1.0,
    vector_width: int = 3,
) -> dict:
    width, height = image_size(bln_path)
    if full_image:
        crop = {"x": 0, "y": 0, "width": width, "height": height}
    else:
        crop = event_extent_crop(event_pixels, (height, width), margin=int(margin))

    x0 = crop["x"]
    y0 = crop["y"]
    w = crop["width"]
    h = crop["height"]
    bln_crop = load_crop_by_rect(bln_path, x0, y0, w, h, display_min=0.0, display_max=1.0)
    overlay = bln_crop["display_rgb"].copy()

    if boundary_path and Path(boundary_path).exists():
        boundary_mask = load_black_boundary_mask(boundary_path, float(boundary_threshold))
        boundary_crop = boundary_mask[y0:y0 + h, x0:x0 + w]
        if boundary_crop.shape == overlay.shape[:2]:
            overlay[boundary_crop] = [0, 80, 255]

    score_identity_col = event_identity_column(score_result)
    class_by_id = dict(zip(score_result[score_identity_col].astype(str), score_result["classification"].astype(str)))
    color_by_class = {
        "likely_real": np.array([0, 230, 70], dtype=np.uint8),
        "likely_noise": np.array([255, 0, 0], dtype=np.uint8),
        "uncertain": np.array([255, 220, 0], dtype=np.uint8),
    }

    pixels = event_pixels.copy()
    pixels["event_id"] = pixels["event_id"].astype(str)
    pixel_identity_col = event_identity_column(pixels)
    pixels["classification"] = pixels[pixel_identity_col].astype(str).map(class_by_id).fillna("uncertain")
    for classification, color in color_by_class.items():
        subset = pixels[pixels["classification"] == classification]
        if subset.empty:
            continue
        mask = pixels_to_crop_mask(subset, "pixel_x", "pixel_y", x0, y0, (h, w))
        overlay[mask] = color

    if draw_vectors:
        overlay = draw_event_direction_vectors(
            overlay,
            event_pixels,
            crop,
            scale=float(vector_scale),
            width=int(vector_width),
        )
    if draw_trace_vectors:
        overlay = draw_selected_trace_vectors(
            overlay,
            score_result,
            crop,
            scale=float(vector_scale),
            width=int(vector_width),
        )
    if draw_all_trace_vectors:
        overlay = draw_all_trace_vectors_overlay(
            overlay,
            score_result,
            crop,
            scale=float(vector_scale),
            width=int(vector_width),
        )

    if not use_full_resolution:
        overlay = to_display(overlay, max_dim=max_dim)
    return {"overlay": overlay, "crop": crop, "bln_path": bln_path}


def build_event_numbered_bln_overlay(
    bln_path: str,
    event_pixels: pd.DataFrame,
    full_image: bool,
    margin: int,
    max_dim: int,
    color_by_id: bool = True,
) -> dict:
    width, height = image_size(bln_path)
    if full_image:
        crop = {"x": 0, "y": 0, "width": width, "height": height}
    else:
        crop = event_extent_crop(event_pixels, (height, width), margin=int(margin))

    x0 = crop["x"]
    y0 = crop["y"]
    w = crop["width"]
    h = crop["height"]
    bln_crop = load_crop_by_rect(bln_path, x0, y0, w, h, display_min=0.0, display_max=1.0)
    overlay = bln_crop["display_rgb"].copy()

    pixels = event_pixels.copy()
    pixels["event_id"] = pixels["event_id"].astype(str)
    event_ids = list(dict.fromkeys(pixels["event_id"].tolist()))
    for index, event_id in enumerate(event_ids):
        subset = pixels[pixels["event_id"] == event_id]
        if subset.empty:
            continue
        mask = pixels_to_crop_mask(subset, "pixel_x", "pixel_y", x0, y0, (h, w))
        overlay[mask] = event_color(index) if color_by_id else [255, 0, 0]

    label_event_centroids(overlay, pixels, event_ids, x0, y0)
    if int(max_dim) > 0:
        overlay = to_display(overlay, max_dim=max_dim)
    return {"overlay": overlay, "crop": crop}


def build_event_type_overlay(
    bln_path: str,
    event_pixels: pd.DataFrame,
    features: pd.DataFrame,
    margin: int,
    max_dim: int,
    show_labels: bool,
    full_image: bool = False,
) -> dict:
    width, height = image_size(bln_path)
    if full_image:
        crop = {"x": 0, "y": 0, "width": width, "height": height}
    else:
        crop = event_extent_crop(event_pixels, (height, width), margin=int(margin))
    x0 = crop["x"]
    y0 = crop["y"]
    w = crop["width"]
    h = crop["height"]
    bln_crop = load_crop_by_rect(bln_path, x0, y0, w, h, display_min=0.0, display_max=1.0)
    overlay = bln_crop["display_rgb"].copy()

    feature_identity_col = event_identity_column(features)
    type_by_id = dict(zip(features[feature_identity_col].astype(str), features["event_type"].astype(str)))
    color_by_type = {
        "linear": np.array([0, 255, 255], dtype=np.uint8),
        "blob_like": np.array([255, 0, 255], dtype=np.uint8),
        "fragmented": np.array([255, 150, 0], dtype=np.uint8),
        "small_noise": np.array([255, 0, 0], dtype=np.uint8),
        "irregular": np.array([160, 160, 160], dtype=np.uint8),
    }
    pixels = event_pixels.copy()
    pixels["event_id"] = pixels["event_id"].astype(str)
    pixel_identity_col = event_identity_column(pixels)
    pixels["event_type"] = pixels[pixel_identity_col].astype(str).map(type_by_id).fillna("irregular")
    for event_type, color in color_by_type.items():
        subset = pixels[pixels["event_type"] == event_type]
        if subset.empty:
            continue
        mask = pixels_to_crop_mask(subset, "pixel_x", "pixel_y", x0, y0, (h, w))
        overlay[mask] = color

    if show_labels:
        event_ids = list(dict.fromkeys(pixels[pixel_identity_col].astype(str).tolist()))
        label_event_centroids(overlay, pixels, event_ids, x0, y0, id_col=pixel_identity_col)

    return {"overlay": to_display(overlay, max_dim=max_dim), "crop": crop}


def build_filtered_event_type_overlay(
    bln_path: str,
    event_pixels: pd.DataFrame,
    features: pd.DataFrame,
    event_type: str,
    margin: int,
    max_dim: int | None,
    full_image: bool = False,
    boundary_path: str = "",
    boundary_threshold: float = 0.20,
) -> dict:
    width, height = image_size(bln_path)
    if full_image:
        crop = {"x": 0, "y": 0, "width": width, "height": height}
    else:
        crop = event_extent_crop(event_pixels, (height, width), margin=int(margin))

    x0 = crop["x"]
    y0 = crop["y"]
    w = crop["width"]
    h = crop["height"]
    render_step = 1
    if max_dim is not None and int(max_dim) > 0:
        render_step = max(1, int(np.ceil(max(h, w) / int(max_dim))))
    overlay = load_scaled_display_crop(
        bln_path,
        x0,
        y0,
        w,
        h,
        render_step=render_step,
        display_min=0.0,
        display_max=1.0,
    )

    if boundary_path and Path(boundary_path).exists():
        boundary_mask = load_black_boundary_mask(boundary_path, float(boundary_threshold))
        boundary_crop = boundary_mask[y0:y0 + h, x0:x0 + w]
        boundary_preview = boundary_crop[::render_step, ::render_step]
        bh = min(boundary_preview.shape[0], overlay.shape[0])
        bw = min(boundary_preview.shape[1], overlay.shape[1])
        if bh > 0 and bw > 0:
            overlay[:bh, :bw][boundary_preview[:bh, :bw]] = [0, 80, 255]

    feature_identity_col = event_identity_column(features)
    type_by_id = dict(zip(features[feature_identity_col].astype(str), features["event_type"].astype(str)))
    pixels = event_pixels.copy()
    pixels["event_id"] = pixels["event_id"].astype(str)
    pixel_identity_col = event_identity_column(pixels)
    pixels["event_type"] = pixels[pixel_identity_col].astype(str).map(type_by_id).fillna("irregular")
    subset = pixels[pixels["event_type"] == str(event_type)]
    if not subset.empty:
        if str(event_type) == "linear":
            event_ids = list(dict.fromkeys(subset[pixel_identity_col].astype(str).tolist()))
            for index, event_id in enumerate(event_ids):
                event_subset = subset[subset[pixel_identity_col].astype(str) == event_id]
                paint_pixels_on_scaled_overlay(
                    overlay,
                    event_subset,
                    "pixel_x",
                    "pixel_y",
                    x0,
                    y0,
                    render_step,
                    event_color(index),
                )
        else:
            paint_pixels_on_scaled_overlay(
                overlay,
                subset,
                "pixel_x",
                "pixel_y",
                x0,
                y0,
                render_step,
                np.array([255, 0, 0], dtype=np.uint8),
            )

    return {"overlay": overlay, "crop": crop}


def load_scaled_display_crop(
    path_str: str,
    crop_x: int,
    crop_y: int,
    crop_width: int,
    crop_height: int,
    render_step: int,
    display_min: float | None = None,
    display_max: float | None = None,
) -> np.ndarray:
    render_step = max(1, int(render_step))
    with Image.open(path_str) as img:
        img_w, img_h = img.size
        crop_width = max(1, min(int(crop_width), img_w))
        crop_height = max(1, min(int(crop_height), img_h))
        x0 = max(0, min(int(crop_x), img_w - crop_width))
        y0 = max(0, min(int(crop_y), img_h - crop_height))
        crop_img = img.crop((x0, y0, x0 + crop_width, y0 + crop_height))
        if render_step > 1:
            preview_size = (
                max(1, int(np.ceil(crop_width / render_step))),
                max(1, int(np.ceil(crop_height / render_step))),
            )
            crop_img = crop_img.resize(preview_size, Image.Resampling.BILINEAR)
        arr = np.asarray(crop_img).copy()
    return display_crop_rgb(arr, display_min, display_max)


def paint_pixels_on_scaled_overlay(
    overlay: np.ndarray,
    pixels: pd.DataFrame,
    x_col: str,
    y_col: str,
    crop_x: int,
    crop_y: int,
    render_step: int,
    color: np.ndarray | list[int] | tuple[int, int, int],
) -> None:
    if pixels.empty:
        return
    render_step = max(1, int(render_step))
    xs = ((pixels[x_col].to_numpy(dtype=np.int64) - int(crop_x)) // render_step).astype(np.int64)
    ys = ((pixels[y_col].to_numpy(dtype=np.int64) - int(crop_y)) // render_step).astype(np.int64)
    valid = (xs >= 0) & (xs < overlay.shape[1]) & (ys >= 0) & (ys < overlay.shape[0])
    if np.any(valid):
        overlay[ys[valid], xs[valid]] = np.asarray(color, dtype=np.uint8)


def event_shape_overlay_uses_full_image(
    events: pd.DataFrame,
    event_pixels: pd.DataFrame,
    bln_path: str,
) -> bool:
    if event_pixels.empty:
        return True
    if {"crop_x", "crop_y"}.issubset(events.columns):
        crop_x = pd.to_numeric(events["crop_x"], errors="coerce")
        crop_y = pd.to_numeric(events["crop_y"], errors="coerce")
        if len(crop_x) and crop_x.fillna(0).eq(0).all() and crop_y.fillna(0).eq(0).all():
            return True

    width, height = image_size(bln_path)
    xs = event_pixels["pixel_x"].to_numpy(dtype=np.int64)
    ys = event_pixels["pixel_y"].to_numpy(dtype=np.int64)
    return bool(
        xs.min(initial=0) <= 0
        and ys.min(initial=0) <= 0
        and xs.max(initial=0) >= width - 1
        and ys.max(initial=0) >= height - 1
    )


def build_single_event_type_full_overlay(
    bln_path: str,
    event_pixels: pd.DataFrame,
    features: pd.DataFrame,
    event_type: str,
) -> dict:
    width, height = image_size(bln_path)
    bln_crop = load_crop_by_rect(bln_path, 0, 0, width, height, display_min=0.0, display_max=1.0)
    overlay = bln_crop["display_rgb"].copy()
    type_by_id = dict(zip(features["event_id"].astype(str), features["event_type"].astype(str)))

    pixels = event_pixels.copy()
    pixels["event_id"] = pixels["event_id"].astype(str)
    pixels["event_type"] = pixels["event_id"].map(type_by_id).fillna("irregular")
    subset = pixels[pixels["event_type"] == str(event_type)]
    if not subset.empty:
        color = event_type_color(str(event_type))
        mask = pixels_to_crop_mask(subset, "pixel_x", "pixel_y", 0, 0, (height, width))
        overlay[mask] = color
    return {"overlay": overlay, "crop": {"x": 0, "y": 0, "width": width, "height": height}}


def build_single_event_overlay(
    bln_path: str,
    event_pixels: pd.DataFrame,
    features: pd.DataFrame,
    event_id: str,
    margin: int,
) -> dict:
    width, height = image_size(bln_path)
    pixels = event_pixels.copy()
    pixels["event_id"] = pixels["event_id"].astype(str)
    subset = pixels[pixels["event_id"] == str(event_id)]
    if subset.empty:
        raise ValueError(f"No pixels found for event {event_id}.")

    crop = event_extent_crop(subset, (height, width), margin=int(margin))
    x0 = crop["x"]
    y0 = crop["y"]
    w = crop["width"]
    h = crop["height"]
    bln_crop = load_crop_by_rect(bln_path, x0, y0, w, h, display_min=0.0, display_max=1.0)
    overlay = bln_crop["display_rgb"].copy()

    event_type = "irregular"
    feature_match = features[features["event_id"].astype(str) == str(event_id)]
    if not feature_match.empty:
        event_type = str(feature_match.iloc[0]["event_type"])
    color = event_type_color(event_type)
    mask = pixels_to_crop_mask(subset, "pixel_x", "pixel_y", x0, y0, (h, w))
    overlay[mask] = color
    label_event_centroids(overlay, subset, [str(event_id)], x0, y0)
    return {"overlay": overlay, "crop": crop}


def event_type_color(event_type: str) -> np.ndarray:
    return {
        "linear": np.array([0, 255, 255], dtype=np.uint8),
        "blob_like": np.array([255, 0, 255], dtype=np.uint8),
        "fragmented": np.array([255, 150, 0], dtype=np.uint8),
        "small_noise": np.array([255, 0, 0], dtype=np.uint8),
        "irregular": np.array([160, 160, 160], dtype=np.uint8),
    }.get(str(event_type), np.array([160, 160, 160], dtype=np.uint8))


def event_color(index: int) -> np.ndarray:
    hue = (index * 0.61803398875) % 1.0
    saturation = 0.85
    value = 0.95
    return np.array(hsv_to_rgb_uint8(hue, saturation, value), dtype=np.uint8)


def hsv_to_rgb_uint8(h: float, s: float, v: float) -> tuple[int, int, int]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


def label_event_centroids(
    overlay: np.ndarray,
    pixels: pd.DataFrame,
    event_ids: list[str],
    crop_x: int,
    crop_y: int,
    id_col: str = "event_id",
) -> None:
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    h, w = overlay.shape[:2]
    for index, event_id in enumerate(event_ids, start=1):
        subset = pixels[pixels[id_col].astype(str) == str(event_id)]
        if subset.empty:
            continue
        cx = float(subset["pixel_x"].mean() - crop_x)
        cy = float(subset["pixel_y"].mean() - crop_y)
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue
        label = str(index)
        bbox = draw.textbbox((cx, cy), label)
        pad = 2
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
        )
        draw.text((cx, cy), label, fill=(0, 0, 0))
    overlay[:] = np.asarray(image)


def draw_event_direction_vectors(
    overlay: np.ndarray,
    event_pixels: pd.DataFrame,
    crop: dict[str, int],
    scale: float,
    width: int,
) -> np.ndarray:
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    x0 = int(crop["x"])
    y0 = int(crop["y"])
    h, w = overlay.shape[:2]
    color = (0, 255, 255)

    event_col = event_identity_column(event_pixels)
    for _, group in event_pixels.groupby(event_col, sort=False):
        coords = group[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float64)
        direction, _, event_length = event_direction_from_pixels(coords)
        if direction is None:
            continue
        center = coords.mean(axis=0)
        half_length = max(8.0, 0.5 * event_length * float(scale))
        start = center - direction * half_length
        end = center + direction * half_length
        x_start = float(start[0] - x0)
        y_start = float(start[1] - y0)
        x_end = float(end[0] - x0)
        y_end = float(end[1] - y0)
        if (
            max(x_start, x_end) < 0
            or min(x_start, x_end) >= w
            or max(y_start, y_end) < 0
            or min(y_start, y_end) >= h
        ):
            continue
        draw.line(
            [(x_start, y_start), (x_end, y_end)],
            fill=color,
            width=max(1, int(width)),
        )

    return np.asarray(image)


def draw_selected_trace_vectors(
    overlay: np.ndarray,
    score_result: pd.DataFrame,
    crop: dict[str, int],
    scale: float,
    width: int,
) -> np.ndarray:
    required = {"event_center_x", "event_center_y", "best_trace_dx", "best_trace_dy", "event_length_pixels"}
    if not required.issubset(score_result.columns):
        return overlay

    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    x0 = int(crop["x"])
    y0 = int(crop["y"])
    h, w = overlay.shape[:2]
    color = (255, 0, 255)

    for _, row in score_result.iterrows():
        center = np.array([row["event_center_x"], row["event_center_y"]], dtype=np.float64)
        direction = vector_from_row(row, "best_trace_dx", "best_trace_dy")
        if not np.isfinite(center).all() or direction is None:
            continue
        event_length = float(row.get("event_length_pixels", 0.0))
        half_length = max(10.0, 0.5 * event_length * float(scale))
        draw_vector_line(
            draw,
            center,
            direction,
            x0,
            y0,
            half_length,
            color,
            max(1, int(width)),
            w,
            h,
        )

    return np.asarray(image)


def draw_all_trace_vectors_overlay(
    overlay: np.ndarray,
    score_result: pd.DataFrame,
    crop: dict[str, int],
    scale: float,
    width: int,
) -> np.ndarray:
    required = {
        "event_center_x",
        "event_center_y",
        "event_length_pixels",
    }
    if not required.issubset(score_result.columns):
        return overlay

    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    x0 = int(crop["x"])
    y0 = int(crop["y"])
    h, w = overlay.shape[:2]

    for _, row in score_result.iterrows():
        center = np.array([row["event_center_x"], row["event_center_y"]], dtype=np.float64)
        if not np.isfinite(center).all():
            continue
        event_length = float(row.get("event_length_pixels", 0.0))
        half_length = max(10.0, 0.5 * event_length * float(scale))
        for spec in trace_vectors_from_row(row):
            draw_vector_line(
                draw,
                center,
                spec["direction"],
                x0,
                y0,
                half_length,
                trace_vector_color(spec),
                max(1, int(width)),
                w,
                h,
            )

    return np.asarray(image)


@st.cache_data(show_spinner=False)
def load_ang_core_data(path_str: str) -> dict:
    metadata = parse_ang_header_metadata(path_str)
    data = pd.read_csv(
        path_str,
        comment="#",
        sep=r"\s+",
        header=None,
        usecols=[0, 1, 2, 3, 4],
        names=["phi1", "PHI", "phi2", "x", "y"],
    )
    for col in ["phi1", "PHI", "phi2", "x", "y"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["phi1", "PHI", "phi2", "x", "y"]).reset_index(drop=True)
    data, coordinate_normalization = normalize_ang_xy_to_grid(data, metadata)
    return {
        "metadata": metadata,
        "data": data,
        "coordinate_normalization": coordinate_normalization,
    }


def normalize_ang_xy_to_grid(data: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, dict]:
    xstep = float(metadata.get("XSTEP", np.nan))
    ystep = float(metadata.get("YSTEP", np.nan))
    if not np.isfinite(xstep) or xstep == 0:
        raise ValueError("ANG header is missing a valid XSTEP value.")
    if not np.isfinite(ystep) or ystep == 0:
        raise ValueError("ANG header is missing a valid YSTEP value.")

    out = data.copy()
    x_origin = float(out["x"].min())
    y_origin = float(out["y"].min())
    first_x = float(out["x"].iloc[0])
    first_y = float(out["y"].iloc[0])

    out["x_zeroed"] = out["x"] - x_origin
    out["y_zeroed"] = out["y"] - y_origin
    out["x_grid"] = out["x_zeroed"] / xstep
    out["y_grid"] = out["y_zeroed"] / ystep
    out["x_grid_round"] = np.rint(out["x_grid"]).astype(np.int64)
    out["y_grid_round"] = np.rint(out["y_grid"]).astype(np.int64)

    return out, {
        "xstep": xstep,
        "ystep": ystep,
        "first_x": first_x,
        "first_y": first_y,
        "x_origin": x_origin,
        "y_origin": y_origin,
        "x_offset_applied": -x_origin,
        "y_offset_applied": -y_origin,
        "x_grid_min": float(out["x_grid"].min()),
        "y_grid_min": float(out["y_grid"].min()),
        "x_grid_max": float(out["x_grid"].max()),
        "y_grid_max": float(out["y_grid"].max()),
        "x_grid_round_max": int(out["x_grid_round"].max()),
        "y_grid_round_max": int(out["y_grid_round"].max()),
    }


@st.cache_data(show_spinner=False)
def build_ang_euler_rgb_image(data: pd.DataFrame) -> np.ndarray:
    required = {"phi1", "PHI", "phi2", "x_grid_round", "y_grid_round"}
    if not required.issubset(data.columns):
        missing = sorted(required - set(data.columns))
        raise ValueError(f"ANG data is missing required columns: {missing}")

    width = int(data["x_grid_round"].max()) + 1
    height = int(data["y_grid_round"].max()) + 1
    if width <= 0 or height <= 0:
        raise ValueError("ANG grid has invalid dimensions.")

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    xs = data["x_grid_round"].to_numpy(dtype=np.int64)
    ys = data["y_grid_round"].to_numpy(dtype=np.int64)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)

    euler = data.loc[valid, ["phi1", "PHI", "phi2"]].to_numpy(dtype=np.float32)
    rgb_values = np.column_stack(
        [
            normalize_vector_to_uint8(euler[:, 0]),
            normalize_vector_to_uint8(euler[:, 1]),
            normalize_vector_to_uint8(euler[:, 2]),
        ]
    )
    rgb[ys[valid], xs[valid]] = rgb_values
    return rgb


def normalize_vector_to_uint8(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    return (np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)


def transform_ang_coordinates_to_dic(ang_data: pd.DataFrame, forward_params: np.ndarray) -> pd.DataFrame:
    required = {"x_grid", "y_grid", "phi1", "PHI", "phi2"}
    if not required.issubset(ang_data.columns):
        missing = sorted(required - set(ang_data.columns))
        raise ValueError(f"ANG data is missing required columns: {missing}")

    params = np.asarray(forward_params, dtype=np.float64)
    if params.shape[0] != 2 or params.shape[1] < 6:
        raise ValueError("Forward transform params must have shape 2 x 6 or larger.")

    x = ang_data["x_grid"].to_numpy(dtype=np.float64)
    y = ang_data["y_grid"].to_numpy(dtype=np.float64)
    x_dic, y_dic = apply_quadratic_polynomial_transform(x, y, params)

    out = pd.DataFrame(
        {
            "x_ebsd": x.astype(np.float32),
            "y_ebsd": y.astype(np.float32),
            "x_dic": x_dic.astype(np.float32),
            "y_dic": y_dic.astype(np.float32),
            "x_dic_round": np.rint(x_dic).astype(np.int64),
            "y_dic_round": np.rint(y_dic).astype(np.int64),
            "phi1": ang_data["phi1"].to_numpy(dtype=np.float32),
            "PHI": ang_data["PHI"].to_numpy(dtype=np.float32),
            "phi2": ang_data["phi2"].to_numpy(dtype=np.float32),
        }
    )
    return out


def apply_quadratic_polynomial_transform(
    x: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    terms = np.column_stack(
        [
            np.ones_like(x),
            x,
            y,
            x * x,
            x * y,
            y * y,
        ]
    )
    transformed = terms @ params[:, :6].T
    return transformed[:, 0], transformed[:, 1]


def dic_coordinate_metrics(dic_coords: pd.DataFrame, dic_width: int | None, dic_height: int | None) -> dict:
    x = dic_coords["x_dic"].to_numpy(dtype=np.float64)
    y = dic_coords["y_dic"].to_numpy(dtype=np.float64)
    if dic_width is None or dic_height is None:
        in_bounds = np.zeros(len(dic_coords), dtype=bool)
    else:
        in_bounds = (x >= 0) & (x < dic_width) & (y >= 0) & (y < dic_height)
    return {
        "x_dic_min": float(np.nanmin(x)) if len(x) else 0.0,
        "x_dic_max": float(np.nanmax(x)) if len(x) else 0.0,
        "y_dic_min": float(np.nanmin(y)) if len(y) else 0.0,
        "y_dic_max": float(np.nanmax(y)) if len(y) else 0.0,
        "in_bounds_count": int(in_bounds.sum()),
        "out_of_bounds_count": int(len(dic_coords) - in_bounds.sum()) if dic_width is not None and dic_height is not None else 0,
    }


@st.cache_data(show_spinner=False)
def build_dic_space_euler_rgb_image(
    dic_coords: pd.DataFrame,
    dic_width: int | None,
    dic_height: int | None,
) -> np.ndarray:
    if dic_width is None or dic_height is None:
        raise ValueError("Transformation metadata does not include DIC image width/height.")
    required = {"x_dic_round", "y_dic_round", "phi1", "PHI", "phi2"}
    if not required.issubset(dic_coords.columns):
        missing = sorted(required - set(dic_coords.columns))
        raise ValueError(f"DIC coordinate data is missing required columns: {missing}")

    rgb = np.zeros((int(dic_height), int(dic_width), 3), dtype=np.uint8)
    xs = dic_coords["x_dic_round"].to_numpy(dtype=np.int64)
    ys = dic_coords["y_dic_round"].to_numpy(dtype=np.int64)
    valid = (xs >= 0) & (xs < dic_width) & (ys >= 0) & (ys < dic_height)
    if not np.any(valid):
        return rgb

    euler = dic_coords.loc[valid, ["phi1", "PHI", "phi2"]].to_numpy(dtype=np.float32)
    rgb_values = np.column_stack(
        [
            normalize_vector_to_uint8(euler[:, 0]),
            normalize_vector_to_uint8(euler[:, 1]),
            normalize_vector_to_uint8(euler[:, 2]),
        ]
    )
    rgb[ys[valid], xs[valid]] = rgb_values
    return rgb


@st.cache_data(show_spinner=False)
def build_forward_filled_dic_euler_rgb_preview(
    dic_coords: pd.DataFrame,
    dic_width: int | None,
    dic_height: int | None,
    max_dim: int = 2200,
    full_resolution: bool = False,
) -> np.ndarray:
    _ = full_resolution
    if dic_width is None or dic_height is None:
        raise ValueError("Transformation metadata does not include DIC image width/height.")
    required = {"x_dic", "y_dic", "x_ebsd", "y_ebsd", "phi1", "PHI", "phi2"}
    if not required.issubset(dic_coords.columns):
        missing = sorted(required - set(dic_coords.columns))
        raise ValueError(f"DIC coordinate data is missing required columns: {missing}")

    step = max(1, int(np.ceil(max(int(dic_width), int(dic_height)) / max(1, int(max_dim)))))
    preview_w = int(np.ceil(int(dic_width) / step))
    preview_h = int(np.ceil(int(dic_height) / step))
    preview = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
    scatter_mask = np.zeros((preview_h, preview_w), dtype=bool)

    xs = np.rint(dic_coords["x_dic"].to_numpy(dtype=np.float64) / step).astype(np.int64)
    ys = np.rint(dic_coords["y_dic"].to_numpy(dtype=np.float64) / step).astype(np.int64)
    valid = (xs >= 0) & (xs < preview_w) & (ys >= 0) & (ys < preview_h)
    if not np.any(valid):
        return preview

    euler = dic_coords.loc[valid, ["phi1", "PHI", "phi2"]].to_numpy(dtype=np.float32)
    rgb_values = np.column_stack(
        [
            normalize_vector_to_uint8(euler[:, 0]),
            normalize_vector_to_uint8(euler[:, 1]),
            normalize_vector_to_uint8(euler[:, 2]),
        ]
    )
    preview[ys[valid], xs[valid]] = rgb_values
    scatter_mask[ys[valid], xs[valid]] = True

    footprint = projected_ebsd_footprint_mask(dic_coords, preview.shape[:2], step)
    fill_mask = footprint & ~scatter_mask
    if np.any(fill_mask):
        _, nearest_indices = ndi.distance_transform_edt(~scatter_mask, return_indices=True)
        preview[fill_mask] = preview[
            nearest_indices[0][fill_mask],
            nearest_indices[1][fill_mask],
        ]
    return preview


def projected_ebsd_footprint_mask(dic_coords: pd.DataFrame, shape: tuple[int, int], step: int) -> np.ndarray:
    height, width = shape
    footprint = np.zeros((height, width), dtype=bool)
    if dic_coords.empty:
        return footprint

    x_ebsd = dic_coords["x_ebsd"].to_numpy(dtype=np.float64)
    y_ebsd = dic_coords["y_ebsd"].to_numpy(dtype=np.float64)
    x_min, x_max = float(np.nanmin(x_ebsd)), float(np.nanmax(x_ebsd))
    y_min, y_max = float(np.nanmin(y_ebsd)), float(np.nanmax(y_ebsd))
    eps = 0.51
    boundary = pd.concat(
        [
            dic_coords[np.abs(dic_coords["y_ebsd"] - y_min) <= eps].sort_values("x_ebsd"),
            dic_coords[np.abs(dic_coords["x_ebsd"] - x_max) <= eps].sort_values("y_ebsd"),
            dic_coords[np.abs(dic_coords["y_ebsd"] - y_max) <= eps].sort_values("x_ebsd", ascending=False),
            dic_coords[np.abs(dic_coords["x_ebsd"] - x_min) <= eps].sort_values("y_ebsd", ascending=False),
        ],
        ignore_index=True,
    )
    if len(boundary) < 3:
        return footprint

    points = [
        (int(round(x / step)), int(round(y / step)))
        for x, y in zip(boundary["x_dic"], boundary["y_dic"])
    ]
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon(points, fill=1)
    return np.asarray(image, dtype=bool)


@st.cache_data(show_spinner=False)
def load_ang_trace_vector_data(path_str: str, chunk_size: int = 250_000, schema_version: int = 2) -> pd.DataFrame:
    _ = schema_version
    ang_core = load_ang_core_data(path_str)
    data = ang_core["data"]
    chunks = []
    for start in range(0, len(data), chunk_size):
        chunk = data.iloc[start:start + chunk_size]
        chunks.append(add_slip_trace_vectors_to_ang_chunk(chunk))
    if not chunks:
        return pd.DataFrame(
            columns=[
                "x_ebsd",
                "y_ebsd",
                "phi1",
                "PHI",
                "phi2",
                "prism",
                "basal",
            ]
        )
    return pd.concat(chunks, ignore_index=True)


def add_slip_trace_vectors_to_ang_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    euler = chunk[["phi1", "PHI", "phi2"]].to_numpy(dtype=np.float64)
    prism, basal = calc_slip_trace_vectors_for_euler(euler)
    return pd.DataFrame(
        {
            "x_ebsd": chunk["x_grid"].to_numpy(dtype=np.float32),
            "y_ebsd": chunk["y_grid"].to_numpy(dtype=np.float32),
            "phi1": chunk["phi1"].to_numpy(dtype=np.float32),
            "PHI": chunk["PHI"].to_numpy(dtype=np.float32),
            "phi2": chunk["phi2"].to_numpy(dtype=np.float32),
            "prism": [
                vector_matrix_to_list(value)
                for value in prism.astype(np.float32)
            ],
            "basal": [
                vector_matrix_to_list(value.reshape(1, 3))
                for value in basal.astype(np.float32)
            ],
        }
    )


@st.cache_data(show_spinner=False)
def build_dic_space_trace_vector_data(
    dic_coords: pd.DataFrame,
    forward_params: np.ndarray,
    chunk_size: int = 250_000,
    schema_version: int = 1,
) -> pd.DataFrame:
    _ = schema_version
    chunks = []
    for start in range(0, len(dic_coords), chunk_size):
        chunk = dic_coords.iloc[start:start + chunk_size]
        chunks.append(add_dic_space_slip_trace_vectors_to_chunk(chunk, forward_params))
    if not chunks:
        return pd.DataFrame(
            columns=[
                "x_dic",
                "y_dic",
                "x_ebsd",
                "y_ebsd",
                "phi1",
                "PHI",
                "phi2",
                "prism",
                "basal",
            ]
        )
    return pd.concat(chunks, ignore_index=True)


def add_dic_space_slip_trace_vectors_to_chunk(chunk: pd.DataFrame, forward_params: np.ndarray) -> pd.DataFrame:
    _ = forward_params
    euler = chunk[["phi1", "PHI", "phi2"]].to_numpy(dtype=np.float64)
    prism_ebsd, basal_ebsd = calc_slip_trace_vectors_for_euler(euler)
    return pd.DataFrame(
        {
            "x_dic": chunk["x_dic"].to_numpy(dtype=np.float32),
            "y_dic": chunk["y_dic"].to_numpy(dtype=np.float32),
            "x_ebsd": chunk["x_ebsd"].to_numpy(dtype=np.float32),
            "y_ebsd": chunk["y_ebsd"].to_numpy(dtype=np.float32),
            "phi1": chunk["phi1"].to_numpy(dtype=np.float32),
            "PHI": chunk["PHI"].to_numpy(dtype=np.float32),
            "phi2": chunk["phi2"].to_numpy(dtype=np.float32),
            "prism": [
                vector_matrix_to_list(value)
                for value in prism_ebsd.astype(np.float32)
            ],
            "basal": [
                vector_matrix_to_list(value.reshape(1, 3))
                for value in basal_ebsd.astype(np.float32)
            ],
        }
    )


def push_trace_vectors_forward_to_dic(
    prism_ebsd: np.ndarray,
    basal_ebsd: np.ndarray,
    x_ebsd: np.ndarray,
    y_ebsd: np.ndarray,
    forward_params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prism_dic = np.zeros_like(prism_ebsd, dtype=np.float64)
    basal_dic = np.zeros_like(basal_ebsd, dtype=np.float64)
    for system_index in range(prism_ebsd.shape[1]):
        prism_dic[:, system_index, :] = push_2d_vectors_with_forward_jacobian(
            prism_ebsd[:, system_index, :],
            x_ebsd,
            y_ebsd,
            forward_params,
        )
    basal_dic[:, :] = push_2d_vectors_with_forward_jacobian(
        basal_ebsd,
        x_ebsd,
        y_ebsd,
        forward_params,
    )
    return prism_dic, basal_dic


def push_2d_vectors_with_forward_jacobian(
    vectors: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    a = params[0, :6]
    b = params[1, :6]
    dx_dx = a[1] + 2.0 * a[3] * x + a[4] * y
    dx_dy = a[2] + a[4] * x + 2.0 * a[5] * y
    dy_dx = b[1] + 2.0 * b[3] * x + b[4] * y
    dy_dy = b[2] + b[4] * x + 2.0 * b[5] * y

    vx = vectors[:, 0]
    vy = vectors[:, 1]
    pushed = np.zeros_like(vectors, dtype=np.float64)
    pushed[:, 0] = dx_dx * vx + dx_dy * vy
    pushed[:, 1] = dy_dx * vx + dy_dy * vy
    norms = np.linalg.norm(pushed[:, :2], axis=1)
    safe = norms > 1e-12
    pushed[safe, :2] /= norms[safe, None]
    pushed[~safe, :2] = 0.0
    pushed[:, 2] = 0.0
    return pushed


def vector_matrix_to_list(value: np.ndarray) -> list:
    return np.round(value.astype(float), 7).tolist()


def display_trace_vector_table(trace_data: pd.DataFrame) -> pd.DataFrame:
    display = trace_data.copy()
    for col in ["prism", "basal"]:
        if col in display.columns:
            display[col] = display[col].apply(format_matrix_cell)
    return display


def format_matrix_cell(value) -> str:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    shape = f"{arr.shape[0]}x{arr.shape[1]}"
    rows = []
    for row in arr:
        rows.append("[" + ", ".join(f"{item:.6f}" for item in row) + "]")
    return shape + "\n[\n  " + "\n  ".join(rows) + "\n]"


def matrix_shape_label(value) -> str:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return f"{arr.shape[0]} x {arr.shape[1]}"


def calc_slip_trace_vectors_for_euler(bunge_euler_radians: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if bunge_euler_radians.ndim == 1:
        bunge_euler_radians = bunge_euler_radians.reshape(1, 3)

    rmat_90_z = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rmat_neg180_x = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    frame_correction = rmat_neg180_x @ rmat_90_z

    prismatic_normals = np.array(
        [
            [0.866025403784, 0.50, 0.00],
            [0.00, 1.00, 0.00],
            [-0.866025403784, 0.50, 0.00],
        ],
        dtype=np.float64,
    )
    basal_normal = np.array([0.00, 0.00, 1.00], dtype=np.float64)

    rotations = R.from_euler("ZXZ", bunge_euler_radians, degrees=False).as_matrix()
    prism_traces = np.zeros((len(bunge_euler_radians), 3, 3), dtype=np.float64)
    for index, normal in enumerate(prismatic_normals):
        normal_lab = rotations @ normal
        normal_image = normal_lab @ frame_correction.T
        prism_traces[:, index, :] = trace_from_projected_plane_normals(normal_image)

    basal_lab = rotations @ basal_normal
    basal_image = basal_lab @ frame_correction.T
    basal_trace = trace_from_projected_plane_normals(basal_image)
    return prism_traces, basal_trace


def trace_from_projected_plane_normals(normals: np.ndarray) -> np.ndarray:
    projected = np.zeros_like(normals, dtype=np.float64)
    projected[:, 0] = normals[:, 0]
    projected[:, 1] = normals[:, 1]
    norms = np.linalg.norm(projected[:, :2], axis=1)
    safe = norms > 1e-12
    projected[safe, :2] /= norms[safe, None]
    projected[~safe, :] = 0.0
    image_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return np.cross(projected, image_normal)


@st.cache_data(show_spinner=False)
def parse_ang_header_metadata(path_str: str) -> dict:
    numeric_keys = {
        "XSTEP",
        "YSTEP",
        "NCOLS_ODD",
        "NCOLS_EVEN",
        "NROWS",
        "COLUMN_COUNT",
        "TEM_PIXperUM",
    }
    metadata: dict[str, float | int | str] = {}
    with Path(path_str).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "# HEADER: End":
                break
            if not stripped.startswith("#"):
                continue
            content = stripped[1:].strip()
            if ":" in content:
                key, value = content.split(":", 1)
            else:
                parts = content.split(None, 1)
                if len(parts) != 2:
                    continue
                key, value = parts
            key = key.strip()
            value = value.strip()
            if key in numeric_keys:
                metadata[key] = parse_ang_numeric_value(value)
            elif key in {"GRID", "VERSION", "COLUMN_HEADERS", "COLUMN_UNITS"}:
                metadata[key] = value
    return metadata


def parse_ang_numeric_value(value: str) -> float | int | str:
    token = value.split()[0] if value.split() else ""
    try:
        number = float(token)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def show_grx_overlay_view(
    result: dict,
    event_result: dict,
    title: str,
    key: str,
) -> None:
    grx_overlay = result["grx_overlay"]
    if grx_overlay is None:
        st.info("Add a GRX overlay image path in the sidebar to show this view.")
        return

    grx_event_overlay = draw_event_overlay(
        grx_overlay["display_rgb"],
        event_result["accepted"],
        event_result.get("standalone_seeds", []),
        event_result.get("merged_seeds", []),
        event_result.get("merged_groups", []),
    )
    st.caption(
        f"{grx_overlay['note']} | origin x,y: {grx_overlay['origin']} | "
        "events were detected from the BLN image"
    )
    zoomable_image(grx_event_overlay, title, key=key)


def zoomable_image(image: np.ndarray, title: str, key: str) -> None:
    root_id = f"zoom-{key}".replace("_", "-")
    canvas_id = f"{root_id}-canvas"
    status_id = f"{root_id}-status"
    data_url = image_to_png_data_url(image)
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html_doc = """
<div id="__ROOT_ID__" class="zoom-root" tabindex="0">
  <div class="zoom-title">__TITLE__</div>
  <canvas id="__CANVAS_ID__"></canvas>
  <div id="__STATUS_ID__" class="zoom-status">Zoom 1.00x | x: -, y: -</div>
</div>
<style>
  .zoom-root {
    width: 100%;
    outline: none;
    border: 1px solid #d8dee9;
    border-radius: 8px;
    background: #101418;
    overflow: hidden;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  .zoom-title {
    color: #eef2f5;
    font-size: 14px;
    font-weight: 600;
    padding: 8px 10px;
    background: #1d252d;
  }
  .zoom-root canvas {
    display: block;
    width: 100%;
    cursor: crosshair;
    background: #050607;
  }
  .zoom-status {
    color: #c9d1d9;
    font-size: 12px;
    padding: 6px 10px;
    background: #1d252d;
  }
</style>
<script>
(() => {
  const root = document.getElementById("__ROOT_ID__");
  const canvas = document.getElementById("__CANVAS_ID__");
  const status = document.getElementById("__STATUS_ID__");
  const ctx = canvas.getContext("2d");
  const img = new Image();
  const dpr = window.devicePixelRatio || 1;

  let scale = 1;
  let fitScale = 1;
  let offsetX = 0;
  let offsetY = 0;
  let cursorX = 0;
  let cursorY = 0;
  let imageCursorX = null;
  let imageCursorY = null;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  function cssWidth() {
    return canvas.width / dpr;
  }

  function cssHeight() {
    return canvas.height / dpr;
  }

  function resizeCanvas() {
    const width = Math.max(320, Math.floor(root.clientWidth));
    const height = Math.min(620, Math.max(380, Math.floor(width * 0.72)));
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fitImage();
  }

  function fitImage() {
    if (!img.width || !img.height) {
      return;
    }
    fitScale = Math.min(cssWidth() / img.width, cssHeight() / img.height);
    scale = fitScale;
    offsetX = (cssWidth() - img.width * scale) / 2;
    offsetY = (cssHeight() - img.height * scale) / 2;
    draw();
  }

  function clampView() {
    const viewW = cssWidth();
    const viewH = cssHeight();
    const imgW = img.width * scale;
    const imgH = img.height * scale;

    if (imgW <= viewW) {
      offsetX = (viewW - imgW) / 2;
    } else {
      offsetX = Math.min(0, Math.max(viewW - imgW, offsetX));
    }

    if (imgH <= viewH) {
      offsetY = (viewH - imgH) / 2;
    } else {
      offsetY = Math.min(0, Math.max(viewH - imgH, offsetY));
    }
  }

  function updateImageCursor() {
    imageCursorX = (cursorX - offsetX) / scale;
    imageCursorY = (cursorY - offsetY) / scale;
    if (
      imageCursorX < 0 ||
      imageCursorY < 0 ||
      imageCursorX >= img.width ||
      imageCursorY >= img.height
    ) {
      imageCursorX = null;
      imageCursorY = null;
    }
  }

  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssWidth(), cssHeight());
    ctx.fillStyle = "#050607";
    ctx.fillRect(0, 0, cssWidth(), cssHeight());
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, offsetX, offsetY, img.width * scale, img.height * scale);

    if (imageCursorX !== null && imageCursorY !== null) {
      ctx.save();
      ctx.strokeStyle = "rgba(255, 230, 80, 0.9)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cursorX - 6, cursorY);
      ctx.lineTo(cursorX + 6, cursorY);
      ctx.moveTo(cursorX, cursorY - 6);
      ctx.lineTo(cursorX, cursorY + 6);
      ctx.stroke();
      ctx.restore();
      status.textContent =
        "Zoom " +
        (scale / fitScale).toFixed(2) +
        "x | x: " +
        Math.round(imageCursorX) +
        ", y: " +
        Math.round(imageCursorY);
    } else {
      status.textContent = "Zoom " + (scale / fitScale).toFixed(2) + "x | x: -, y: -";
    }
  }

  function canvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  }

  function zoomAt(factor, x, y) {
    if (!img.width || !img.height) {
      return;
    }
    const imageX = (x - offsetX) / scale;
    const imageY = (y - offsetY) / scale;
    const nextScale = Math.min(fitScale * 40, Math.max(fitScale, scale * factor));
    scale = nextScale;
    offsetX = x - imageX * scale;
    offsetY = y - imageY * scale;
    clampView();
    updateImageCursor();
    draw();
  }

  canvas.addEventListener("mousemove", (event) => {
    root.focus({ preventScroll: true });
    const point = canvasPoint(event);
    cursorX = point.x;
    cursorY = point.y;
    if (dragging) {
      offsetX += cursorX - lastX;
      offsetY += cursorY - lastY;
      clampView();
    }
    lastX = cursorX;
    lastY = cursorY;
    updateImageCursor();
    draw();
  });

  canvas.addEventListener("mousedown", (event) => {
    const point = canvasPoint(event);
    dragging = true;
    lastX = point.x;
    lastY = point.y;
  });

  window.addEventListener("mouseup", () => {
    dragging = false;
  });

  root.addEventListener("mouseenter", () => {
    root.focus({ preventScroll: true });
  });

  root.addEventListener("keydown", (event) => {
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      zoomAt(1.25, cursorX || cssWidth() / 2, cursorY || cssHeight() / 2);
    } else if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      zoomAt(0.8, cursorX || cssWidth() / 2, cursorY || cssHeight() / 2);
    } else if (event.key === "0") {
      event.preventDefault();
      fitImage();
    }
  });

  img.onload = resizeCanvas;
  img.src = "__DATA_URL__";
  window.addEventListener("resize", resizeCanvas);
})();
</script>
"""
    html_doc = (
        html_doc.replace("__ROOT_ID__", root_id)
        .replace("__CANVAS_ID__", canvas_id)
        .replace("__STATUS_ID__", status_id)
        .replace("__TITLE__", safe_title)
        .replace("__DATA_URL__", data_url)
    )
    components.html(html_doc, height=700, scrolling=False)


def image_to_png_data_url(image: np.ndarray) -> str:
    if image.dtype == np.uint8:
        arr = image
    else:
        arr = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = gray_to_rgb(arr)
    image_buffer = BytesIO()
    Image.fromarray(arr[..., :3]).save(image_buffer, format="PNG")
    encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_title_with_dimensions(title: str, image: np.ndarray) -> str:
    height, width = image.shape[:2]
    return f"{title} ({width} x {height})"


def draw_hough_overlay(rgb: np.ndarray, lines: list, seeds: list[Point]) -> np.ndarray:
    out = rgb.astype(np.float32) / 255.0
    out = out.copy()
    line_mask = np.zeros(out.shape[:2], dtype=bool)
    for p0, p1 in lines:
        x0, y0 = p0
        x1, y1 = p1
        rr, cc = draw_line(y0, x0, y1, x1)
        valid = (rr >= 0) & (rr < out.shape[0]) & (cc >= 0) & (cc < out.shape[1])
        line_mask[rr[valid], cc[valid]] = True
    line_mask = ndi.binary_dilation(line_mask, iterations=2)
    out[line_mask] = [1.0, 0.0, 0.0]
    return draw_points(out, seeds, color=(1.0, 1.0, 0.0), radius=4)


def draw_seed_overlay(rgb: np.ndarray, detection_mask: np.ndarray, seeds: list[Point]) -> np.ndarray:
    out = rgb.astype(np.float32) / 255.0
    out = out.copy()
    mask_vis = ndi.binary_dilation(detection_mask, iterations=1)
    out[mask_vis] = [1.0, 0.0, 0.0]
    return draw_points(out, seeds, color=(0.0, 1.0, 1.0), radius=4)


def draw_event_overlay(
    rgb: np.ndarray,
    lines: list[DicLine],
    standalone_seeds: list[Point],
    merged_seeds: list[Point],
    merged_groups: list[dict] | None = None,
) -> np.ndarray:
    out = rgb.astype(np.float32) / 255.0
    out = out.copy()
    mask = np.zeros(out.shape[:2], dtype=bool)
    for line in lines:
        for point in line.points:
            if 0 <= point.x < mask.shape[1] and 0 <= point.y < mask.shape[0]:
                mask[point.y, point.x] = True
    merged_mask = merged_event_mask(out.shape[:2], merged_groups or [])
    if merged_mask.any():
        merged_halo = ndi.binary_dilation(merged_mask, iterations=5)
        merged_inner = ndi.binary_dilation(merged_mask, iterations=2)
        out[merged_halo & ~merged_inner] = [0.0, 0.25, 1.0]
    mask = ndi.binary_dilation(mask, iterations=2)
    out[mask] = [1.0, 0.0, 0.0]
    out = draw_points(
        out,
        standalone_seeds,
        color=(1.0, 1.0, 0.0),
        radius=4,
        outline_color=(1.0, 1.0, 1.0),
    )
    out = draw_points(
        out,
        merged_seeds,
        color=(0.0, 0.25, 1.0),
        radius=4,
        outline_color=(1.0, 1.0, 1.0),
    )
    return draw_merged_group_labels(out, merged_groups or [])


def merged_event_mask(shape: tuple[int, int], merged_groups: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for group in merged_groups:
        line = group.get("line")
        if line is None:
            continue
        for point in line.points:
            if 0 <= point.x < shape[1] and 0 <= point.y < shape[0]:
                mask[point.y, point.x] = True
    return mask


def draw_merged_group_labels(image: np.ndarray, merged_groups: list[dict]) -> np.ndarray:
    if not merged_groups:
        return image

    arr = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    pil_image = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil_image)

    for group_index, group in enumerate(merged_groups, start=1):
        label = f"M{group_index}"
        for seed in group.get("seeds", []):
            x = int(seed.x) + 6
            y = int(seed.y) - 12
            draw.rectangle((x - 1, y - 1, x + 7 * len(label), y + 10), fill=(0, 64, 255))
            draw.text((x + 1, y - 1), label, fill=(255, 255, 255))
    return np.asarray(pil_image).astype(np.float32) / 255.0


def draw_points(
    image: np.ndarray,
    seeds: list[Point],
    color: tuple[float, float, float],
    radius: int,
    outline_color: tuple[float, float, float] | None = None,
) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for seed in seeds:
        if outline_color is not None:
            outline_radius = radius + 1
            y0 = max(0, seed.y - outline_radius)
            y1 = min(h, seed.y + outline_radius + 1)
            x0 = max(0, seed.x - outline_radius)
            x1 = min(w, seed.x + outline_radius + 1)
            out[y0:y1, x0:x1] = outline_color
        y0 = max(0, seed.y - radius)
        y1 = min(h, seed.y + radius + 1)
        x0 = max(0, seed.x - radius)
        x1 = min(w, seed.x + radius + 1)
        out[y0:y1, x0:x1] = color
    return out


def to_display(image: np.ndarray, cmap: str | None = None, max_dim: int = 1400) -> np.ndarray:
    if image.ndim == 2:
        arr = normalize_for_display(image)
        if cmap == "magma":
            import matplotlib.pyplot as plt

            arr = (plt.get_cmap("magma")(arr)[..., :3] * 255).astype(np.uint8)
        else:
            arr = gray_to_rgb((arr * 255).astype(np.uint8))
    else:
        arr = image
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    step = max(1, int(np.ceil(max(arr.shape[:2]) / max_dim)))
    return arr[::step, ::step]


def mask_for_visibility(mask: np.ndarray, dilate: bool, max_dim: int = 1400) -> np.ndarray:
    display_mask = mask.astype(bool, copy=False)
    if not dilate:
        return display_mask
    downsample_step = max(1, int(np.ceil(max(display_mask.shape[:2]) / max_dim)))
    iterations = max(1, downsample_step)
    return ndi.binary_dilation(display_mask, iterations=iterations).astype(bool, copy=False)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    if image.dtype == bool:
        return image.astype(np.float32)
    values = image.astype(np.float32, copy=False)
    lo, hi = np.percentile(values, [1, 99.5])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(values, [1.0, 99.7])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def scalar_to_uint8(arr: np.ndarray, low_pct: float, high_pct: float, gamma: float) -> np.ndarray:
    normalized = robust_percentile(arr, low_pct, high_pct)
    normalized = np.power(np.clip(normalized, 0.0, 1.0), gamma).astype(np.float32)
    return (normalized * 255).astype(np.uint8)


def direct_to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    values = np.clip(values, 0, 255).astype(np.uint8)
    if values.ndim == 2:
        return gray_to_rgb(values)
    if values.ndim == 3 and values.shape[2] >= 3:
        return values[..., :3]
    if values.ndim == 3 and values.shape[2] == 1:
        return gray_to_rgb(values[..., 0])
    return gray_to_rgb(np.squeeze(values))


def scale_to_uint8_rgb(arr: np.ndarray, display_min: float, display_max: float) -> np.ndarray:
    values = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    display_min = float(display_min)
    display_max = float(display_max)
    if display_max <= display_min:
        scaled = np.zeros(values.shape, dtype=np.uint8)
    else:
        scaled = (
            np.clip((values - display_min) / (display_max - display_min), 0.0, 1.0) * 255
        ).astype(np.uint8)
    if scaled.ndim == 2:
        return gray_to_rgb(scaled)
    if scaled.ndim == 3 and scaled.shape[2] >= 3:
        return scaled[..., :3]
    if scaled.ndim == 3 and scaled.shape[2] == 1:
        return gray_to_rgb(scaled[..., 0])
    return gray_to_rgb(np.squeeze(scaled))


def normalize_to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        normalized = np.zeros(values.shape, dtype=np.uint8)
    else:
        normalized = (np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
    if normalized.ndim == 2:
        return gray_to_rgb(normalized)
    if normalized.ndim == 3 and normalized.shape[2] >= 3:
        return normalized[..., :3]
    if normalized.ndim == 3 and normalized.shape[2] == 1:
        return gray_to_rgb(normalized[..., 0])
    return gray_to_rgb(np.squeeze(normalized))


def array_stats(arr: np.ndarray) -> dict[str, float]:
    values = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def robust_percentile(arr: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    values = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(values, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    rgb = arr[..., :3]
    if rgb.dtype == np.uint8:
        return rgb.astype(np.uint8, copy=False)
    return normalize_to_uint8_rgb(rgb)


def display_crop_rgb(
    arr: np.ndarray,
    display_min: float | None = None,
    display_max: float | None = None,
) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return to_uint8_rgb(arr)
    if arr.dtype == np.uint8:
        return gray_to_rgb(np.squeeze(arr))
    if display_min is not None and display_max is not None:
        return scale_to_uint8_rgb(np.squeeze(arr), display_min, display_max)
    return normalize_to_uint8_rgb(np.squeeze(arr))


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=2)


def remove_small_objects_by_label(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels = measure.label(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    if keep.size:
        keep[0] = False
    return keep[labels]


def select_spaced_seeds(
    seed_candidates: list[tuple[int, int, float]],
    shape: tuple[int, int],
    seed_spacing: int,
    max_seeds: int,
) -> list[Point]:
    if max_seeds <= 0:
        return []
    strongest_by_pixel: dict[tuple[int, int], float] = {}
    for row, col, score in seed_candidates:
        key = (row, col)
        strongest_by_pixel[key] = max(score, strongest_by_pixel.get(key, float("-inf")))
    seed_candidates = [(row, col, score) for (row, col), score in strongest_by_pixel.items()]
    seed_candidates.sort(key=lambda item: item[2], reverse=True)
    blocked = np.zeros(shape, dtype=bool)
    accepted: list[Point] = []
    radius = max(1, seed_spacing)
    h, w = shape

    for row, col, _score in seed_candidates:
        if row < 0 or row >= h or col < 0 or col >= w:
            continue
        if blocked[row, col]:
            continue
        accepted.append(Point(x=col, y=row))
        if len(accepted) >= max_seeds:
            break
        y0 = max(0, row - radius)
        y1 = min(h, row + radius + 1)
        x0 = max(0, col - radius)
        x1 = min(w, col + radius + 1)
        blocked[y0:y1, x0:x1] = True
    return accepted


def should_merge_events(candidate: DicLine, existing: DicLine, params: PipelineParams) -> bool:
    if line_overlap_fraction(candidate, existing) >= params.duplicate_overlap:
        return True
    if params.merge_distance_tolerance <= 0 or params.merge_angle_tolerance <= 0:
        return False

    candidate_angle = line_angle_degrees(candidate)
    existing_angle = line_angle_degrees(existing)
    if candidate_angle is None or existing_angle is None:
        return False
    if angle_difference_degrees(candidate_angle, existing_angle) > params.merge_angle_tolerance:
        return False

    return line_min_distance(candidate, existing) <= params.merge_distance_tolerance


def merge_event_lines(primary: DicLine, secondary: DicLine, connect_gaps: bool = True) -> DicLine:
    points = set(primary.points) | set(secondary.points)
    if connect_gaps:
        points = connect_points_by_endpoint_bridges(points)
    return DicLine(
        is_manual=False,
        points=points,
        seed_point=primary.seed_point or secondary.seed_point,
        intensity_difference_tolerance=primary.intensity_difference_tolerance,
        bfl_tolerance=primary.bfl_tolerance,
        min_intensity=primary.min_intensity,
    )


def connect_points_by_endpoint_bridges(points: set[Point]) -> set[Point]:
    if len(points) < 2:
        return points

    mask, origin_x, origin_y = points_to_local_mask(points)
    connected = connect_mask_components_by_endpoint_bridges(mask)
    return local_mask_to_points(connected, origin_x, origin_y)


def points_to_local_mask(points: set[Point], padding: int = 1) -> tuple[np.ndarray, int, int]:
    xs = np.array([point.x for point in points], dtype=int)
    ys = np.array([point.y for point in points], dtype=int)
    origin_x = int(xs.min()) - padding
    origin_y = int(ys.min()) - padding
    width = int(xs.max()) - origin_x + padding + 1
    height = int(ys.max()) - origin_y + padding + 1
    mask = np.zeros((height, width), dtype=bool)
    mask[ys - origin_y, xs - origin_x] = True
    return mask, origin_x, origin_y


def local_mask_to_points(mask: np.ndarray, origin_x: int, origin_y: int) -> set[Point]:
    rows, cols = np.where(mask)
    return {Point(x=int(origin_x + col), y=int(origin_y + row)) for row, col in zip(rows, cols)}


def connect_mask_components_by_endpoint_bridges(mask: np.ndarray) -> np.ndarray:
    connected = mask.astype(bool, copy=True)

    while True:
        labels = measure.label(connected, connectivity=2)
        component_count = int(labels.max())
        if component_count <= 1:
            return connected

        endpoint_sets = [
            component_endpoint_candidates(labels == component_id)
            for component_id in range(1, component_count + 1)
        ]
        bridge = closest_endpoint_bridge(endpoint_sets)
        if bridge is None:
            return connected

        point_a, point_b = bridge
        rr, cc = draw_line(
            int(point_a[0]),
            int(point_a[1]),
            int(point_b[0]),
            int(point_b[1]),
        )
        connected[rr, cc] = True


def component_endpoint_candidates(component_mask: np.ndarray) -> np.ndarray:
    skeleton = morphology.skeletonize(component_mask.astype(bool, copy=False))
    coords = set(map(tuple, np.argwhere(skeleton)))
    if not coords:
        return np.argwhere(component_mask)

    endpoints = []
    for row, col in coords:
        neighbor_count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (row + dr, col + dc) in coords:
                    neighbor_count += 1
        if neighbor_count == 1:
            endpoints.append((row, col))

    if endpoints:
        return np.array(endpoints, dtype=int)
    return np.array(sorted(coords), dtype=int)


def closest_endpoint_bridge(endpoint_sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    best_pair = None
    best_distance = float("inf")

    for index_a, endpoints_a in enumerate(endpoint_sets):
        if len(endpoints_a) == 0:
            continue
        tree = cKDTree(endpoints_a.astype(np.float32))
        for endpoints_b in endpoint_sets[index_a + 1:]:
            if len(endpoints_b) == 0:
                continue
            distances, nearest_indices = tree.query(endpoints_b.astype(np.float32), k=1)
            best_b_index = int(np.argmin(distances))
            if float(distances[best_b_index]) < best_distance:
                best_distance = float(distances[best_b_index])
                best_pair = (
                    endpoints_a[int(nearest_indices[best_b_index])],
                    endpoints_b[best_b_index],
                )

    return best_pair


def line_angle_degrees(line: DicLine) -> float | None:
    if len(line.points) < 2:
        return None
    coords = np.array([(point.x, point.y) for point in line.points], dtype=np.float32)
    centered = coords - coords.mean(axis=0)
    if not np.any(centered):
        return None
    _, _singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    dx, dy = vh[0]
    return math.degrees(math.atan2(float(dy), float(dx))) % 180.0


def angle_difference_degrees(angle_a: float, angle_b: float) -> float:
    diff = abs(angle_a - angle_b) % 180.0
    return min(diff, 180.0 - diff)


def line_min_distance(line_a: DicLine, line_b: DicLine) -> float:
    if not line_a.points or not line_b.points:
        return float("inf")
    if line_a.points & line_b.points:
        return 0.0

    coords_a = np.array([(point.y, point.x) for point in line_a.points], dtype=np.float32)
    coords_b = np.array([(point.y, point.x) for point in line_b.points], dtype=np.float32)
    if len(coords_a) > len(coords_b):
        coords_a, coords_b = coords_b, coords_a
    distances, _indices = cKDTree(coords_b).query(coords_a, k=1)
    return float(np.min(distances))


def point_in_any_line(point: Point, lines: list[DicLine], tolerance: float) -> bool:
    threshold = tolerance * tolerance
    for line in lines:
        for line_point in line.points:
            if (point.x - line_point.x) ** 2 + (point.y - line_point.y) ** 2 <= threshold:
                return True
    return False


def line_overlap_fraction(line_a: DicLine, line_b: DicLine) -> float:
    if not line_a.points or not line_b.points:
        return 0.0
    overlap = len(line_a.points & line_b.points)
    return overlap / min(len(line_a.points), len(line_b.points))


if __name__ == "__main__":
    main()
