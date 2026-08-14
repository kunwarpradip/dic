from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage import measure


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import DIC_check_slip_twin_traces_hcp as hcp_trace_module  # noqa: E402
from DIC_check_slip_traces_bcc import calcSlipTracesBCC110, calcSlipTracesBCC112  # noqa: E402


HCP_MILLER_INDEX_DIR = ROOT / "hcp_slip_twin_miller_indices"
hcp_trace_module.HCP_MILLER_INDEX_DIR = HCP_MILLER_INDEX_DIR

calcSlipTracesHCPBasal = hcp_trace_module.calcSlipTracesHCPBasal
calcSlipTracesHCPPrism = hcp_trace_module.calcSlipTracesHCPPrism
calcSlipTracesHCPPyra_I_A = hcp_trace_module.calcSlipTracesHCPPyra_I_A
calcSlipTracesHCPPyra_II_CA = hcp_trace_module.calcSlipTracesHCPPyra_II_CA
calcTwinTracesHCP = hcp_trace_module.calcTwinTracesHCP


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

EVENT_SHAPE_TYPE_OPTIONS = ["linear", "blob_like", "irregular", "fragmented", "small_noise"]
DEFAULT_SCORING_SHAPE_TYPES = ["linear", "irregular", "fragmented", "small_noise"]


def default_crystal_config() -> dict[str, Any]:
    return {
        "crystal_structure": "HCP (Hexagonal Close Packed)",
        "ebsd_reference_frame": EBSD_REFERENCE_FRAMES[0],
        "active_modes": DEFAULT_CRYSTAL_MODES["HCP (Hexagonal Close Packed)"].copy(),
        "hcp_lattice_a": 2.95,
        "hcp_lattice_c": 4.686,
        "slip_angle_tolerance_deg": 5.0,
        "twin_angle_tolerance_deg": 7.0,
    }


def load_alignment_transform_parameters(path_str: str) -> dict:
    with Path(path_str).expanduser().open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    forward_params = np.asarray(data.get("forward_transform_ebsd_to_dic", {}).get("params", []), dtype=float)
    inverse_params = np.asarray(data.get("inverse_map_dic_to_ebsd", {}).get("params", []), dtype=float)
    if forward_params.shape[0] != 2 or inverse_params.shape[0] != 2:
        raise ValueError("Transformation JSON does not contain valid forward and inverse parameter matrices.")
    return {
        "metadata": {
            "transform_family": data.get("transform_family"),
            "polynomial_order": data.get("polynomial_order"),
            "matched_points": len(data.get("matched_point_ids") or []),
            "rmse_forward_pixels": data.get("rmse_forward_pixels"),
            "rmse_inverse_pixels": data.get("rmse_inverse_pixels"),
            "dic_image_shape": data.get("dic_image_shape"),
            "ebsd_image_shape": data.get("ebsd_image_shape"),
        },
        "forward_params": forward_params,
        "inverse_params": inverse_params,
    }


def parse_ang_header_metadata(path_str: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with Path(path_str).expanduser().open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("#"):
                break
            header_text = stripped[1:].strip()
            match = re.match(r"^([A-Za-z0-9_]+)\s*[:=]?\s*(.+?)\s*$", header_text)
            if match:
                key = match.group(1).upper()
                value = match.group(2).split()[0]
                metadata[key] = value
    return metadata


def load_ang_core_data(path_str: str) -> dict:
    metadata = parse_ang_header_metadata(path_str)
    data = pd.read_csv(
        Path(path_str).expanduser(),
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
    return {"metadata": metadata, "data": data, "coordinate_normalization": coordinate_normalization}


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
    out["x_zeroed"] = out["x"] - x_origin
    out["y_zeroed"] = out["y"] - y_origin
    out["x_grid"] = out["x_zeroed"] / xstep
    out["y_grid"] = out["y_zeroed"] / ystep
    out["x_grid_round"] = np.rint(out["x_grid"]).astype(np.int64)
    out["y_grid_round"] = np.rint(out["y_grid"]).astype(np.int64)
    return out, {
        "xstep": xstep,
        "ystep": ystep,
        "x_origin": x_origin,
        "y_origin": y_origin,
        "x_grid_max": float(out["x_grid"].max()),
        "y_grid_max": float(out["y_grid"].max()),
    }


def metadata_int(metadata: dict, key: str) -> int | None:
    try:
        return int(float(metadata.get(key)))
    except (TypeError, ValueError):
        return None


def ang_grid_to_ebsd_pixel_mapping(ang_metadata: dict, transform_metadata: dict, data: pd.DataFrame) -> dict:
    ang_ncols = max([v for v in [metadata_int(ang_metadata, "NCOLS_ODD"), metadata_int(ang_metadata, "NCOLS_EVEN")] if v], default=None)
    ang_nrows = metadata_int(ang_metadata, "NROWS")
    if not ang_ncols:
        ang_ncols = int(round(float(data["x_grid"].max()))) + 1
    if not ang_nrows:
        ang_nrows = int(round(float(data["y_grid"].max()))) + 1
    ebsd_shape = transform_metadata.get("ebsd_image_shape") or []
    if len(ebsd_shape) >= 2:
        ebsd_width, ebsd_height = int(ebsd_shape[1]), int(ebsd_shape[0])
        x_scale = (ebsd_width - 1.0) / max(ang_ncols - 1.0, 1.0)
        y_scale = (ebsd_height - 1.0) / max(ang_nrows - 1.0, 1.0)
    else:
        ebsd_width = ebsd_height = None
        x_scale = y_scale = 1.0
    return {
        "ang_ncols": int(ang_ncols),
        "ang_nrows": int(ang_nrows),
        "ebsd_width": ebsd_width,
        "ebsd_height": ebsd_height,
        "x_scale": float(x_scale),
        "y_scale": float(y_scale),
    }


def apply_quadratic_polynomial_transform(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    terms = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    transformed = terms @ params[:, :6].T
    return transformed[:, 0], transformed[:, 1]


def transform_ang_coordinates_to_dic(ang_data: pd.DataFrame, forward_params: np.ndarray, x_ebsd_scale: float, y_ebsd_scale: float) -> pd.DataFrame:
    x_grid = ang_data["x_grid"].to_numpy(dtype=np.float64)
    y_grid = ang_data["y_grid"].to_numpy(dtype=np.float64)
    x_ebsd = x_grid * float(x_ebsd_scale)
    y_ebsd = y_grid * float(y_ebsd_scale)
    x_dic, y_dic = apply_quadratic_polynomial_transform(x_ebsd, y_ebsd, np.asarray(forward_params, dtype=np.float64))
    return pd.DataFrame({
        "x_ebsd": x_ebsd.astype(np.float32),
        "y_ebsd": y_ebsd.astype(np.float32),
        "x_dic": x_dic.astype(np.float32),
        "y_dic": y_dic.astype(np.float32),
        "phi1": ang_data["phi1"].to_numpy(dtype=np.float32),
        "PHI": ang_data["PHI"].to_numpy(dtype=np.float32),
        "phi2": ang_data["phi2"].to_numpy(dtype=np.float32),
    })


def dic_coordinate_metrics(dic_coords: pd.DataFrame, dic_width: int | None, dic_height: int | None) -> dict:
    x = dic_coords["x_dic"].to_numpy(dtype=np.float64)
    y = dic_coords["y_dic"].to_numpy(dtype=np.float64)
    if dic_width and dic_height:
        in_bounds = (x >= 0) & (x < dic_width) & (y >= 0) & (y < dic_height)
    else:
        in_bounds = np.zeros(len(dic_coords), dtype=bool)
    return {
        "x_dic_min": float(np.nanmin(x)) if len(x) else 0.0,
        "x_dic_max": float(np.nanmax(x)) if len(x) else 0.0,
        "y_dic_min": float(np.nanmin(y)) if len(y) else 0.0,
        "y_dic_max": float(np.nanmax(y)) if len(y) else 0.0,
        "in_bounds_count": int(in_bounds.sum()),
        "out_of_bounds_count": int(len(dic_coords) - in_bounds.sum()),
    }


def build_trace_preview(dic_coords: pd.DataFrame, max_rows: int, crystal_config: dict | None = None, progress_callback=None) -> pd.DataFrame:
    sample = dic_coords.head(max(1, int(max_rows))).copy()
    rows = []
    total = len(sample)
    for index, row in enumerate(sample.itertuples(index=False), start=1):
        euler = np.array([row.phi1, row.PHI, row.phi2], dtype=np.float64)
        trace_info = trace_candidates_from_crystal_config(euler, crystal_config, fallback_angle_tolerance=5.0)
        labels = [str(label) for label in trace_info["labels"]]
        vectors = np.asarray(trace_info["vectors"], dtype=np.float64)
        vector_map = {
            label: np.round(vector[:2], 6).tolist()
            for label, vector in zip(labels, vectors)
        }
        rows.append({
            "x_dic": float(row.x_dic),
            "y_dic": float(row.y_dic),
            "phi1": float(row.phi1),
            "PHI": float(row.PHI),
            "phi2": float(row.phi2),
            "selected_trace_count": int(len(labels)),
            "trace_labels": ", ".join(labels),
            "trace_vectors": vector_map,
        })
        if progress_callback and (index == total or index % 1000 == 0):
            progress_callback(index, total)
    return pd.DataFrame(rows)


def load_alignment_event_data(events_path: str, event_pixels_path: str) -> dict:
    events = pd.read_csv(Path(events_path).expanduser())
    event_pixels = pd.read_csv(Path(event_pixels_path).expanduser())
    if "event_id" not in events.columns:
        raise ValueError("Events CSV must include an event_id column.")
    required = {"event_id", "pixel_x", "pixel_y"}
    if not required.issubset(event_pixels.columns):
        raise ValueError("Event pixels CSV must include event_id, pixel_x, and pixel_y columns.")
    events = events.copy()
    event_pixels = event_pixels.copy()
    events["event_id"] = events["event_id"].astype(str)
    event_pixels["event_id"] = event_pixels["event_id"].astype(str)
    event_pixels["pixel_x"] = event_pixels["pixel_x"].astype(np.int64)
    event_pixels["pixel_y"] = event_pixels["pixel_y"].astype(np.int64)
    counts = event_pixels.groupby("event_id", sort=False).size()
    if "num_pixels" not in events.columns:
        events = events.merge(counts.rename("num_pixels"), on="event_id", how="left")
    events["num_pixels"] = events["num_pixels"].fillna(0).astype(np.int64)
    return {
        "events": events,
        "event_pixels": event_pixels,
        "metrics": {
            "event_count": int(events["event_id"].nunique()),
            "pixel_count": int(len(event_pixels)),
            "pixel_event_count": int(event_pixels["event_id"].nunique()),
            "min_pixels_per_event": int(counts.min()) if len(counts) else 0,
            "max_pixels_per_event": int(counts.max()) if len(counts) else 0,
        },
    }


def compute_event_morphology_features(events: pd.DataFrame, event_pixels: pd.DataFrame, min_pixels: int, linearity_threshold: float, aspect_threshold: float, blob_density_threshold: float) -> pd.DataFrame:
    event_lookup = events.set_index(events["event_id"].astype(str), drop=False)
    rows = []
    for event_id, group in event_pixels.groupby("event_id", sort=False):
        coords = group[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float64)
        direction, linearity, event_length = event_direction_from_pixels(coords)
        xs = coords[:, 0] if len(coords) else np.array([0.0])
        ys = coords[:, 1] if len(coords) else np.array([0.0])
        bbox_w = int(xs.max() - xs.min() + 1) if len(coords) else 0
        bbox_h = int(ys.max() - ys.min() + 1) if len(coords) else 0
        density = float(len(coords) / max(1, bbox_w * bbox_h))
        aspect = morphology_aspect_ratio(coords)
        components = event_connected_component_count(coords)
        event_type = classify_event_morphology(len(coords), linearity, aspect, density, components, min_pixels, linearity_threshold, aspect_threshold, blob_density_threshold)
        row = {
            "event_id": str(event_id),
            "event_type": event_type,
            "num_pixels": int(len(coords)),
            "linearity": round(linearity, 4),
            "aspect_ratio": round(aspect, 4),
            "density": round(density, 4),
            "component_count": int(components),
            "event_length_pixels": round(event_length, 3),
            "event_angle_deg": round(vector_angle_degrees(direction), 3) if direction is not None else np.nan,
        }
        if str(event_id) in event_lookup.index:
            source = event_lookup.loc[str(event_id)]
            if isinstance(source, pd.DataFrame):
                source = source.iloc[0]
            for col in ["original_event_id", "cut_index", "method"]:
                if col in source:
                    row[col] = source[col]
        rows.append(row)
    return pd.DataFrame(rows)


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
    direction /= max(np.linalg.norm(direction), 1e-12)
    second = singular_values[1] if len(singular_values) > 1 else 0.0
    linearity = 1.0 - float(second / singular_values[0])
    event_length = float(2.0 * singular_values[0] / max(1.0, np.sqrt(len(coords))))
    return direction, float(np.clip(linearity, 0.0, 1.0)), event_length


def image_vector_to_cartesian(vector: np.ndarray) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float64).copy()
    out[..., 1] *= -1.0
    return out


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
    return int(measure.label(mask, connectivity=2).max())


def classify_event_morphology(num_pixels: int, linearity: float, aspect_ratio: float, density: float, component_count: int, min_pixels: int, linearity_threshold: float, aspect_threshold: float, blob_density_threshold: float) -> str:
    if num_pixels < int(min_pixels):
        return "small_noise"
    if component_count > 1:
        return "fragmented"
    if linearity >= float(linearity_threshold) and aspect_ratio >= float(aspect_threshold):
        return "linear"
    if density >= float(blob_density_threshold):
        return "blob_like"
    return "irregular"


def score_events_with_trace_alignment(events: pd.DataFrame, event_pixels: pd.DataFrame, dic_coords: pd.DataFrame, angle_tolerance_deg: float, max_lookup_distance: float, min_linearity: float, crystal_config: dict | None, morphology: pd.DataFrame | None = None, included_shape_types: set[str] | None = None, progress_callback=None) -> pd.DataFrame:
    if morphology is not None and included_shape_types is not None:
        keep_ids = set(morphology.loc[morphology["event_type"].isin(included_shape_types), "event_id"].astype(str))
        events = events[events["event_id"].astype(str).isin(keep_ids)].copy()
        event_pixels = event_pixels[event_pixels["event_id"].astype(str).isin(keep_ids)].copy()
    trace_points = dic_coords[["x_dic", "y_dic", "x_ebsd", "y_ebsd", "phi1", "PHI", "phi2"]].copy()
    finite = np.isfinite(trace_points[["x_dic", "y_dic", "x_ebsd", "y_ebsd"]].to_numpy(dtype=np.float64)).all(axis=1)
    trace_points = trace_points.loc[finite].reset_index(drop=True)
    if trace_points.empty:
        raise ValueError("No finite transformed ANG/DIC coordinates are available for scoring.")
    tree = cKDTree(trace_points[["x_dic", "y_dic"]].to_numpy(dtype=np.float64))
    rows = []
    groups = list(event_pixels.groupby("event_id", sort=False))
    total = len(groups)
    for index, (event_id, group) in enumerate(groups, start=1):
        coords = group[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float64)
        direction, linearity, event_length = event_direction_from_pixels(coords)
        if direction is None:
            rows.append(empty_event_score_row(str(event_id), len(coords), "too_few_pixels"))
            continue
        event_direction_cart = image_vector_to_cartesian(direction)
        center = coords.mean(axis=0)
        distance, nearest_index = tree.query(center.reshape(1, -1), k=1)
        center_distance = float(np.asarray(distance)[0])
        nearest = trace_points.iloc[int(np.asarray(nearest_index)[0])]
        trace_info = trace_candidates_from_crystal_config(nearest[["phi1", "PHI", "phi2"]].to_numpy(dtype=np.float64), crystal_config, angle_tolerance_deg)
        candidates = normalize_2d_vectors(trace_info["vectors"][:, :2])
        angles = angular_mismatch_values(event_direction_cart, candidates)
        best_index = select_ordered_trace_assignment_index(angles, trace_info["modes"], trace_info["tolerances"], trace_info["mode_priority"])
        best_angle = float(angles[best_index])
        tolerance = float(trace_info["tolerances"][best_index])
        passes = best_angle <= tolerance and center_distance <= float(max_lookup_distance) and linearity >= float(min_linearity)
        rows.append({
            "event_id": str(event_id),
            "classification": "likely_real" if passes else "likely_noise",
            "score": round(float(np.clip(1.0 - best_angle / max(tolerance, 1e-9), 0.0, 1.0)), 4),
            "best_mode": str(trace_info["labels"][best_index]),
            "best_trace_mode": str(trace_info["modes"][best_index]),
            "best_trace_kind": str(trace_info["kinds"][best_index]),
            "num_pixels": int(len(coords)),
            "event_center_x": round(float(center[0]), 3),
            "event_center_y": round(float(center[1]), 3),
            "nearest_ang_x_dic": round(float(nearest["x_dic"]), 3),
            "nearest_ang_y_dic": round(float(nearest["y_dic"]), 3),
            "nearest_ang_x_ebsd": round(float(nearest["x_ebsd"]), 3),
            "nearest_ang_y_ebsd": round(float(nearest["y_ebsd"]), 3),
            "nearest_phi1": round(float(nearest["phi1"]), 8),
            "nearest_Phi": round(float(nearest["PHI"]), 8),
            "nearest_phi2": round(float(nearest["phi2"]), 8),
            "center_lookup_distance": round(center_distance, 3),
            "event_angle_deg": round(vector_angle_degrees(event_direction_cart), 3),
            "event_vector": np.round(event_direction_cart, 6).tolist(),
            "best_trace_vector": np.round(candidates[best_index], 6).tolist(),
            "best_trace_angle_deg": round(vector_angle_degrees(candidates[best_index]), 3),
            "linearity": round(linearity, 4),
            "event_length_pixels": round(event_length, 3),
            "best_angle_error_deg": round(best_angle, 3),
            "angle_tolerance_deg": round(tolerance, 3),
            "passes_angle": bool(best_angle <= tolerance),
            "passes_lookup": bool(center_distance <= float(max_lookup_distance)),
            "passes_linearity": bool(linearity >= float(min_linearity)),
        })
        if progress_callback and (index == total or index % 50 == 0):
            progress_callback(index, total)
    return pd.DataFrame(rows)


def trace_candidates_from_crystal_config(euler_radians: np.ndarray, crystal_config: dict | None, fallback_angle_tolerance: float) -> dict:
    config = normalized_crystal_config(crystal_config, fallback_angle_tolerance)
    if config["ebsd_convention"] != "EDAX":
        raise ValueError("Oxford reference frame is available in the UI but is not implemented in the parent trace functions yet.")
    vectors = []
    labels = []
    modes = []
    kinds = []
    tolerances = []
    for mode in config["active_modes"]:
        mode_vectors, kind = call_trace_function_for_mode(mode, euler_radians, config)
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
    config = crystal_config or default_crystal_config()
    ebsd_reference = str(config.get("ebsd_reference_frame", EBSD_REFERENCE_FRAMES[0]))
    return {
        "crystal_structure": str(config.get("crystal_structure", "HCP (Hexagonal Close Packed)")),
        "ebsd_convention": "EDAX" if ebsd_reference.startswith("EDAX") else "Oxford",
        "active_modes": [mode for mode in config.get("active_modes", []) if isinstance(mode, str)],
        "hcp_lattice_a": float(config.get("hcp_lattice_a") or 2.95),
        "hcp_lattice_c": float(config.get("hcp_lattice_c") or 4.686),
        "slip_tolerance": float(config.get("slip_angle_tolerance_deg") or fallback_angle_tolerance),
        "twin_tolerance": float(config.get("twin_angle_tolerance_deg") or config.get("slip_angle_tolerance_deg") or fallback_angle_tolerance),
    }


def call_trace_function_for_mode(mode: str, euler: np.ndarray, config: dict) -> tuple[np.ndarray, str]:
    if mode.startswith("Basal"):
        return calcSlipTracesHCPBasal(euler, "radians", config["ebsd_convention"]), "slip"
    if mode.startswith("Prismatic"):
        return calcSlipTracesHCPPrism(euler, "radians", config["ebsd_convention"]), "slip"
    if mode.startswith("Pyramidal I"):
        return calcSlipTracesHCPPyra_I_A(euler, "radians", config["hcp_lattice_a"], config["hcp_lattice_c"], config["ebsd_convention"]), "slip"
    if mode.startswith("Pyramidal II"):
        return calcSlipTracesHCPPyra_II_CA(euler, "radians", config["hcp_lattice_a"], config["hcp_lattice_c"], config["ebsd_convention"]), "slip"
    if mode.startswith("{10-12}"):
        return calcTwinTracesHCP(euler, "t1", "radians", config["hcp_lattice_a"], config["hcp_lattice_c"], config["ebsd_convention"]), "twin"
    if mode.startswith("{11-21}"):
        return calcTwinTracesHCP(euler, "t2", "radians", config["hcp_lattice_a"], config["hcp_lattice_c"], config["ebsd_convention"]), "twin"
    if mode.startswith("{11-22}"):
        return calcTwinTracesHCP(euler, "c1", "radians", config["hcp_lattice_a"], config["hcp_lattice_c"], config["ebsd_convention"]), "twin"
    if mode.startswith("{10-11}"):
        return calcTwinTracesHCP(euler, "c2", "radians", config["hcp_lattice_a"], config["hcp_lattice_c"], config["ebsd_convention"]), "twin"
    if mode.startswith("{11-24}"):
        return calcTwinTracesHCP(euler, "c3", "radians", config["hcp_lattice_a"], config["hcp_lattice_c"], config["ebsd_convention"]), "twin"
    if mode.startswith("BCC {110}"):
        return calcSlipTracesBCC110(euler, "radians"), "slip"
    if mode.startswith("BCC {112}"):
        return calcSlipTracesBCC112(euler, "radians"), "slip"
    raise ValueError(f"No trace function is wired for selected mode: {mode}")


def short_mode_label(mode: str) -> str:
    known = {
        "Basal - 1x plane trace": "basal",
        "Prismatic - 3x plane traces": "prism",
        "Pyramidal I - 6x plane traces": "pyra_i",
        "Pyramidal II - 6x plane traces": "pyra_ii",
        "BCC {110} Slip - 6x plane traces": "bcc_110",
        "BCC {112} Slip - 12x plane traces": "bcc_112",
    }
    return known.get(mode, re.sub(r"[^A-Za-z0-9]+", "_", mode).strip("_").lower())


def normalize_2d_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    out = np.zeros_like(vectors)
    safe = norms > 1e-12
    out[safe] = vectors[safe] / norms[safe, None]
    return out


def angular_mismatch_values(event_direction: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    dots = np.abs(candidates @ event_direction[:2])
    return np.degrees(np.arccos(np.clip(dots, 0.0, 1.0)))


def select_ordered_trace_assignment_index(angle_values: np.ndarray, modes: np.ndarray, tolerances: np.ndarray, mode_priority: list[str]) -> int:
    angles = np.asarray(angle_values, dtype=np.float64)
    passing = angles <= np.asarray(tolerances, dtype=np.float64)
    for mode in mode_priority:
        indices = np.flatnonzero((np.asarray(modes, dtype=object) == mode) & passing)
        if len(indices):
            return int(indices[np.argmin(angles[indices])])
    return int(np.argmin(angles))


def vector_angle_degrees(vector: np.ndarray | None) -> float:
    if vector is None:
        return float("nan")
    angle = np.degrees(np.arctan2(float(vector[1]), float(vector[0])))
    if angle < 0:
        angle += 180.0
    if angle >= 180.0:
        angle -= 180.0
    return float(angle)


def empty_event_score_row(event_id: str, num_pixels: int, reason: str) -> dict:
    return {
        "event_id": event_id,
        "classification": "likely_noise",
        "score": 0.0,
        "best_mode": "unknown",
        "num_pixels": int(num_pixels),
        "reason": reason,
    }
