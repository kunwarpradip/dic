from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure, morphology
from skimage.draw import line as draw_line
from skimage.exposure import equalize_adapthist
from skimage.filters import meijering, threshold_otsu
from skimage.transform import probabilistic_hough_line

from .algorithm import detect_line_from_seed
from .models import DicLine, Point


BOUNDARY_EVENT_COLUMNS = [
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
BOUNDARY_PIXEL_COLUMNS = ["event_id", "original_event_id", "cut_index", "pixel_x", "pixel_y"]


@dataclass(frozen=True)
class AutoPipelineParams:
    image_path: str
    display_min: float = 0.0
    display_max: float = 1.0
    display_brightness: float = 0.0
    use_full_image: bool = False
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 500
    crop_height: int = 500
    clahe_clip_limit: float = 0.300
    ridge_sigma_max: int = 3
    ridge_percentile: float = 80.0
    threshold_multiplier: float = 0.80
    min_object_size: int = 40
    closing_radius: int = 1
    use_skeletonize: bool = True
    hough_threshold: int = 20
    hough_line_length: int = 30
    hough_line_gap: int = 10
    hough_seed_spacing: int = 20
    hough_max_seeds: int = 200
    hough_use_all_seeds: bool = True
    intensity_difference_tolerance: int = 75
    best_fit_line_tolerance: float = 7.0
    min_intensity: int = 120
    min_points_threshold: int = 10
    duplicate_overlap: float = 0.50
    merge_distance_tolerance: float = 5.0
    merge_angle_tolerance: float = 12.0
    connect_merged_event_gaps: bool = False


def image_size(path_str: str) -> tuple[int, int]:
    with Image.open(path_str) as img:
        return img.size


def image_value_summary(path_str: str) -> dict[str, float]:
    with Image.open(path_str) as img:
        arr = np.asarray(img).copy()
    values = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    percentiles = np.percentile(values, [0.5, 99.5])
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "p0_5": float(percentiles[0]),
        "p99_5": float(percentiles[1]),
        "mean": float(values.mean()),
    }


def default_display_range(summary: dict[str, float]) -> tuple[float, float]:
    if summary["min"] <= 0.0 and summary["max"] >= 1.0:
        return 0.0, 1.0
    return summary["min"], summary["max"]


def sanitize_crop(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    width = max(1, min(int(width), image_width))
    height = max(1, min(int(height), image_height))
    x = max(0, min(int(x), image_width - width))
    y = max(0, min(int(y), image_height - height))
    return x, y, width, height


def load_region(params: AutoPipelineParams) -> dict:
    img_w, img_h = image_size(params.image_path)
    if params.use_full_image:
        x0, y0, width, height = 0, 0, img_w, img_h
    else:
        x0, y0, width, height = sanitize_crop(
            params.crop_x,
            params.crop_y,
            params.crop_width,
            params.crop_height,
            img_w,
            img_h,
        )

    with Image.open(params.image_path) as img:
        arr = np.asarray(img.crop((x0, y0, x0 + width, y0 + height))).copy()

    display_rgb = display_crop_rgb(arr, params.display_min, params.display_max, params.display_brightness)
    return {
        "raw": arr,
        "raw_normalized_rgb": normalize_to_uint8_rgb(np.squeeze(arr)),
        "display_rgb": display_rgb,
        "detection_rgb": display_rgb,
        "origin": (x0, y0),
        "image_size": (img_w, img_h),
        "raw_stats": array_stats(arr),
    }


def run_preprocessing(params: AutoPipelineParams) -> dict:
    crop = load_region(params)
    gray = detection_rgb_to_unit_gray(crop["detection_rgb"])
    enhanced = equalize_adapthist(gray, clip_limit=params.clahe_clip_limit).astype(np.float32)
    ridges = meijering(
        enhanced,
        sigmas=tuple(range(1, max(1, params.ridge_sigma_max) + 1)),
        black_ridges=False,
    ).astype(np.float32)

    percentile_threshold = float(np.percentile(ridges, params.ridge_percentile))
    otsu_threshold = float(threshold_otsu(ridges) * params.threshold_multiplier)
    ridge_threshold = max(percentile_threshold, otsu_threshold)
    candidate_mask = ridges > ridge_threshold
    candidate_clean = remove_small_objects_by_label(candidate_mask, params.min_object_size)
    if params.closing_radius > 0:
        candidate_clean = morphology.closing(candidate_clean, morphology.disk(params.closing_radius))

    if params.use_skeletonize:
        skeleton_mask = morphology.skeletonize(candidate_clean)
        detection_mask = skeleton_mask
    else:
        skeleton_mask = None
        detection_mask = candidate_clean

    display_mask = detection_mask
    if params.use_skeletonize:
        display_mask = ndi.binary_dilation(detection_mask, iterations=2)

    grow_gray = (np.clip(enhanced, 0.0, 1.0) * 255).astype(np.uint8)
    return {
        "params": params,
        "crop": crop,
        "gray": gray,
        "enhanced": enhanced,
        "ridges": ridges,
        "percentile_threshold": percentile_threshold,
        "otsu_threshold": otsu_threshold,
        "ridge_threshold": ridge_threshold,
        "candidate_mask": candidate_mask,
        "candidate_clean": candidate_clean,
        "skeleton_mask": skeleton_mask,
        "detection_mask": detection_mask,
        "display_mask": display_mask,
        "grow_rgb": gray_to_rgb(grow_gray),
    }


def run_hough_seed_detection(preprocess: dict, params: AutoPipelineParams) -> dict:
    lines = probabilistic_hough_line(
        preprocess["detection_mask"],
        threshold=params.hough_threshold,
        line_length=params.hough_line_length,
        line_gap=params.hough_line_gap,
    )
    seed_candidates: list[tuple[int, int, float]] = []
    enhanced = preprocess["enhanced"]
    for p0, p1 in lines:
        x0, y0 = p0
        x1, y1 = p1
        rr, cc = draw_line(y0, x0, y1, x1)
        valid = (rr >= 0) & (rr < enhanced.shape[0]) & (cc >= 0) & (cc < enhanced.shape[1])
        rr = rr[valid]
        cc = cc[valid]
        if len(rr) == 0:
            continue
        best_index = int(np.argmax(enhanced[rr, cc]))
        seed_candidates.append((int(rr[best_index]), int(cc[best_index]), float(enhanced[rr[best_index], cc[best_index]])))

    max_seeds = len(seed_candidates) if params.hough_use_all_seeds else params.hough_max_seeds
    seeds = select_spaced_seeds(
        seed_candidates,
        enhanced.shape,
        params.hough_seed_spacing,
        max_seeds,
    )
    return {
        "lines": lines,
        "raw_count": int(len(lines)),
        "candidate_seed_count": int(len(seed_candidates)),
        "max_seeds": int(max_seeds),
        "seeds": seeds,
        "overlay": draw_hough_overlay(preprocess["crop"]["display_rgb"], lines, seeds),
    }


def run_mask_event_detection(preprocess: dict, params: AutoPipelineParams) -> dict:
    labels = measure.label(preprocess["candidate_clean"], connectivity=2)
    props = measure.regionprops(labels)
    accepted: list[DicLine] = []
    for prop in props:
        if prop.area < max(1, int(params.min_object_size)):
            continue
        rows, cols = prop.coords.T
        points = {Point(x=int(col), y=int(row)) for row, col in zip(rows, cols)}
        if not points:
            continue
        accepted.append(
            DicLine(
                is_manual=False,
                points=points,
                seed_point=None,
                intensity_difference_tolerance=params.intensity_difference_tolerance,
                bfl_tolerance=params.best_fit_line_tolerance,
                min_intensity=params.min_intensity,
            )
        )
    return {
        "accepted": accepted,
        "accepted_count": int(len(accepted)),
        "component_count": int(labels.max()),
        "overlay": draw_mask_event_overlay(preprocess["crop"]["display_rgb"], accepted),
    }


def run_boundary_cut_events(
    preprocess: dict,
    event_result: dict,
    boundary_path: str,
    black_threshold: float,
    boundary_dilation_radius: int,
    min_segment_pixels: int,
    connectivity: int,
) -> dict:
    source_events = event_result.get("accepted", [])
    crop_x, crop_y = preprocess["crop"]["origin"]
    crop_h, crop_w = preprocess["crop"]["display_rgb"].shape[:2]
    boundary_mask = load_black_boundary_mask(boundary_path, black_threshold)
    if boundary_dilation_radius > 0:
        cut_mask = morphology.binary_dilation(
            boundary_mask,
            morphology.disk(int(boundary_dilation_radius)),
        )
    else:
        cut_mask = boundary_mask

    if cut_mask.shape[0] < crop_y + crop_h or cut_mask.shape[1] < crop_x + crop_w:
        raise ValueError(
            "Boundary image is smaller than the selected DIC region. "
            f"Boundary mask is {cut_mask.shape[1]} x {cut_mask.shape[0]}, "
            f"selected DIC region ends at x={crop_x + crop_w}, y={crop_y + crop_h}."
        )

    boundary_crop = cut_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    accepted: list[DicLine] = []
    summaries: list[dict] = []
    removed_points: set[Point] = set()
    event_rows: list[dict] = []
    pixel_rows: list[dict] = []
    new_event_index = 0

    for source_index, line in enumerate(source_events, start=1):
        if not line.points:
            continue
        xs = np.array([point.x for point in line.points], dtype=int)
        ys = np.array([point.y for point in line.points], dtype=int)
        valid = (xs >= 0) & (xs < crop_w) & (ys >= 0) & (ys < crop_h)
        xs = xs[valid]
        ys = ys[valid]
        if len(xs) == 0:
            continue

        pad = max(3, int(boundary_dilation_radius) + 3)
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, crop_w)
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, crop_h)

        local_event = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        local_event[ys - y0, xs - x0] = True
        local_cut_mask = boundary_crop[y0:y1, x0:x1]
        boundary_hit_pixels = local_event & local_cut_mask
        cut_event = local_event & ~local_cut_mask

        hit_rows, hit_cols = np.where(boundary_hit_pixels)
        removed_points.update(Point(x=int(x0 + col), y=int(y0 + row)) for row, col in zip(hit_rows, hit_cols))

        labels = measure.label(cut_event, connectivity=int(connectivity))
        props = measure.regionprops(labels)
        kept_segments = [prop for prop in props if prop.area >= int(min_segment_pixels)]
        discarded_small_segments = len(props) - len(kept_segments)

        for cut_index, segment in enumerate(kept_segments, start=1):
            rr, cc = segment.coords.T
            points = {Point(x=int(x0 + col), y=int(y0 + row)) for row, col in zip(rr, cc)}
            if not points:
                continue
            new_event_index += 1
            event_id = f"event_{new_event_index:05d}"
            original_event_id = f"source_{source_index:05d}"
            point_xs = [point.x for point in points]
            point_ys = [point.y for point in points]
            accepted.append(
                DicLine(
                    is_manual=False,
                    points=points,
                    seed_point=line.seed_point,
                    intensity_difference_tolerance=line.intensity_difference_tolerance,
                    bfl_tolerance=line.bfl_tolerance,
                    min_intensity=line.min_intensity,
                )
            )
            event_rows.append(
                {
                    "event_id": event_id,
                    "original_event_id": original_event_id,
                    "cut_index": int(cut_index),
                    "method": "boundary_cut",
                    "num_pixels": int(len(points)),
                    "bbox_min_x": int(crop_x + min(point_xs)),
                    "bbox_max_x": int(crop_x + max(point_xs)),
                    "bbox_min_y": int(crop_y + min(point_ys)),
                    "bbox_max_y": int(crop_y + max(point_ys)),
                    "removed_boundary_pixels_from_original": int(boundary_hit_pixels.sum()),
                    "segments_from_original": int(len(kept_segments)),
                    "discarded_small_segments": int(discarded_small_segments),
                }
            )
            for point in sorted(points):
                pixel_rows.append(
                    {
                        "event_id": event_id,
                        "original_event_id": original_event_id,
                        "cut_index": int(cut_index),
                        "pixel_x": int(crop_x + point.x),
                        "pixel_y": int(crop_y + point.y),
                    }
                )

        summaries.append(
            {
                "source_index": source_index,
                "source_pixels": int(len(xs)),
                "removed_boundary_pixels": int(boundary_hit_pixels.sum()),
                "kept_segments": int(len(kept_segments)),
                "discarded_small_segments": int(discarded_small_segments),
            }
        )

    overlay = draw_boundary_cut_overlay(
        preprocess["crop"]["display_rgb"],
        accepted,
        boundary_crop,
        removed_points,
    )
    return {
        "accepted": accepted,
        "accepted_count": int(len(accepted)),
        "source_count": int(len(source_events)),
        "source_pixel_count": int(sum(line.size for line in source_events)),
        "kept_pixel_count": int(sum(line.size for line in accepted)),
        "removed_pixel_count": int(len(removed_points)),
        "events_touching_boundary": int(sum(1 for item in summaries if item["removed_boundary_pixels"] > 0)),
        "events_split": int(sum(1 for item in summaries if item["kept_segments"] > 1)),
        "discarded_small_segments": int(sum(item["discarded_small_segments"] for item in summaries)),
        "boundary_pixel_count": int(boundary_crop.sum()),
        "overlay": overlay,
        "summary": summaries,
        "events": pd.DataFrame(event_rows, columns=BOUNDARY_EVENT_COLUMNS),
        "event_pixels": pd.DataFrame(pixel_rows, columns=BOUNDARY_PIXEL_COLUMNS),
    }


def load_black_boundary_mask(boundary_path: str, black_threshold: float) -> np.ndarray:
    with Image.open(boundary_path) as img:
        arr = np.asarray(img).copy()
    if arr.ndim == 3:
        gray = arr[..., :3].astype(np.float32).mean(axis=2) / 255.0
    else:
        gray = arr.astype(np.float32)
        if gray.max() > 1:
            gray = gray / 255.0
    return gray < float(black_threshold)


def trace_hough_events(preprocess: dict, hough: dict, params: AutoPipelineParams, progress_callback=None) -> dict:
    accepted_entries: list[dict] = []
    rejected = 0
    skipped_covered = 0
    grow_rgb = preprocess["grow_rgb"]
    h, w = grow_rgb.shape[:2]
    accepted_pixel_mask = np.zeros((h, w), dtype=bool)

    seeds = hough["seeds"]
    total_seeds = len(seeds)
    for seed_index, seed in enumerate(seeds, start=1):
        if 0 <= seed.y < h and 0 <= seed.x < w and accepted_pixel_mask[seed.y, seed.x]:
            skipped_covered += 1
            if progress_callback is not None:
                progress_callback(seed_index, total_seeds)
            continue

        result = detect_line_from_seed(
            seed.x,
            seed.y,
            grow_rgb,
            intensity_difference_tolerance=params.intensity_difference_tolerance,
            bfl_tolerance=params.best_fit_line_tolerance,
            min_intensity=params.min_intensity,
            min_points_threshold=params.min_points_threshold,
        )
        if result.line is None:
            rejected += 1
            if progress_callback is not None:
                progress_callback(seed_index, total_seeds)
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
            mark_line_pixels_visited(accepted_pixel_mask, merged_line)
        else:
            accepted_entries.append({"line": result.line, "seeds": {seed}, "was_merged": False})
            mark_line_pixels_visited(accepted_pixel_mask, result.line)
        if progress_callback is not None:
            progress_callback(seed_index, total_seeds)

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
        {"line": entry["line"], "seeds": sorted(entry["seeds"])}
        for entry in accepted_entries
        if entry["was_merged"]
    ]
    return {
        "accepted": accepted,
        "accepted_count": int(len(accepted)),
        "rejected": int(rejected),
        "skipped_covered": int(skipped_covered),
        "merged": int(len(merged_groups)),
        "standalone_seeds": standalone_seeds,
        "merged_seeds": merged_seeds,
        "merged_groups": merged_groups,
        "overlay": draw_event_overlay(
            preprocess["crop"]["display_rgb"],
            accepted,
            standalone_seeds,
            merged_seeds,
            merged_groups,
        ),
    }


def mark_line_pixels_visited(visited: np.ndarray, line: DicLine) -> None:
    h, w = visited.shape[:2]
    for point in line.points:
        if 0 <= point.x < w and 0 <= point.y < h:
            visited[point.y, point.x] = True


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
    candidates = [(row, col, score) for (row, col), score in strongest_by_pixel.items()]
    candidates.sort(key=lambda item: item[2], reverse=True)

    blocked = np.zeros(shape, dtype=bool)
    accepted: list[Point] = []
    radius = max(1, int(seed_spacing))
    h, w = shape
    for row, col, _score in candidates:
        if row < 0 or row >= h or col < 0 or col >= w or blocked[row, col]:
            continue
        accepted.append(Point(x=int(col), y=int(row)))
        if len(accepted) >= max_seeds:
            break
        y0 = max(0, row - radius)
        y1 = min(h, row + radius + 1)
        x0 = max(0, col - radius)
        x1 = min(w, col + radius + 1)
        blocked[y0:y1, x0:x1] = True
    return accepted


def should_merge_events(candidate: DicLine, existing: DicLine, params: AutoPipelineParams) -> bool:
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
    rows, cols = np.where(connected)
    return {Point(x=int(origin_x + col), y=int(origin_y + row)) for row, col in zip(rows, cols)}


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
        rr, cc = draw_line(int(point_a[0]), int(point_a[1]), int(point_b[0]), int(point_b[1]))
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
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
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


def line_overlap_fraction(line_a: DicLine, line_b: DicLine) -> float:
    if not line_a.points or not line_b.points:
        return 0.0
    overlap = len(line_a.points & line_b.points)
    return overlap / min(len(line_a.points), len(line_b.points))


def draw_hough_overlay(rgb: np.ndarray, lines: list, seeds: list[Point]) -> np.ndarray:
    out = rgb.astype(np.float32) / 255.0
    line_mask = np.zeros(out.shape[:2], dtype=bool)
    for p0, p1 in lines:
        x0, y0 = p0
        x1, y1 = p1
        rr, cc = draw_line(y0, x0, y1, x1)
        valid = (rr >= 0) & (rr < out.shape[0]) & (cc >= 0) & (cc < out.shape[1])
        line_mask[rr[valid], cc[valid]] = True
    line_mask = ndi.binary_dilation(line_mask, iterations=2)
    out[line_mask] = [1.0, 0.0, 0.0]
    return (draw_points(out, seeds, color=(1.0, 1.0, 0.0), radius=4) * 255).astype(np.uint8)


def draw_event_overlay(
    rgb: np.ndarray,
    lines: list[DicLine],
    standalone_seeds: list[Point],
    merged_seeds: list[Point],
    merged_groups: list[dict] | None = None,
) -> np.ndarray:
    out = rgb.astype(np.float32) / 255.0
    mask = np.zeros(out.shape[:2], dtype=bool)
    for line in lines:
        for point in line.points:
            if 0 <= point.x < mask.shape[1] and 0 <= point.y < mask.shape[0]:
                mask[point.y, point.x] = True
    merged_mask = np.zeros(out.shape[:2], dtype=bool)
    for group in merged_groups or []:
        for point in group["line"].points:
            if 0 <= point.x < merged_mask.shape[1] and 0 <= point.y < merged_mask.shape[0]:
                merged_mask[point.y, point.x] = True
    if merged_mask.any():
        halo = ndi.binary_dilation(merged_mask, iterations=5)
        inner = ndi.binary_dilation(merged_mask, iterations=2)
        out[halo & ~inner] = [0.0, 0.25, 1.0]
    out[ndi.binary_dilation(mask, iterations=2)] = [1.0, 0.0, 0.0]
    out = draw_points(out, standalone_seeds, color=(1.0, 1.0, 0.0), radius=4, outline_color=(1.0, 1.0, 1.0))
    out = draw_points(out, merged_seeds, color=(0.0, 0.25, 1.0), radius=4, outline_color=(1.0, 1.0, 1.0))
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def draw_mask_event_overlay(rgb: np.ndarray, lines: list[DicLine]) -> np.ndarray:
    out = rgb.astype(np.float32) / 255.0
    mask = np.zeros(out.shape[:2], dtype=bool)
    for line in lines:
        for point in line.points:
            if 0 <= point.x < mask.shape[1] and 0 <= point.y < mask.shape[0]:
                mask[point.y, point.x] = True
    out[mask] = [1.0, 0.0, 0.0]
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def draw_boundary_cut_overlay(
    rgb: np.ndarray,
    cut_lines: list[DicLine],
    boundary_crop: np.ndarray,
    removed_points: set[Point],
) -> np.ndarray:
    out = rgb.astype(np.float32) / 255.0
    kept_mask = np.zeros(out.shape[:2], dtype=bool)
    removed_mask = np.zeros(out.shape[:2], dtype=bool)
    for line in cut_lines:
        for point in line.points:
            if 0 <= point.x < kept_mask.shape[1] and 0 <= point.y < kept_mask.shape[0]:
                kept_mask[point.y, point.x] = True
    for point in removed_points:
        if 0 <= point.x < removed_mask.shape[1] and 0 <= point.y < removed_mask.shape[0]:
            removed_mask[point.y, point.x] = True
    boundary_display = boundary_crop.astype(bool, copy=False)
    if boundary_display.shape == kept_mask.shape:
        out[boundary_display] = [0.0, 0.31, 1.0]
    out[kept_mask] = [1.0, 0.0, 0.0]
    out[removed_mask] = [0.0, 0.86, 0.31]
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


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
            out[
                max(0, seed.y - outline_radius): min(h, seed.y + outline_radius + 1),
                max(0, seed.x - outline_radius): min(w, seed.x + outline_radius + 1),
            ] = outline_color
        out[
            max(0, seed.y - radius): min(h, seed.y + radius + 1),
            max(0, seed.x - radius): min(w, seed.x + radius + 1),
        ] = color
    return out


def detection_rgb_to_unit_gray(detection_rgb: np.ndarray) -> np.ndarray:
    gray = np.nan_to_num(
        detection_rgb[..., 2].astype(np.float32, copy=False),
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )
    return np.clip(gray / 255.0, 0.0, 1.0)


def display_crop_rgb(arr: np.ndarray, display_min: float, display_max: float, brightness_adjust: float = 0.0) -> np.ndarray:
    squeezed = np.squeeze(arr)
    if squeezed.ndim == 3 and squeezed.shape[2] >= 3:
        return apply_output_brightness(to_uint8_rgb(squeezed), brightness_adjust)
    return scale_to_uint8_rgb(squeezed, display_min, display_max, brightness_adjust)


def scale_to_uint8_rgb(arr: np.ndarray, display_min: float, display_max: float, brightness_adjust: float = 0.0) -> np.ndarray:
    values = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if display_max <= display_min:
        scaled = np.zeros(values.shape, dtype=np.uint8)
    else:
        scaled = (np.clip((values - display_min) / (display_max - display_min), 0.0, 1.0) * 255).astype(np.uint8)
    scaled = apply_output_brightness(scaled, brightness_adjust)
    if scaled.ndim == 2:
        return gray_to_rgb(scaled)
    if scaled.ndim == 3 and scaled.shape[2] >= 3:
        return scaled[..., :3]
    return gray_to_rgb(np.squeeze(scaled))


def apply_output_brightness(values: np.ndarray, brightness_adjust: float = 0.0) -> np.ndarray:
    if abs(float(brightness_adjust)) < 1e-9:
        return values
    shifted = values.astype(np.float32, copy=False) + float(brightness_adjust) / 100.0 * 127.5
    return np.clip(shifted, 0.0, 255.0).astype(np.uint8)


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
    return gray_to_rgb(np.squeeze(normalized))


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    rgb = arr[..., :3]
    if rgb.dtype == np.uint8:
        return rgb.astype(np.uint8, copy=False)
    return normalize_to_uint8_rgb(rgb)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=2)


def array_stats(arr: np.ndarray) -> dict[str, float]:
    values = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def remove_small_objects_by_label(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels = measure.label(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= max(1, int(min_size))
    if keep.size:
        keep[0] = False
    return keep[labels]


def display_image(image: np.ndarray, mode: str = "gray") -> np.ndarray:
    if image.dtype == bool:
        return gray_to_rgb((image.astype(np.uint8) * 255))
    if image.ndim == 2:
        arr = normalize_for_display(image)
        if mode == "magma":
            import matplotlib.pyplot as plt

            return (plt.get_cmap("magma")(arr)[..., :3] * 255).astype(np.uint8)
        return gray_to_rgb((arr * 255).astype(np.uint8))
    if image.dtype == np.uint8:
        return image[..., :3]
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)[..., :3]


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(image.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(values, [1.0, 99.5])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
