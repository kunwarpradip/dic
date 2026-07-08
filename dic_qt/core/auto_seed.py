from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Point


@dataclass(frozen=True)
class AutoSeedSettings:
    min_distance: int = 20
    max_seeds: int = 300
    ridge_percentile: float = 97.5
    threshold_multiplier: float = 0.85
    clahe_clip_limit: float = 0.02
    min_object_size: int = 30
    closing_radius: int = 2
    ridge_sigmas: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)


@dataclass(frozen=True)
class AutoSeedResult:
    seeds: list[Point]
    detection_image_rgb: np.ndarray
    candidate_pixel_count: int
    skeleton_pixel_count: int
    ridge_threshold: float


def generate_auto_seed_result(
    image_rgb: np.ndarray,
    settings: AutoSeedSettings,
) -> AutoSeedResult:
    """Generate DIC-only seed pixels for the existing seed-growing algorithm."""

    from scipy import ndimage as ndi
    from skimage import measure, morphology
    from skimage.exposure import equalize_adapthist
    from skimage.feature import peak_local_max
    from skimage.filters import meijering, threshold_otsu

    gray = _robust_normalize(image_rgb[..., 2].astype(np.float32, copy=False))
    enhanced = equalize_adapthist(
        gray,
        clip_limit=settings.clahe_clip_limit,
    ).astype(np.float32, copy=False)

    ridges = meijering(
        enhanced,
        sigmas=settings.ridge_sigmas,
        black_ridges=False,
    ).astype(np.float32, copy=False)

    ridge_threshold = _ridge_threshold(
        ridges,
        settings.ridge_percentile,
        settings.threshold_multiplier,
    )
    candidate_mask = ridges > ridge_threshold
    candidate_mask = _remove_small_objects(measure, candidate_mask, settings.min_object_size)
    if settings.closing_radius > 0:
        candidate_mask = morphology.closing(
            candidate_mask,
            morphology.disk(settings.closing_radius),
        )

    skeleton = morphology.skeletonize(candidate_mask)
    seeds = _select_seed_points(
        ridges,
        skeleton,
        min_distance=settings.min_distance,
        max_seeds=settings.max_seeds,
        threshold=ridge_threshold,
        peak_local_max=peak_local_max,
        measure=measure,
        ndi=ndi,
    )

    detection_gray = (np.clip(enhanced, 0.0, 1.0) * 255).astype(np.uint8)
    detection_image_rgb = np.repeat(detection_gray[..., None], 3, axis=2)
    return AutoSeedResult(
        seeds=seeds,
        detection_image_rgb=detection_image_rgb,
        candidate_pixel_count=int(candidate_mask.sum()),
        skeleton_pixel_count=int(skeleton.sum()),
        ridge_threshold=float(ridge_threshold),
    )


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(values, [1.0, 99.7])
    if hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _ridge_threshold(
    ridges: np.ndarray,
    ridge_percentile: float,
    threshold_multiplier: float,
) -> float:
    from skimage.filters import threshold_otsu

    percentile_threshold = float(np.percentile(ridges, ridge_percentile))
    if np.allclose(ridges.min(), ridges.max()):
        return percentile_threshold
    otsu_threshold = float(threshold_otsu(ridges) * threshold_multiplier)
    return max(percentile_threshold, otsu_threshold)


def _remove_small_objects(measure, mask: np.ndarray, min_size: int) -> np.ndarray:
    labels = measure.label(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    if keep.size:
        keep[0] = False
    return keep[labels]


def _select_seed_points(
    ridges: np.ndarray,
    skeleton: np.ndarray,
    min_distance: int,
    max_seeds: int,
    threshold: float,
    peak_local_max,
    measure,
    ndi,
) -> list[Point]:
    if max_seeds <= 0 or not skeleton.any():
        return []

    min_distance = max(1, int(min_distance))
    max_seeds = max(1, int(max_seeds))
    peak_coords = peak_local_max(
        ridges,
        labels=skeleton.astype(np.uint8),
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=False,
        num_peaks=max_seeds * 4,
    )

    coords: list[tuple[int, int]] = [(int(row), int(col)) for row, col in peak_coords]

    # Ensure each skeleton component has at least one candidate peak.
    labels = measure.label(skeleton)
    objects = ndi.find_objects(labels)
    for label_index, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        component = labels[slc] == label_index
        if not component.any():
            continue
        local_ridges = np.where(component, ridges[slc], -np.inf)
        local_row, local_col = np.unravel_index(np.argmax(local_ridges), local_ridges.shape)
        row = int(local_row + slc[0].start)
        col = int(local_col + slc[1].start)
        coords.append((row, col))

    unique_coords = list(dict.fromkeys(coords))
    unique_coords.sort(key=lambda rc: float(ridges[rc[0], rc[1]]), reverse=True)
    accepted = _greedy_spaced_points(unique_coords, ridges.shape, min_distance, max_seeds)
    return [Point(x=col, y=row) for row, col in accepted]


def _greedy_spaced_points(
    coords: list[tuple[int, int]],
    shape: tuple[int, int],
    min_distance: int,
    max_seeds: int,
) -> list[tuple[int, int]]:
    blocked = np.zeros(shape, dtype=bool)
    accepted: list[tuple[int, int]] = []
    h, w = shape
    radius = max(1, int(min_distance))

    for row, col in coords:
        if row < 0 or row >= h or col < 0 or col >= w:
            continue
        if blocked[row, col]:
            continue
        accepted.append((row, col))
        if len(accepted) >= max_seeds:
            break

        row0 = max(0, row - radius)
        row1 = min(h, row + radius + 1)
        col0 = max(0, col - radius)
        col1 = min(w, col + radius + 1)
        blocked[row0:row1, col0:col1] = True

    return accepted
