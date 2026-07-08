from __future__ import annotations

import random
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from .models import DicLine, Point


NEIGHBORS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


@dataclass(frozen=True)
class LineDetectionResult:
    line: DicLine | None
    point_count: int
    seed_blue_intensity: int | None
    rejection_reason: str | None = None


def get_best_fit_mask(
    seed_x: int,
    seed_y: int,
    image_rgb: np.ndarray,
    intensity_difference_tolerance: int,
    bfl_tolerance: float,
    min_intensity: int,
) -> set[Point]:
    """Faithful port of Java DicAlgorithm.getBestFitMask.

    Important Java behaviors preserved:
    - stack/DFS traversal
    - seed is accepted before intensity checks
    - blue channel intensity
    - randomized 8-neighbor expansion
    - vertical regression error, not perpendicular distance
    - neighbor accepted if intensity >= current - tolerance and >= minIntensity
    """

    h, w, _ = image_rgb.shape
    stack = [Point(int(seed_x), int(seed_y))]
    mask_points: set[Point] = set()
    blue = image_rgb[..., 2]

    while stack:
        current = stack.pop()
        if current in mask_points:
            continue
        if current.x < 0 or current.x >= w or current.y < 0 or current.y >= h:
            continue

        mask_points.add(current)
        regression = _simple_regression(mask_points) if len(mask_points) > 1 else None

        for new_point in _expanded_points(current):
            if new_point.x < 0 or new_point.x >= w or new_point.y < 0 or new_point.y >= h:
                continue

            intensity_threshold = int(blue[current.y, current.x])
            new_intensity = int(blue[new_point.y, new_point.x])
            bfl_error = 0.0
            if regression is not None:
                slope, intercept = regression
                predicted_y = slope * new_point.x + intercept
                bfl_error = abs(predicted_y - new_point.y)

            if not (bfl_error < bfl_tolerance):
                continue
            if new_intensity < intensity_threshold - intensity_difference_tolerance:
                continue
            if new_intensity < min_intensity:
                continue
            stack.append(new_point)

    return mask_points


def create_line_from_seed(
    seed_x: int,
    seed_y: int,
    image_rgb: np.ndarray,
    intensity_difference_tolerance: int = 75,
    bfl_tolerance: float = 7.0,
    min_intensity: int = 120,
    min_points_threshold: int = 10,
) -> DicLine | None:
    return detect_line_from_seed(
        seed_x,
        seed_y,
        image_rgb,
        intensity_difference_tolerance=intensity_difference_tolerance,
        bfl_tolerance=bfl_tolerance,
        min_intensity=min_intensity,
        min_points_threshold=min_points_threshold,
    ).line


def detect_line_from_seed(
    seed_x: int,
    seed_y: int,
    image_rgb: np.ndarray,
    intensity_difference_tolerance: int = 75,
    bfl_tolerance: float = 7.0,
    min_intensity: int = 120,
    min_points_threshold: int = 10,
) -> LineDetectionResult:
    h, w, _ = image_rgb.shape
    if seed_x < 0 or seed_x >= w or seed_y < 0 or seed_y >= h:
        return LineDetectionResult(
            line=None,
            point_count=0,
            seed_blue_intensity=None,
            rejection_reason="Seed is outside the image bounds",
        )

    seed_blue_intensity = int(image_rgb[seed_y, seed_x, 2])
    points = get_best_fit_mask(
        seed_x,
        seed_y,
        image_rgb,
        intensity_difference_tolerance,
        bfl_tolerance,
        min_intensity,
    )
    if len(points) < min_points_threshold:
        if seed_blue_intensity < min_intensity:
            reason = (
                f"Seed blue intensity {seed_blue_intensity} is below "
                f"minimum intensity {min_intensity}"
            )
        else:
            reason = (
                f"Seed only grew {len(points)} points; at least "
                f"{min_points_threshold} are required"
            )
        return LineDetectionResult(
            line=None,
            point_count=len(points),
            seed_blue_intensity=seed_blue_intensity,
            rejection_reason=reason,
        )
    line = DicLine(
        id=uuid4(),
        is_manual=False,
        points=points,
        seed_point=Point(int(seed_x), int(seed_y)),
        intensity_difference_tolerance=intensity_difference_tolerance,
        bfl_tolerance=bfl_tolerance,
        min_intensity=min_intensity,
    )
    return LineDetectionResult(
        line=line,
        point_count=len(points),
        seed_blue_intensity=seed_blue_intensity,
    )


def merge_lines(lines: list[DicLine], selected_ids: set) -> tuple[list[DicLine], DicLine | None]:
    if len(selected_ids) < 2:
        return lines, None
    new_points: set[Point] = set()
    new_lines = [line for line in lines if line.id not in selected_ids]
    intensity_difference_tolerance = 0
    bfl_tolerance = 0.0
    min_intensity = 0
    for line in lines:
        if line.id not in selected_ids:
            continue
        intensity_difference_tolerance = line.intensity_difference_tolerance
        bfl_tolerance = line.bfl_tolerance
        min_intensity = line.min_intensity
        new_points.update(line.points)
    merged = DicLine(
        id=uuid4(),
        is_manual=True,
        points=new_points,
        seed_point=None,
        intensity_difference_tolerance=intensity_difference_tolerance,
        bfl_tolerance=bfl_tolerance,
        min_intensity=min_intensity,
    )
    new_lines.append(merged)
    return new_lines, merged


def get_cut_line_points(start: Point, end: Point, cut_line_width: int) -> set[Point]:
    slope = 0.0
    has_slope = False
    if abs(end.x - start.x) > 1e-8:
        has_slope = True
        slope = (end.y - start.y) / (end.x - start.x)

    intercept = start.x
    if has_slope:
        intercept = start.y - slope * start.x

    points: set[Point] = set()
    start_x = min(start.x, end.x)
    end_x = max(start.x, end.x)
    half = cut_line_width // 2
    for x in range(start_x, end_x + 1):
        y = int(slope * x + intercept)
        for i in range(x - half, x + half + 1):
            for j in range(y - half, y + half + 1):
                points.add(Point(i, j))
    return points


def cut_selected_lines(
    lines: list[DicLine], selected_ids: set, start: Point, end: Point, cut_line_width: int
) -> tuple[list[DicLine], list[DicLine]]:
    cut_points = get_cut_line_points(start, end, cut_line_width)
    candidates = [line for line in lines if line.id in selected_ids]
    line_to_cut = next((line for line in candidates if line.points & cut_points), None)
    if line_to_cut is None:
        return lines, []

    exclusion_points = line_to_cut.points & cut_points
    components = _connected_components(line_to_cut.points - exclusion_points)
    new_lines = [line for line in lines if line.id != line_to_cut.id]
    daughters: list[DicLine] = []
    for component in components:
        if len(component) <= 1:
            continue
        daughter = DicLine(
            id=uuid4(),
            is_manual=True,
            points=component,
            seed_point=None,
            intensity_difference_tolerance=line_to_cut.intensity_difference_tolerance,
            bfl_tolerance=line_to_cut.bfl_tolerance,
            min_intensity=line_to_cut.min_intensity,
        )
        daughters.append(daughter)
        new_lines.append(daughter)

    if daughters and exclusion_points:
        smallest = min(daughters, key=lambda line: len(line.points))
        smallest.points.update(exclusion_points)

    return new_lines, daughters


def is_point_in_line(point: Point, line: DicLine, tolerance: float = 3.0) -> bool:
    tt = tolerance * tolerance
    for p in line.points:
        if (point.x - p.x) ** 2 + (point.y - p.y) ** 2 < tt:
            return True
    return False


def line_in_visible_area(line: DicLine, x0: float, y0: float, x1: float, y1: float) -> bool:
    return any(x0 <= p.x <= x1 and y0 <= p.y <= y1 for p in line.points)


def convex_hull(points: set[Point]) -> list[Point]:
    pts = sorted(points)
    if len(pts) <= 1:
        return pts

    def cross(o: Point, a: Point, b: Point) -> int:
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _expanded_points(point: Point) -> list[Point]:
    neighbors = [Point(point.x + dx, point.y + dy) for dx, dy in NEIGHBORS]
    random.shuffle(neighbors)
    return neighbors


def _simple_regression(points: set[Point]) -> tuple[float, float] | None:
    n = len(points)
    sx = sum(p.x for p in points)
    sy = sum(p.y for p in points)
    sxx = sum(p.x * p.x for p in points)
    sxy = sum(p.x * p.y for p in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _connected_components(points: set[Point]) -> list[set[Point]]:
    remaining = set(points)
    components: list[set[Point]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for dx, dy in NEIGHBORS:
                neighbor = Point(current.x + dx, current.y + dy)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components
