from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import AffineTransform, PolynomialTransform, warp

from .repository import app_data_dir_for_image


@dataclass(frozen=True)
class AlignmentPoint:
    id: int
    x: float
    y: float


@dataclass(frozen=True)
class AlignmentResult:
    aligned_rgb: np.ndarray
    point_ids: list[int]
    metadata: dict[str, object]


POLYNOMIAL_ORDER = 2


def points_json_path(image_path: str | Path) -> Path:
    image_path = Path(image_path)
    return app_data_dir_for_image(image_path) / f"{image_path.stem}.json"


def load_alignment_points(image_path: str | Path) -> list[AlignmentPoint]:
    path = points_json_path(image_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    points: list[AlignmentPoint] = []
    for row in data:
        points.append(
            AlignmentPoint(
                id=int(row["id"]),
                x=float(row["x"]),
                y=float(row["y"]),
            )
        )
    return points


def save_alignment_points(image_path: str | Path, points: list[AlignmentPoint]) -> Path:
    path = points_json_path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(point) for point in sorted(points, key=lambda p: p.id)], f, indent=2)
    return path


def paired_points(
    dic_points: list[AlignmentPoint],
    ebsd_points: list[AlignmentPoint],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    dic_by_id = {point.id: point for point in dic_points}
    ebsd_by_id = {point.id: point for point in ebsd_points}
    ids = sorted(set(dic_by_id) & set(ebsd_by_id))
    dic_xy = np.array([[dic_by_id[point_id].x, dic_by_id[point_id].y] for point_id in ids])
    ebsd_xy = np.array([[ebsd_by_id[point_id].x, ebsd_by_id[point_id].y] for point_id in ids])
    return dic_xy, ebsd_xy, ids


def align_ebsd_to_dic(
    ebsd_rgb: np.ndarray,
    dic_rgb: np.ndarray,
    ebsd_points: list[AlignmentPoint],
    dic_points: list[AlignmentPoint],
) -> AlignmentResult:
    dic_xy, ebsd_xy, point_ids = paired_points(dic_points, ebsd_points)
    if len(point_ids) < 6:
        raise ValueError("At least 6 matched DIC/EBSD control-point pairs are required")

    inverse_transform = _estimate_polynomial_transform(dic_xy, ebsd_xy, POLYNOMIAL_ORDER)

    out_shape = dic_rgb.shape[:2]
    aligned = warp(
        ebsd_rgb,
        inverse_map=inverse_transform,
        output_shape=out_shape,
        order=1,
        mode="constant",
        cval=0,
        preserve_range=True,
    )
    metadata = alignment_transformation_metadata(
        ebsd_rgb,
        dic_rgb,
        ebsd_points,
        dic_points,
    )
    return AlignmentResult(
        aligned_rgb=np.clip(aligned, 0, 255).astype(np.uint8),
        point_ids=point_ids,
        metadata=metadata,
    )


def alignment_transformation_metadata(
    ebsd_rgb: np.ndarray,
    dic_rgb: np.ndarray,
    ebsd_points: list[AlignmentPoint],
    dic_points: list[AlignmentPoint],
) -> dict[str, object]:
    dic_xy, ebsd_xy, point_ids = paired_points(dic_points, ebsd_points)
    if len(point_ids) < 6:
        raise ValueError("At least 6 matched DIC/EBSD control-point pairs are required")

    inverse_transform = _estimate_polynomial_transform(dic_xy, ebsd_xy, POLYNOMIAL_ORDER)
    forward_transform = _estimate_polynomial_transform(ebsd_xy, dic_xy, POLYNOMIAL_ORDER)
    return _alignment_metadata(
        dic_xy=dic_xy,
        ebsd_xy=ebsd_xy,
        point_ids=point_ids,
        dic_shape=dic_rgb.shape,
        ebsd_shape=ebsd_rgb.shape,
        forward_transform=forward_transform,
        inverse_transform=inverse_transform,
    )


def _estimate_polynomial_transform(
    source_xy: np.ndarray,
    destination_xy: np.ndarray,
    order: int,
) -> PolynomialTransform:
    if hasattr(PolynomialTransform, "from_estimate"):
        transform = PolynomialTransform.from_estimate(source_xy, destination_xy, order=order)
    else:
        transform = PolynomialTransform()
        if not transform.estimate(source_xy, destination_xy, order=order):
            raise ValueError("Could not estimate quadratic alignment transform")
    return transform


def _alignment_metadata(
    dic_xy: np.ndarray,
    ebsd_xy: np.ndarray,
    point_ids: list[int],
    dic_shape: tuple[int, ...],
    ebsd_shape: tuple[int, ...],
    forward_transform: PolynomialTransform,
    inverse_transform: PolynomialTransform,
) -> dict[str, object]:
    predicted_dic_xy = forward_transform(ebsd_xy)
    residual_vectors = predicted_dic_xy - dic_xy
    residual_distances = np.linalg.norm(residual_vectors, axis=1)

    predicted_ebsd_xy = inverse_transform(dic_xy)
    inverse_residual_vectors = predicted_ebsd_xy - ebsd_xy
    inverse_residual_distances = np.linalg.norm(inverse_residual_vectors, axis=1)

    affine_summary = _affine_summary(ebsd_xy, dic_xy)
    forward_terms = _polynomial_coefficients(forward_transform.params, POLYNOMIAL_ORDER)
    inverse_terms = _polynomial_coefficients(inverse_transform.params, POLYNOMIAL_ORDER)

    return {
        "schema_version": 1,
        "transform_family": "polynomial",
        "polynomial_order": POLYNOMIAL_ORDER,
        "coordinate_order": "x_y",
        "matched_point_ids": point_ids,
        "dic_image_shape": list(dic_shape),
        "ebsd_image_shape": list(ebsd_shape),
        "dic_points_xy": dic_xy.tolist(),
        "ebsd_points_xy": ebsd_xy.tolist(),
        "forward_transform_ebsd_to_dic": {
            "description": "Polynomial transform that maps EBSD pixel coordinates into DIC pixel coordinates.",
            "params": forward_transform.params.tolist(),
        },
        "forward_polynomial_ebsd_to_dic": {
            "input_variables": ["x_ebsd", "y_ebsd"],
            "output_variables": ["x_dic", "y_dic"],
            "terms": _polynomial_term_names(POLYNOMIAL_ORDER),
            "x_coefficients": forward_terms["x_coefficients"],
            "y_coefficients": forward_terms["y_coefficients"],
            "x_equation": _polynomial_equation("x_dic", "x_ebsd", "y_ebsd", forward_terms["x_coefficients"]),
            "y_equation": _polynomial_equation("y_dic", "x_ebsd", "y_ebsd", forward_terms["y_coefficients"]),
        },
        "inverse_map_dic_to_ebsd": {
            "description": "Polynomial transform used by skimage.warp as inverse_map: output DIC coordinates to input EBSD coordinates.",
            "params": inverse_transform.params.tolist(),
        },
        "inverse_polynomial_dic_to_ebsd": {
            "input_variables": ["x_dic", "y_dic"],
            "output_variables": ["x_ebsd", "y_ebsd"],
            "terms": _polynomial_term_names(POLYNOMIAL_ORDER),
            "x_coefficients": inverse_terms["x_coefficients"],
            "y_coefficients": inverse_terms["y_coefficients"],
            "x_equation": _polynomial_equation("x_ebsd", "x_dic", "y_dic", inverse_terms["x_coefficients"]),
            "y_equation": _polynomial_equation("y_ebsd", "x_dic", "y_dic", inverse_terms["y_coefficients"]),
        },
        "residuals_forward_ebsd_to_dic": _residual_rows(
            point_ids,
            residual_vectors,
            residual_distances,
        ),
        "rmse_forward_pixels": float(np.sqrt(np.mean(residual_distances**2))),
        "max_error_forward_pixels": float(np.max(residual_distances)),
        "residuals_inverse_dic_to_ebsd": _residual_rows(
            point_ids,
            inverse_residual_vectors,
            inverse_residual_distances,
        ),
        "rmse_inverse_pixels": float(np.sqrt(np.mean(inverse_residual_distances**2))),
        "max_error_inverse_pixels": float(np.max(inverse_residual_distances)),
        "affine_approximation_ebsd_to_dic": affine_summary,
    }


def _polynomial_term_names(order: int) -> list[str]:
    terms: list[str] = []
    for degree in range(order + 1):
        for y_power in range(degree + 1):
            x_power = degree - y_power
            if x_power == 0 and y_power == 0:
                terms.append("1")
            elif y_power == 0:
                terms.append("x" if x_power == 1 else f"x^{x_power}")
            elif x_power == 0:
                terms.append("y" if y_power == 1 else f"y^{y_power}")
            else:
                x_term = "x" if x_power == 1 else f"x^{x_power}"
                y_term = "y" if y_power == 1 else f"y^{y_power}"
                terms.append(f"{x_term}*{y_term}")
    return terms


def _polynomial_coefficients(params: np.ndarray, order: int) -> dict[str, list[dict[str, float | str]]]:
    terms = _polynomial_term_names(order)
    return {
        "x_coefficients": [
            {"term": term, "coefficient": float(coef)}
            for term, coef in zip(terms, params[0])
        ],
        "y_coefficients": [
            {"term": term, "coefficient": float(coef)}
            for term, coef in zip(terms, params[1])
        ],
    }


def _polynomial_equation(
    output_name: str,
    input_x_name: str,
    input_y_name: str,
    coefficients: list[dict[str, float | str]],
) -> str:
    pieces: list[str] = []
    for item in coefficients:
        term = str(item["term"]).replace("x", input_x_name).replace("y", input_y_name)
        coef = float(item["coefficient"])
        pieces.append(f"({coef:.17g})*{term}")
    return f"{output_name} = " + " + ".join(pieces)


def _residual_rows(
    point_ids: list[int],
    vectors_xy: np.ndarray,
    distances: np.ndarray,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for point_id, vector, distance in zip(point_ids, vectors_xy, distances):
        rows.append(
            {
                "id": int(point_id),
                "dx": float(vector[0]),
                "dy": float(vector[1]),
                "error_pixels": float(distance),
            }
        )
    return rows


def _affine_summary(ebsd_xy: np.ndarray, dic_xy: np.ndarray) -> dict[str, object]:
    if hasattr(AffineTransform, "from_estimate"):
        affine = AffineTransform.from_estimate(ebsd_xy, dic_xy)
    else:
        affine = AffineTransform()
        if not affine.estimate(ebsd_xy, dic_xy):
            return {"available": False}
    matrix = affine.params
    try:
        inverse_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        inverse_matrix = None
    return {
        "available": True,
        "description": "Best-fit affine approximation for summary only; the saved alignment image uses the polynomial transform.",
        "matrix_3x3": matrix.tolist(),
        "inverse_matrix_3x3": inverse_matrix.tolist() if inverse_matrix is not None else None,
        "translation_xy": [float(v) for v in affine.translation],
        "rotation_degrees": float(np.degrees(affine.rotation)),
        "scale_xy": [float(v) for v in affine.scale],
        "shear_degrees": float(np.degrees(affine.shear)),
    }


def save_aligned_image(output_path: str | Path, image_rgb: np.ndarray) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(output_path)
    return output_path


def save_alignment_metadata(output_path: str | Path, metadata: dict[str, object]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return output_path
