from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LoadedImage:
    raw: np.ndarray
    display_rgb: np.ndarray
    detection_rgb: np.ndarray
    is_scalar: bool
    display_note: str


def load_rgb_uint8(path: str | Path) -> np.ndarray:
    return load_image_data(path).display_rgb


def load_image_data(path: str | Path) -> LoadedImage:
    with Image.open(path) as img:
        arr = np.asarray(img).copy()
        if _is_rgb_like(arr):
            rgb = _rgb_like_to_uint8(arr)
            return LoadedImage(
                raw=arr,
                display_rgb=rgb,
                detection_rgb=rgb,
                is_scalar=False,
                display_note="RGB image displayed without scalar contrast enhancement",
            )

    display_gray = _scalar_to_display_uint8(arr)
    detection_gray = _scalar_to_detection_uint8(arr)
    return LoadedImage(
        raw=arr,
        display_rgb=_gray_to_rgb(display_gray),
        detection_rgb=_gray_to_rgb(detection_gray),
        is_scalar=True,
        display_note="Scalar image contrast enhanced for display",
    )


def blue_channel(image_rgb: np.ndarray) -> np.ndarray:
    return image_rgb[..., 2]


def rgb_to_qimage_bytes(image_rgb: np.ndarray) -> tuple[bytes, int, int, int]:
    h, w, channels = image_rgb.shape
    if channels != 3:
        raise ValueError("Expected RGB image")
    contiguous = np.ascontiguousarray(image_rgb)
    return contiguous.tobytes(), w, h, w * 3


def _is_rgb_like(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[2] >= 3


def _rgb_like_to_uint8(arr: np.ndarray) -> np.ndarray:
    rgb = arr[..., :3]
    if rgb.dtype == np.uint8:
        return rgb.astype(np.uint8, copy=False)

    channels = [_scalar_to_display_uint8(rgb[..., channel]) for channel in range(3)]
    return np.stack(channels, axis=2)


def _scalar_to_display_uint8(arr: np.ndarray) -> np.ndarray:
    normalized = _robust_normalize_scalar(arr, lower_percentile=0.5, upper_percentile=99.5)
    normalized = _apply_gamma(normalized, gamma=0.55)
    return (normalized * 255).astype(np.uint8)


def _scalar_to_detection_uint8(arr: np.ndarray) -> np.ndarray:
    normalized = _robust_normalize_scalar(arr, lower_percentile=1.0, upper_percentile=99.7)
    normalized = _apply_gamma(normalized, gamma=0.55)
    return (normalized * 255).astype(np.uint8)


def _robust_normalize_scalar(
    arr: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32, copy=False) / 255.0

    values = np.asarray(arr, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.percentile(values, [lower_percentile, upper_percentile])
    if hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32)

    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _apply_gamma(values: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return values
    return np.power(np.clip(values, 0.0, 1.0), gamma).astype(np.float32)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=2)
