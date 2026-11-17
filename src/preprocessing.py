"""Preprocesado de Dani"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class InsufficientStarsError(ValueError):
    """Raised when a region contains fewer stars than requested."""


def structuring_element_size(area_max: int, margin: int = 2) -> int:
    """Return an odd elliptical-kernel size based on the maximum star area."""
    diameter = int(np.ceil(2 * np.sqrt(area_max / np.pi)))
    size = diameter + margin
    return size if size % 2 else size + 1


def _detect_components(
    threshold: int,
    top_hat: np.ndarray,
    area_min: int,
    area_max: int,
    ratio_max: float,
    fill_min: float,
    shape_area_min: int,
) -> list[tuple[float, float, float]]:
    _, binary = cv2.threshold(top_hat, threshold, 255, cv2.THRESH_BINARY)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    stars: list[tuple[float, float, float]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not area_min <= area <= area_max:
            continue

        if area >= shape_area_min:
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            ratio = max(width, height) / min(width, height)
            fill = area / (width * height)
            if ratio > ratio_max or fill < fill_min:
                continue

        brightness = float(top_hat[labels == label].max())
        stars.append((float(centroids[label][0]), float(centroids[label][1]), brightness))
    return stars


def detect_stars(
    image: np.ndarray | str | Path,
    *,
    area_min: int = 2,
    area_max: int = 150,
    max_iter: int = 50,
    n_min: int = 3,
    n_max: int = 15,
    initial_threshold: int = 225,
    threshold_step: int = 5,
    structuring_size: int | None = None,
    ratio_max: float = 2.0,
    fill_min: float = 0.5,
    shape_area_min: int = 9,
    minimum_threshold: int = 20,
) -> np.ndarray:
    """Detect stars and return ``(x, y, brightness)`` rows."""
    if isinstance(image, (str, Path)):
        grayscale = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if grayscale is None:
            raise FileNotFoundError(image)
    else:
        if image is None or image.size == 0:
            raise ValueError("image must contain pixel data")
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    if structuring_size is None:
        structuring_size = structuring_element_size(area_max)
    median = cv2.medianBlur(grayscale, 3)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (structuring_size, structuring_size)
    )
    top_hat = cv2.morphologyEx(median, cv2.MORPH_TOPHAT, kernel)

    threshold = initial_threshold
    args = (area_min, area_max, ratio_max, fill_min, shape_area_min)
    stars = _detect_components(threshold, top_hat, *args)
    for _ in range(max_iter):
        if len(stars) >= n_min or threshold <= minimum_threshold:
            break
        threshold = max(0, threshold - threshold_step)
        stars = _detect_components(threshold, top_hat, *args)

    if len(stars) < n_min:
        raise InsufficientStarsError(
            f"{len(stars)} stars detected (minimum {n_min}), final threshold {threshold}"
        )
    stars.sort(key=lambda star: star[2], reverse=True)
    return np.asarray(stars[:n_max], dtype=float).reshape(-1, 3)