"""Draw constellation guides and names over astronomical images.

The COCO annotations in this project identify the bounding region of each
constellation.  This module detects bright points inside each region and
connects them with a deterministic nearest-neighbour skeleton.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .preprocessing import InsufficientStarsError, detect_stars


DEFAULT_COLOURS = (
    (80, 220, 255),
    (255, 170, 80),
    (120, 255, 150),
    (220, 140, 255),
)


# Ajusta una caja COCO a los límites válidos de la imagen.
def _clip_bbox(bbox: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    left = max(0, min(width - 1, int(round(x))))
    top = max(0, min(height - 1, int(round(y))))
    right = max(left + 1, min(width, int(round(x + w))))
    bottom = max(top + 1, min(height, int(round(y + h))))
    return left, top, right, bottom


def detect_star_centres(
    image: np.ndarray,
    bbox: Sequence[float],
    *,
    threshold_sigma: float = 1.8,
    min_area: int = 2,
    max_area: int = 250,
) -> list[tuple[int, int]]:
    """Return preprocessed bright-star centres inside ``bbox``."""
    # Recorta la región de interés para buscar estrellas solo dentro de la constelación.
    height, width = image.shape[:2]
    left, top, right, bottom = _clip_bbox(bbox, width, height)
    roi = image[top:bottom, left:right]
    try:
        # Intenta primero el detector principal basado en el preprocesado del proyecto.
        stars = detect_stars(
            roi,
            area_min=min_area,
            area_max=max_area,
            n_min=1,
        )
    except InsufficientStarsError:
        # Una región con pocas estrellas no debe impedir dibujar el resto de la imagen.
        stars = np.empty((0, 3))

    # Convierte las coordenadas locales de la región a coordenadas globales de la imagen.
    points = [(int(round(left + x)), int(round(top + y))) for x, y, _ in stars]
    if len(points) >= 2:
        return points

    # Small or low-contrast boxes may not provide enough components for the
    # adaptive Top-Hat pass; retain the simpler detector as a visual fallback.
    # Fallback sencillo: resalta píxeles brillantes y busca componentes conectados.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mean, std = cv2.meanStdDev(gray)
    threshold = max(20.0, float(mean[0, 0] + threshold_sigma * std[0, 0]))
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    fallback: list[tuple[int, int]] = []
    for label in range(1, count):
        # Ignora el fondo y los objetos demasiado pequeños o demasiado grandes.
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            cx, cy = centroids[label]
            fallback.append((int(round(left + cx)), int(round(top + cy))))
    return fallback


def _nearest_neighbour_edges(
    points: Sequence[tuple[int, int]], max_distance: float
) -> list[tuple[int, int]]:
    """Build a sparse, deterministic graph over detected stars."""
    if len(points) < 2:
        return []

    # Calcula todas las distancias para conectar cada estrella con su vecina más cercana.
    edges: set[tuple[int, int]] = set()
    coordinates = np.asarray(points, dtype=np.float32)
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    for index in range(len(points)):
        distances[index, index] = np.inf
        neighbour = int(np.argmin(distances[index]))
        # Limita las líneas para evitar unir estrellas que no pertenecen a la misma figura.
        if distances[index, neighbour] <= max_distance:
            edges.add(tuple(sorted((index, neighbour))))
    return sorted(edges)


def _draw_label(
    image: np.ndarray,
    name: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int],
) -> None:
    # Calcula un tamaño de texto proporcional a la resolución de la imagen.
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(image.shape[:2]) / 1400)
    thickness = max(1, int(round(scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(name, font, scale, thickness)
    x = max(0, min(image.shape[1] - text_width - 4, x))
    y = max(text_height + baseline + 2, min(image.shape[0] - 2, y))
    # Dibuja una sombra negra antes del texto para mejorar su legibilidad.
    cv2.putText(image, name, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, name, (x, y), font, scale, colour, thickness, cv2.LINE_AA)


def draw_constellation_overlay(
    image: np.ndarray,
    detections: Iterable[Mapping[str, Any]],
    *,
    alpha: float = 0.85,
    line_thickness: int = 1,
    draw_star_points: bool = True,
) -> np.ndarray:
    """Return a copy of ``image`` annotated with constellation lines and names.

    Each detection must contain ``name`` and ``bbox`` in COCO format
    ``[x, y, width, height]``.  A detection may optionally provide
    ``colour`` and ``stars``; otherwise stars are extracted from the image.
    """
    if image is None or image.size == 0:
        raise ValueError("image must contain pixel data")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1]")

    # Trabaja sobre copias para no modificar la imagen recibida por el llamador.
    result = image.copy()
    overlay = result.copy()
    for index, detection in enumerate(detections):
        # Lee los datos de la detección y asigna un color estable a cada constelación.
        name = str(detection["name"])
        bbox = detection["bbox"]
        colour = tuple(detection.get("colour", DEFAULT_COLOURS[index % len(DEFAULT_COLOURS)]))
        points = [tuple(map(int, point)) for point in detection.get("stars", ())]
        if not points:
            # Si no se proporcionan estrellas, las detecta automáticamente en la caja.
            points = detect_star_centres(image, bbox)
        if not points:
            # Aunque no haya estrellas, conserva el nombre de la constelación.
            left, top, right, bottom = _clip_bbox(bbox, image.shape[1], image.shape[0])
            _draw_label(result, name, (left, top), colour)
            continue

        left, top, right, bottom = _clip_bbox(bbox, image.shape[1], image.shape[0])
        diagonal = float(np.hypot(right - left, bottom - top))
        # Usa la diagonal de la caja para adaptar la distancia máxima entre estrellas.
        edges = _nearest_neighbour_edges(points, max_distance=max(18.0, diagonal * 0.45))
        for start, end in edges:
            cv2.line(overlay, points[start], points[end], colour, line_thickness, cv2.LINE_AA)
        if draw_star_points:
            for point in points:
                cv2.circle(overlay, point, max(2, line_thickness + 1), colour, -1, cv2.LINE_AA)
        _draw_label(result, name.replace("_", " ").title(), (left, top), colour)

    # Mezcla las líneas y puntos con la imagen original usando la transparencia indicada.
    cv2.addWeighted(overlay, alpha, result, 1.0 - alpha, 0, result)
    return result


def overlay_from_coco(
    image_path: str | Path,
    annotation_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Annotate one image using the matching image entry in a COCO file."""
    # Normaliza las rutas y carga la imagen de entrada.
    image_path = Path(image_path)
    annotation_path = Path(annotation_path)
    output_path = Path(output_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    # Lee el archivo COCO y localiza la entrada correspondiente a la imagen.
    dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_entry = next(
        (
            entry
            for entry in dataset["images"]
            if Path(entry["file_name"]).name == image_path.name
        ),
        None,
    )
    if image_entry is None:
        raise ValueError(f"image is not present in COCO annotations: {image_path.name}")

    # Mapea los IDs de categoría a sus nombres para interpretar las anotaciones.
    categories = {category["id"]: category["name"] for category in dataset["categories"]}
    # Prepara solo las constelaciones; se excluyen objetos auxiliares como estrellas y luna.
    detections = [
        {"name": categories[annotation["category_id"]], "bbox": annotation["bbox"]}
        for annotation in dataset["annotations"]
        if annotation["image_id"] == image_entry["id"]
        and categories[annotation["category_id"]] not in {"star", "moon"}
    ]
    # Crea la carpeta de salida y guarda la imagen anotada.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), draw_constellation_overlay(image, detections)):
        raise OSError(f"could not write output image: {output_path}")
    return output_path
