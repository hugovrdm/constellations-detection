# Constelaciones: superposición sobre imágenes

El módulo `src/constellation_overlay.py` implementa la visualización solicitada:

1. lee las cajas COCO de cada constelación;
2. detecta los puntos brillantes dentro de cada caja;
3. conecta los puntos mediante un esqueleto de vecino más cercano;
4. escribe el nombre de la constelación sobre una copia de la imagen original.

Ejemplo:

```bash
python scripts/overlay_constellations.py \
  --image data/raw/test/2022-09-06-00-00-00-s_png_jpg.rf.07d6574c16079b22fceece07860ae05d.jpg \
  --annotations data/raw/test/_annotations.coco.json \
  --output data/processed/2022-09-06-overlay.jpg
```

La función reutilizable es `draw_constellation_overlay(image, detections)`. Las
detecciones usan el formato COCO:

```python
{"name": "orion", "bbox": [x, y, width, height]}
```

## Relación con la literatura

La estrategia sigue la idea de separar preprocesado/detección de estrellas,
identificación geométrica y visualización:

- Ji, Liu y Wang, **“Constellation Detection”**, EE368, Stanford University.
  El trabajo describe preprocesado de imágenes, comparación con patrones de
  constelaciones y superposición de líneas y nombres:
  <https://web.stanford.edu/class/ee368/Project_Spring_1415/Reports/Ji_Liu_Wang.pdf>
- Nadamoto et al., **“Constellation identification method using point set
  data”** (2023), DOI: `10.1007/s10015-023-00860-4`.
  El método representa las estrellas como un conjunto de puntos y usa sus
  relaciones geométricas para identificar constelaciones:
  <https://doi.org/10.1007/s10015-023-00860-4>

Las anotaciones actuales de Roboflow contienen cajas de constelación, no la
correspondencia de cada estrella con una plantilla IAU. Por ello esta primera
implementación infiere el esqueleto local a partir de los centroides brillantes.
Para una identificación astronómica completa habría que añadir una plantilla
de estrellas/segmentos por constelación y resolver una transformación
geométrica (o entrenar el detector de constelaciones).
