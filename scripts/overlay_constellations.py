#!/usr/bin/env python3
"""Overlay detected constellation guides on one image or a directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow execution as ``python scripts/overlay_constellations.py`` from the
# repository root without requiring the project to be installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constellation_overlay import overlay_from_coco


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(overlay_from_coco(args.image, args.annotations, args.output))


if __name__ == "__main__":
    main()
