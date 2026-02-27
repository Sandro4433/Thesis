# vision_apriltag.py
from __future__ import annotations

from typing import Any, List

import numpy as np
from pupil_apriltags import Detector


def create_detector(
    families: str = "tag25h9",
    nthreads: int = 4,
    quad_decimate: float = 1.0,
    quad_sigma: float = 0.0,
    refine_edges: int = 1,
) -> Detector:
    return Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
    )


def detect_tags(detector: Detector, gray: np.ndarray) -> List[Any]:
    return detector.detect(gray)