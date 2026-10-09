"""Measuring one pill: real size in millimetres and the colour of its face.

The crop comes from the perspective-corrected pack, so pixels have a known size (the layout's millimetres over
the canvas). The pill is separated from the compartment floor by its colour distance from the crop's border.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.models import PackLayout
from ..core.physical import PhysicalFeatures
from .segmentation import Rectified

#: Lightness counts a little less than hue: the blister and shadows change it most.
_WEIGHTS = np.array([0.9, 1.0, 1.0], dtype=np.float32)
MINIMUM_CONTRAST = 9.0


def millimetres_per_pixel(rectified: Rectified, layout: PackLayout) -> float:
    h, w = rectified.canvas.shape[:2]
    return float((layout.width_mm / w + layout.height_mm / h) / 2)


def measure_pill(crop: np.ndarray, mm_per_px: float) -> PhysicalFeatures | None:
    """``None`` when no single clear pill stands out from the floor of the crop."""
    if crop is None or crop.ndim != 3 or min(crop.shape[:2]) < 8:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    floor = np.median(border, axis=0)
    distance = cv2.GaussianBlur(np.sqrt((((lab - floor) * _WEIGHTS) ** 2).sum(axis=2)), (0, 0), 1.0)
    if float(distance.max()) < MINIMUM_CONTRAST:
        return None
    scaled = np.clip(distance * (255.0 / float(distance.max())), 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(MINIMUM_CONTRAST, otsu * float(distance.max()) / 255.0)
    mask = (distance > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    h, w = mask.shape
    centre = (w / 2, h / 2)
    inside = [c for c in contours if cv2.pointPolygonTest(c, centre, False) >= 0]
    pill = max(inside or contours, key=cv2.contourArea)
    if cv2.contourArea(pill) < 0.03 * h * w:
        return None
    (_, _), (rw, rh), _ = cv2.minAreaRect(pill)
    face = np.zeros_like(mask)
    cv2.drawContours(face, [pill], -1, 255, cv2.FILLED)
    eroded = cv2.erode(face, np.ones((3, 3), np.uint8), iterations=2)
    face = eroded if eroded.any() else face
    l_value, a_value, b_value = (float(v) for v in lab[face > 0].mean(axis=0))
    return PhysicalFeatures(
        length_mm=round(max(rw, rh) * mm_per_px, 2),
        width_mm=round(min(rw, rh) * mm_per_px, 2),
        lightness=round(l_value * 100 / 255, 2),
        a=round(a_value - 128, 2),
        b=round(b_value - 128, 2),
    )
