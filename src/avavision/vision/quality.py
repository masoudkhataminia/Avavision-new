"""Capture quality: sharpness, exposure and glare, for the whole frame and for each compartment.

Thresholds must be calibrated on the real station. Set the camera exposure so the whitest tablet stays below
``compartment_glare_level``: only then can a near-white, colourless patch be told apart from a white tablet.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel

from ..core.geometry import Rect
from ..core.models import CompartmentIndex, PackLayout
from ..core.observation import CaptureIssue, QualityAssessment, QualityMetrics


class QualityPolicy(BaseModel):
    analysis_max_dimension: int = 640
    minimum_sharpness: float = 60
    minimum_mean_luminance: float = 50
    maximum_mean_luminance: float = 215
    shadow_clip_level: int = 12
    maximum_shadow_clip_fraction: float = 0.35
    glare_level: int = 250
    maximum_glare_fraction: float = 0.03
    #: A compartment is obscured when more than ``maximum_compartment_glare_fraction`` of its inner area (inset
    #: by ``compartment_inset`` of the cell on each side) is near-white and colourless. Set from the first real
    #: photos (D-130): glare covered 12–22 % of the affected compartments, all others stayed below 2 %.
    compartment_glare_level: int = 245
    compartment_glare_saturation: int = 40
    maximum_compartment_glare_fraction: float = 0.03
    compartment_inset: float = 0.1


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def crop_normalized(image: np.ndarray, rect: Rect) -> np.ndarray | None:
    h, w = image.shape[:2]
    x0, y0 = max(0, int(rect.x * w)), max(0, int(rect.y * h))
    x1, y1 = min(w, int(np.ceil(rect.max_x * w))), min(h, int(np.ceil(rect.max_y * h)))
    return image[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else None


def assess(image: np.ndarray, region: Rect | None = None, policy: QualityPolicy | None = None) -> QualityAssessment:
    policy = policy or QualityPolicy()
    gray = to_gray(image)
    if region is not None and (cropped := crop_normalized(gray, region)) is not None:
        gray = cropped
    scale = policy.analysis_max_dimension / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if min(gray.shape) < 3:
        return QualityAssessment(issues=[CaptureIssue.BLURRY])
    metrics = QualityMetrics(
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F, ksize=1)[1:-1, 1:-1].var()),
        mean_luminance=float(gray.mean()),
        shadow_clip_fraction=float((gray <= policy.shadow_clip_level).mean()),
        glare_fraction=float((gray >= policy.glare_level).mean()),
    )
    issues = []
    if metrics.sharpness < policy.minimum_sharpness:
        issues.append(CaptureIssue.BLURRY)
    if (
        metrics.mean_luminance < policy.minimum_mean_luminance
        or metrics.shadow_clip_fraction > policy.maximum_shadow_clip_fraction
    ):
        issues.append(CaptureIssue.TOO_DARK)
    if metrics.mean_luminance > policy.maximum_mean_luminance:
        issues.append(CaptureIssue.TOO_BRIGHT)
    if metrics.glare_fraction > policy.maximum_glare_fraction:
        issues.append(CaptureIssue.GLARE)
    return QualityAssessment(issues=issues, metrics=metrics)


def obscured_compartments(
    canvas: np.ndarray, layout: PackLayout, policy: QualityPolicy | None = None
) -> list[CompartmentIndex]:
    """Compartments of the perspective-corrected pack whose content glare or a covering hides.

    A frame can pass the whole-frame glare check while a reflection still hides the tablets of one or two
    compartments; those compartments must go to the pharmacist instead of being counted."""
    policy = policy or QualityPolicy()
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    washed = (hsv[..., 2] >= policy.compartment_glare_level) & (hsv[..., 1] <= policy.compartment_glare_saturation)
    washed = cv2.morphologyEx(washed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    inset = policy.compartment_inset
    result = []
    for index in layout.all_compartments:
        rect = layout.cell_rect(index)
        inner = Rect(
            x=rect.x + rect.width * inset,
            y=rect.y + rect.height * inset,
            width=rect.width * (1 - 2 * inset),
            height=rect.height * (1 - 2 * inset),
        )
        pocket = crop_normalized(washed, inner)
        if pocket is not None and float(pocket.mean()) > policy.maximum_compartment_glare_fraction:
            result.append(index)
    return result
