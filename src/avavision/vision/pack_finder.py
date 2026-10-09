"""Finding the pack in a camera frame.

Two methods:
- **Markers (recommended for the station):** four ArUco markers printed on the tray, ids 0–3 at the
  top-left, top-right, bottom-right and bottom-left corners. Sub-pixel accurate, orientation is never
  ambiguous, and it works whatever colour the pack is.
- **Outline:** the pack card's own outline, for setups without a tray. The card may be lighter, darker or more
  colourful than the background (real cards are often dark blue or green on a white bench).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from ..core.geometry import Point, Quad
from ..core.models import PackLayout

MARKER_DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_IDS = (0, 1, 2, 3)  # top-left, top-right, bottom-right, bottom-left


class FinderMode(StrEnum):
    MARKERS = "markers"
    OUTLINE = "outline"


class Orientation(StrEnum):
    """For the outline method only: how the pack lies in the image."""

    AUTOMATIC = "automatic"
    UPRIGHT = "upright"
    QUARTER_TURN = "quarterTurn"
    HALF_TURN = "halfTurn"
    THREE_QUARTER_TURN = "threeQuarterTurn"


@dataclass
class FoundPack:
    quad: Quad  # normalized image coordinates
    confidence: float


def _detector() -> cv2.aruco.ArucoDetector:
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICTIONARY), parameters)


_ARUCO = _detector()


def find_by_markers(image: np.ndarray) -> FoundPack | None:
    """Quad through the centres of markers 0–3; ``None`` unless all four are visible."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _ARUCO.detectMarkers(gray)
    if ids is None:
        return None
    centres = {int(i): c.reshape(4, 2).mean(axis=0) for i, c in zip(ids.ravel(), corners, strict=True)}
    if not all(i in centres for i in MARKER_IDS):
        return None
    h, w = gray.shape[:2]
    points = [Point(x=float(centres[i][0] / w), y=float(centres[i][1] / h)) for i in MARKER_IDS]
    return FoundPack(quad=Quad.from_corners(points), confidence=1.0)


def _order_corners(points: np.ndarray) -> np.ndarray:
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(s)], points[np.argmin(d)], points[np.argmax(s)], points[np.argmax(d)]])


def _outline_masks(image: np.ndarray) -> list[np.ndarray]:
    """Candidate card masks: brighter than the background, darker than it, and (in colour) more saturated."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    masks = []
    if float(blurred.std()) >= 4:  # a featureless frame has no brightness edge
        _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks += [bright, cv2.bitwise_not(bright)]
    if image.ndim == 3:
        saturation = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1], (5, 5), 0)
        level, coloured = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if level >= 40:  # a real colour difference, not noise in a grey scene
            masks.append(coloured)
    return [cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) for m in masks]


def _quad(hull: np.ndarray) -> np.ndarray | None:
    """The hull simplified to four corners, loosening the tolerance until it fits."""
    perimeter = cv2.arcLength(hull, True)
    for tolerance in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, tolerance * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(float)
        if len(approx) < 4:
            return None
    return None


def find_by_outline(image: np.ndarray, analysis_size: int = 1000, minimum_confidence: float = 0.75) -> FoundPack | None:
    """The largest card-like quadrilateral covering a fifth to 95 % of the frame (grey or BGR ``image``).

    Blisters, labels and reflections break a card's silhouette, so each candidate is closed into its convex
    hull; confidence is how solid the candidate is within that hull and how well four corners fit it. The
    largest acceptable candidate wins, because a striped card also yields smaller, equally solid bands."""
    scale = min(1.0, analysis_size / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    h, w = small.shape[:2]
    best: FoundPack | None = None
    best_area = 0.0
    for mask in _outline_masks(small):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(contour)
            if area < 0.2 * w * h:
                break
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            corners = _quad(hull)
            if corners is None or hull_area > 0.95 * w * h:  # the whole frame, not a card on a background
                continue
            ordered = _order_corners(corners)
            quad_area = cv2.contourArea(ordered.astype(np.float32))
            fit = min(quad_area, hull_area) / max(quad_area, hull_area, 1)
            confidence = float(area / max(hull_area, 1) * fit)
            if confidence >= minimum_confidence and hull_area > best_area:
                points = [Point(x=float(x / w), y=float(y / h)) for x, y in ordered]
                best, best_area = FoundPack(quad=Quad.from_corners(points), confidence=confidence), hull_area
    return best


def quarter_turns(quad: Quad, orientation: Orientation, image_w: int, image_h: int, layout: PackLayout) -> int:
    fixed = {
        Orientation.UPRIGHT: 0,
        Orientation.QUARTER_TURN: 1,
        Orientation.HALF_TURN: 2,
        Orientation.THREE_QUARTER_TURN: 3,
    }
    if orientation in fixed:
        return fixed[orientation]
    pixels = quad.scaled(image_w, image_h)
    landscape = pixels.top_left.distance(pixels.top_right) >= pixels.top_left.distance(pixels.bottom_left)
    return 0 if landscape == (layout.aspect_ratio >= 1) else 1


def find_pack(
    image: np.ndarray, mode: FinderMode, layout: PackLayout, orientation: Orientation = Orientation.AUTOMATIC
) -> FoundPack | None:
    """``image`` is grey or BGR; the outline method also uses colour when it has it."""
    if mode == FinderMode.MARKERS:
        return find_by_markers(image)
    found = find_by_outline(image)
    if found is None:
        return None
    turns = quarter_turns(found.quad, orientation, image.shape[1], image.shape[0], layout)
    return FoundPack(quad=found.quad.rotated(turns), confidence=found.confidence)


def marker_sheet(marker_pixels: int = 400, margin: int = 60) -> np.ndarray:
    """A printable sheet with the four tray markers (cut out and stick on the tray corners)."""
    dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICTIONARY)
    cell = marker_pixels + 2 * margin
    sheet = np.full((2 * cell, 2 * cell), 255, np.uint8)
    for k, marker_id in enumerate(MARKER_IDS):
        row, col = [(0, 0), (0, 1), (1, 1), (1, 0)][k]
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_pixels)
        y, x = row * cell + margin, col * cell + margin
        sheet[y : y + marker_pixels, x : x + marker_pixels] = marker
        cv2.putText(sheet, f"id {marker_id}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    return sheet
