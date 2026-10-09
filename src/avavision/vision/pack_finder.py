"""Finding the pack in a camera frame.

Two methods:
- **Markers (recommended for the station):** four ArUco markers printed on the tray, ids 0–3 at the
  top-left, top-right, bottom-right and bottom-left corners. Sub-pixel accurate, orientation is never
  ambiguous, and it works whatever colour the pack is.
- **Outline:** the largest bright quadrilateral in the frame, for setups without a tray.
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


def find_by_markers(gray: np.ndarray) -> FoundPack | None:
    """Quad through the centres of markers 0–3; ``None`` unless all four are visible."""
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


def find_by_outline(gray: np.ndarray, analysis_size: int = 1000) -> FoundPack | None:
    """Largest convex quadrilateral covering at least a fifth of the frame."""
    scale = min(1.0, analysis_size / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    h, w = small.shape[:2]
    if float(blurred.std()) < 4:  # a featureless frame: nothing to find
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: FoundPack | None = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(contour)
        if area < 0.2 * w * h:
            break
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        ordered = _order_corners(approx.reshape(4, 2).astype(float))
        if area > 0.95 * w * h:  # the whole frame, not a pack on a background
            continue
        rect_area = cv2.contourArea(ordered.astype(np.float32))
        confidence = float(min(1.0, area / max(rect_area, 1)))
        points = [Point(x=float(x / w), y=float(y / h)) for x, y in ordered]
        best = FoundPack(quad=Quad.from_corners(points), confidence=confidence)
        break
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
    gray: np.ndarray, mode: FinderMode, layout: PackLayout, orientation: Orientation = Orientation.AUTOMATIC
) -> FoundPack | None:
    if mode == FinderMode.MARKERS:
        return find_by_markers(gray)
    found = find_by_outline(gray)
    if found is None:
        return None
    turns = quarter_turns(found.quad, orientation, gray.shape[1], gray.shape[0], layout)
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
