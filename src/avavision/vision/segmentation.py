"""Training-free pill finding: the pack is rectified, then every compartment is segmented by two independent
methods (colour difference and edges). Agreement gives confident pills; disagreement, or a blob that looks
like touching pills, gives low-confidence pills that the decision engine sends to review."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..core.geometry import Rect
from ..core.models import CompartmentIndex, PackLayout
from ..core.observation import Detection, Registration


@dataclass
class Rectified:
    """The pack warped to a flat, axis-aligned canvas."""

    canvas: np.ndarray  # BGR
    image_to_canvas: np.ndarray  # 3×3, image pixels → canvas pixels
    canvas_to_image: np.ndarray
    image_size: tuple[int, int]  # (width, height)

    @property
    def pixels_per_unit(self) -> tuple[float, float]:
        return float(self.canvas.shape[1]), float(self.canvas.shape[0])

    def canvas_box_to_image(self, x0: float, y0: float, x1: float, y1: float) -> Rect:
        corners = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.float64)
        mapped = cv2.perspectiveTransform(corners, self.canvas_to_image).reshape(4, 2)
        w, h = self.image_size
        xs, ys = mapped[:, 0] / w, mapped[:, 1] / h
        return Rect(
            x=float(xs.min()), y=float(ys.min()), width=float(xs.max() - xs.min()), height=float(ys.max() - ys.min())
        )

    def image_box_to_canvas(self, box: Rect) -> tuple[int, int, int, int]:
        w, h = self.image_size
        corners = np.array(
            [
                [[box.x * w, box.y * h]],
                [[box.max_x * w, box.y * h]],
                [[box.max_x * w, box.max_y * h]],
                [[box.x * w, box.max_y * h]],
            ],
            dtype=np.float64,
        )
        mapped = cv2.perspectiveTransform(corners, self.image_to_canvas).reshape(4, 2)
        x0, y0 = np.floor(mapped.min(axis=0)).astype(int)
        x1, y1 = np.ceil(mapped.max(axis=0)).astype(int)
        return int(x0), int(y0), int(x1), int(y1)


def rectify(image: np.ndarray, registration: Registration, layout: PackLayout, width: int = 1400) -> Rectified:
    h, w = image.shape[:2]
    canvas_w = width
    canvas_h = round(width / layout.aspect_ratio)
    src = np.array([[c.x * w, c.y * h] for c in registration.quad.corners], dtype=np.float32)
    dst = np.array([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]], dtype=np.float32)
    forward = cv2.getPerspectiveTransform(src, dst)
    canvas = cv2.warpPerspective(image, forward, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
    return Rectified(canvas=canvas, image_to_canvas=forward, canvas_to_image=np.linalg.inv(forward), image_size=(w, h))


@dataclass
class Region:
    x0: int
    y0: int
    x1: int
    y1: int
    area: int
    #: Area over convex-hull area. A single tablet or capsule is convex (close to 1).
    solidity: float
    #: Deepest notch in the outline relative to the blob's smaller side. Touching pills leave a notch
    #: between them even when they overlap enough to look like one capsule.
    notch: float
    #: Distance-transform maxima, used to estimate how many pills a non-convex blob holds.
    peaks: int

    def touching(self, policy: SegmentationPolicy) -> bool:
        return self.solidity < policy.minimum_solidity or self.notch > policy.maximum_notch


@dataclass
class SegmentationPolicy:
    inset: float = 0.06  # fraction of each compartment trimmed on every side (blister walls)
    minimum_area_fraction: float = 0.004
    maximum_area_fraction: float = 0.45
    minimum_colour_contrast: float = 9.0  # Lab distance from the compartment background
    deviation_multiplier: float = 5.0
    minimum_edge_strength: float = 28.0
    minimum_solidity: float = 0.9
    maximum_notch: float = 0.12
    agreed_confidence: float = 0.9
    disagreed_confidence: float = 0.5


def _components(mask: np.ndarray, policy: SegmentationPolicy) -> list[Region]:
    h, w = mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    regions = []
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        fraction = area / (w * h)
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not (policy.minimum_area_fraction <= fraction <= policy.maximum_area_fraction) or touches:
            continue
        blob = (labels[y : y + bh, x : x + bw] == label).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(contour))
        solidity = min(1.0, float(area / hull)) if hull > 0 else 1.0
        notch = 0.0
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if len(contour) > 3 and hull_indices is not None and len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
            except cv2.error:
                defects = None
            if defects is not None:
                (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
                notch = float(defects[:, 0, 3].max() / 256.0 / max(1.0, min(rw, rh)))
        regions.append(Region(int(x), int(y), int(x + bw), int(y + bh), int(area), solidity, notch, _peak_count(blob)))
    return regions


def _peak_count(blob: np.ndarray) -> int:
    """Separate maxima of the distance transform: one per round-ish pill in a blob."""
    distance = cv2.distanceTransform(np.pad(blob, 1), cv2.DIST_L2, 5)
    top = float(distance.max())
    if top < 3:
        return 1
    peaks = (distance >= 0.7 * top).astype(np.uint8)
    count, _ = cv2.connectedComponents(peaks)
    return max(2, count - 1)


def _fill(mask: np.ndarray, close: int = 7) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def colour_regions(pocket: np.ndarray, policy: SegmentationPolicy) -> list[Region]:
    lab = cv2.cvtColor(pocket, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    ring = max(1, min(h, w) // 12)
    border = np.concatenate(
        [
            lab[:ring].reshape(-1, 3),
            lab[-ring:].reshape(-1, 3),
            lab[:, :ring].reshape(-1, 3),
            lab[:, -ring:].reshape(-1, 3),
        ]
    )
    background = np.median(border, axis=0)
    weights = np.array([0.9, 1.0, 1.0], dtype=np.float32)  # lightness slightly discounted against soft shadows
    distance = cv2.GaussianBlur(np.sqrt((((lab - background) * weights) ** 2).sum(axis=2)), (5, 5), 0)
    border_distance = np.sqrt((((border - background) * weights) ** 2).sum(axis=1))
    threshold = max(policy.minimum_colour_contrast, float(np.median(border_distance)) * policy.deviation_multiplier)
    return _components(_fill((distance > threshold).astype(np.uint8)), policy)


def edge_regions(pocket: np.ndarray, policy: SegmentationPolicy) -> list[Region]:
    gray = cv2.GaussianBlur(cv2.cvtColor(pocket, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    magnitude = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    edges = (magnitude > policy.minimum_edge_strength).astype(np.uint8)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    return _components(_fill(edges, close=3), policy)


@dataclass
class PocketResult:
    compartment: CompartmentIndex
    boxes: list[tuple[int, int, int, int]]  # canvas pixels
    confidence: float
    agreed: bool


class PocketSegmenter:
    """The built-in detector used until AvaVision's own trained detector ships."""

    def __init__(self, policy: SegmentationPolicy | None = None):
        self.policy = policy or SegmentationPolicy()

    def segment(self, rectified: Rectified, layout: PackLayout) -> list[PocketResult]:
        canvas = rectified.canvas
        cw, ch = canvas.shape[1], canvas.shape[0]
        results = []
        for index in layout.all_compartments:
            cell = layout.cell_rect(index)
            dx, dy = cell.width * self.policy.inset, cell.height * self.policy.inset
            x0, y0 = int((cell.x + dx) * cw), int((cell.y + dy) * ch)
            x1, y1 = int((cell.max_x - dx) * cw), int((cell.max_y - dy) * ch)
            pocket = canvas[y0:y1, x0:x1]
            if pocket.shape[0] < 8 or pocket.shape[1] < 8:
                continue
            colour = colour_regions(pocket, self.policy)
            edge = edge_regions(pocket, self.policy)
            touching = any(r.touching(self.policy) for r in colour)
            agreed = len(colour) == len(edge) and not touching
            chosen = colour if len(colour) >= len(edge) else edge
            boxes = []
            for r in chosen:
                # Touching pills: one low-confidence box per suspected pill keeps the count honest.
                parts = r.peaks if r.touching(self.policy) else 1
                share = (r.x1 - r.x0) / parts
                for k in range(parts):
                    boxes.append((x0 + r.x0 + int(k * share), y0 + r.y0, x0 + r.x0 + int((k + 1) * share), y0 + r.y1))
            confidence = self.policy.agreed_confidence if agreed else self.policy.disagreed_confidence
            results.append(PocketResult(index, boxes, confidence, agreed))
        return results

    def detect(self, rectified: Rectified, layout: PackLayout) -> list[Detection]:
        return [
            Detection(label="pill", confidence=pocket.confidence, box=rectified.canvas_box_to_image(*box))
            for pocket in self.segment(rectified, layout)
            for box in pocket.boxes
        ]
