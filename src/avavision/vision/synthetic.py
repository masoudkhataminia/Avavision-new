"""A simulated verification station: renders a tray with markers and a filled pack.

Used by tests, benchmarks and the app's demo camera, so everything can be tried without hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..core.geometry import Rect
from ..core.models import CompartmentIndex, PackLayout, cell
from .codes import code_image
from .pack_finder import MARKER_DICTIONARY, MARKER_IDS

#: Colour (BGR), axes (px at 1920 wide) and shape of a few fictional medications. White tablets stay below
#: clipping, as the station's exposure must keep them (see ``vision.quality``).
PALETTE: dict[str, tuple[tuple[int, int, int], tuple[int, int], str]] = {
    "metformin-500": ((232, 235, 237), (26, 14), "oval"),
    "atorvastatin-20": ((238, 238, 238), (14, 14), "round"),
    "aspirin-100": ((70, 175, 235), (12, 12), "round"),
    "perindopril-5": ((120, 200, 120), (16, 10), "oval"),
    "amlodipine-5": ((236, 236, 236), (24, 9), "capsule"),
    "warfarin-1": ((90, 120, 200), (11, 11), "round"),
}


@dataclass
class Scene:
    """Content of each compartment: medication ids (repeated for quantity)."""

    contents: dict[CompartmentIndex, list[str]] = field(default_factory=dict)
    foreign: list[CompartmentIndex] = field(default_factory=list)
    #: Text of the QR code on the pack's header card, shown beside the tray.
    card: str | None = None


@dataclass
class Station:
    width: int = 1920
    height: int = 1080
    marker_size: int = 90
    #: Marker-centre rectangle in image pixels (x0, y0, x1, y1).
    frame: tuple[int, int, int, int] = (260, 120, 1660, 960)
    rows: int = 4
    columns: int = 7
    #: Compartment grid inside the marker frame (normalized).
    grid: Rect = field(default_factory=lambda: Rect(x=0.08, y=0.1, width=0.84, height=0.8))

    def layout(self) -> PackLayout:
        x0, y0, x1, y1 = self.frame
        return PackLayout(
            id="station-7x4",
            display_name="Simulated station · 7 × 4",
            rows=self.rows,
            columns=self.columns,
            row_labels=["Morning", "Midday", "Evening", "Bedtime"],
            column_labels=[f"Day {d}" for d in range(1, 8)],
            width_mm=(x1 - x0) / 5,
            height_mm=(y1 - y0) / 5,
            grid_region=self.grid,
            border_band=0.1,
            is_calibrated=True,
        )

    def render(self, scene: Scene, seed: int = 0, noise: float = 3.0, blur: float = 0.6) -> np.ndarray:
        rng = np.random.default_rng(seed)
        image = np.full((self.height, self.width, 3), (38, 36, 34), np.uint8)
        x0, y0, x1, y1 = self.frame
        fw, fh = x1 - x0, y1 - y0
        # Pack card and blister wells.
        card = (x0 + int(0.03 * fw), y0 + int(0.04 * fh), x1 - int(0.03 * fw), y1 - int(0.04 * fh))
        cv2.rectangle(image, card[:2], card[2:], (222, 226, 228), cv2.FILLED)
        layout = self.layout()
        for r in range(self.rows):
            for c in range(self.columns):
                rect = layout.cell_rect(cell(r, c))
                a = (int(x0 + (rect.x + 0.05 * rect.width) * fw), int(y0 + (rect.y + 0.05 * rect.height) * fh))
                b = (int(x0 + (rect.max_x - 0.05 * rect.width) * fw), int(y0 + (rect.max_y - 0.05 * rect.height) * fh))
                cv2.rectangle(image, a, b, (200, 204, 206), 2)
        # Pills.
        for index, medications in scene.contents.items():
            rect = layout.cell_rect(index)
            cx = x0 + rect.center.x * fw
            cy = y0 + rect.center.y * fh
            n = len(medications)
            for k, medication in enumerate(medications):
                colour, axes, shape = PALETTE[medication]
                offset = (k - (n - 1) / 2) * rect.width * fw * 0.36
                centre = (int(cx + offset + rng.uniform(-3, 3)), int(cy + rng.uniform(-6, 6)))
                angle = float(rng.uniform(0, 180))
                shade = tuple(int(v) for v in np.clip(np.array(colour) * 0.75, 0, 255))
                cv2.ellipse(image, centre, (axes[0] + 2, axes[1] + 2), angle, 0, 360, shade, cv2.FILLED)
                cv2.ellipse(image, centre, axes, angle, 0, 360, colour, cv2.FILLED)
        for index in scene.foreign:
            rect = layout.cell_rect(index)
            p = (int(x0 + rect.center.x * fw), int(y0 + (rect.y + 0.3 * rect.height) * fh))
            cv2.line(image, p, (p[0] + 25, p[1] + 8), (40, 40, 160), 3)
        if scene.card:
            code = code_image(scene.card, scale=5)
            top, left = 400, 40
            cv2.rectangle(
                image, (left - 20, top - 20), (left + code.shape[1] + 20, top + code.shape[0] + 50), (250,) * 3, -1
            )
            image[top : top + code.shape[0], left : left + code.shape[1]] = code[..., None]
        # Tray markers.
        dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICTIONARY)
        centres = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for marker_id, (mx, my) in zip(MARKER_IDS, centres, strict=True):
            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, self.marker_size)
            half = self.marker_size // 2
            pad = 14
            cv2.rectangle(
                image,
                (mx - half - pad, my - half - pad),
                (mx + half + pad, my + half + pad),
                (255, 255, 255),
                cv2.FILLED,
            )
            image[my - half : my - half + self.marker_size, mx - half : mx - half + self.marker_size] = marker[
                ..., None
            ]
        if blur:
            image = cv2.GaussianBlur(image, (0, 0), blur)
        if noise:
            image = np.clip(image + rng.normal(0, noise, image.shape), 0, 255).astype(np.uint8)
        return image


def full_scene(layout: PackLayout, items: list[str]) -> Scene:
    return Scene(contents={i: list(items) for i in layout.all_compartments})
