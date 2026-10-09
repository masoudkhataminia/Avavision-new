"""Colour signatures of compartments, for comparing compartments that should hold the same tablets (core.peers).

A signature is how much of a pocket is covered by each hue, weighted by how colourful the pixels are, smoothed
over neighbouring hues so lighting shifts do not jump between bins. Position does not matter, so tablets may lie
anywhere in their pocket. The card's own colour (seen between and through the pockets) is left out.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel

from ..core.models import CompartmentIndex, PackLayout


class SignaturePolicy(BaseModel):
    #: Part of each cell on every side left out (pocket walls and the card around them).
    inset: float = 0.12
    #: Pixels below ``minimum_chroma`` count for nothing, above ``minimum_chroma + chroma_span`` fully.
    minimum_chroma: float = 6
    chroma_span: float = 15
    hue_sigma: float = 10
    #: Hues this close (degrees) to the card's colour are left out.
    card_hue_span: float = 40
    #: Signature resolution in degrees.
    step: int = 5
    analysis_scale: float = 0.5


def _hue_chroma(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    return (np.degrees(np.arctan2(b, a)) + 360) % 360, np.hypot(a, b)


def _smoothed(hue: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Per-degree hue density (circular), smoothed with a normalized Gaussian."""
    histogram = np.bincount(hue.astype(int).ravel() % 360, weights=weight.ravel(), minlength=360)[:360]
    offsets = np.arange(-180, 180)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    spectrum = np.fft.ifft(np.fft.fft(histogram) * np.fft.fft(np.roll(kernel, -180))).real
    return np.maximum(spectrum, 0)


def card_hue(canvas: np.ndarray, layout: PackLayout, policy: SignaturePolicy | None = None) -> float | None:
    """The card's dominant colour outside the compartments, or ``None`` for a grey or white card."""
    policy = policy or SignaturePolicy()
    h, w = canvas.shape[:2]
    outside = np.ones((h, w), bool)
    for index in layout.all_compartments:
        r = layout.cell_rect(index)
        outside[int(r.y * h) : int(np.ceil(r.max_y * h)), int(r.x * w) : int(np.ceil(r.max_x * w))] = False
    hue, chroma = _hue_chroma(canvas)
    colourful = outside & (chroma > policy.minimum_chroma + policy.chroma_span)
    if colourful.sum() < 0.01 * outside.sum():
        return None
    return float(np.argmax(_smoothed(hue[colourful], np.ones(int(colourful.sum())), policy.hue_sigma)))


def colour_signature(pocket: np.ndarray, skip_hue: float | None, policy: SignaturePolicy | None = None) -> list[float]:
    policy = policy or SignaturePolicy()
    if policy.analysis_scale != 1:
        pocket = cv2.resize(
            pocket, None, fx=policy.analysis_scale, fy=policy.analysis_scale, interpolation=cv2.INTER_AREA
        )
    hue, chroma = _hue_chroma(pocket)
    weight = np.clip((chroma - policy.minimum_chroma) / policy.chroma_span, 0, 1)
    if skip_hue is not None:
        weight = weight * (np.abs((hue - skip_hue + 180) % 360 - 180) > policy.card_hue_span)
    density = _smoothed(hue, weight, policy.hue_sigma) / max(hue.size, 1)
    return [float(v) * policy.step for v in density[:: policy.step]]


def compartment_signatures(
    canvas: np.ndarray,
    layout: PackLayout,
    skip: set[CompartmentIndex] | None = None,
    policy: SignaturePolicy | None = None,
) -> dict[CompartmentIndex, list[float]]:
    """Signatures of every compartment of the perspective-corrected pack, except those in ``skip``."""
    policy = policy or SignaturePolicy()
    skip_hue = card_hue(canvas, layout, policy)
    h, w = canvas.shape[:2]
    result = {}
    for index in layout.all_compartments:
        if skip and index in skip:
            continue
        r = layout.cell_rect(index)
        dx, dy = r.width * policy.inset, r.height * policy.inset
        pocket = canvas[int((r.y + dy) * h) : int((r.max_y - dy) * h), int((r.x + dx) * w) : int((r.max_x - dx) * w)]
        if pocket.size:
            result[index] = colour_signature(pocket, skip_hue, policy)
    return result
