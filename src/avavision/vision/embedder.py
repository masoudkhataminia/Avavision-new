"""Appearance embedders: turn pill crops into vectors the brain can remember and compare."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .runtime import create_session

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class Embedder(Protocol):
    id: str
    title: str

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """BGR uint8 crops → (n, d) L2-normalized float32."""
        ...


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.maximum(norms, 1e-12)).astype(np.float32)


def square_fit(crop: np.ndarray, size: int, fill: int = 0) -> np.ndarray:
    """Letterboxes a crop into a size×size square without distorting the pill's shape."""
    h, w = crop.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), fill, dtype=np.uint8)
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    out[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return out


class OnnxEmbedder:
    """DINOv2-family embedder in ONNX.

    Two model formats are understood:
    - AvaVision's own export (``training/export_onnx.py``): input ``image`` (N,3,224,224) RGB 0–255,
      output ``embedding`` (already normalized).
    - Hugging Face DINOv2 (``onnx-community/dinov2-with-registers-small``, Apache-2.0 weights): input
      ``pixel_values`` (ImageNet-normalized), output ``last_hidden_state``; the embedding is the CLS token
      concatenated with the mean patch token, as in DINOv2's own linear evaluation.
    """

    def __init__(self, path: Path | str, embedder_id: str, title: str, input_size: int = 224, batch: int = 64):
        self.session = create_session(path)
        self.id = embedder_id
        self.title = title
        self.input_size = input_size
        self.batch = batch
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.registers = 4

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 0), dtype=np.float32)
        outputs = []
        for start in range(0, len(crops), self.batch):
            chunk = crops[start : start + self.batch]
            rgb = np.stack([cv2.cvtColor(square_fit(c, self.input_size), cv2.COLOR_BGR2RGB) for c in chunk])
            x = rgb.transpose(0, 3, 1, 2).astype(np.float32)
            if self.input_name == "image":
                outputs.append(self.session.run(["embedding"], {"image": x})[0])
            else:
                x = (x / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                hidden = self.session.run([self.output_names[0]], {self.input_name: x})[0]
                cls = hidden[:, 0]
                patches = hidden[:, 1 + self.registers :].mean(axis=1)
                outputs.append(np.concatenate([cls, patches], axis=1))
        return _normalize_rows(np.concatenate(outputs))


class ClassicEmbedder:
    """Training-free fallback: colour (Lab histogram), size and shape. Weak at telling similar white tablets
    apart, which the brain's calibration measures and accounts for, but it needs no model download."""

    id = "avavision-classic-v1"
    title = "Classic colour and shape descriptor (built in)"

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalize_rows(np.stack([self._describe(c) for c in crops]))

    @staticmethod
    def _describe(crop: np.ndarray) -> np.ndarray:
        square = square_fit(crop, 96)
        lab = cv2.cvtColor(square, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        border = np.concatenate([gray[:4].ravel(), gray[-4:].ravel(), gray[:, :4].ravel(), gray[:, -4:].ravel()])
        mask = (np.abs(gray.astype(int) - int(np.median(border))) > 20).astype(np.uint8)
        if mask.sum() < 20:
            mask[:] = 1
        hist = cv2.calcHist([lab], [0, 1, 2], mask, [6, 6, 6], [0, 256, 0, 256, 0, 256]).ravel()
        hist /= max(hist.sum(), 1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape = np.zeros(4, dtype=np.float32)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            (_, _), (w, h), _ = cv2.minAreaRect(c)
            shape = np.array(
                [
                    area / (96 * 96),
                    4 * np.pi * area / max(perimeter**2, 1),
                    min(w, h) / max(w, h, 1),
                    *cv2.HuMoments(cv2.moments(c)).ravel()[:1],
                ],
                dtype=np.float32,
            )
        return np.concatenate([np.sqrt(hist), shape * 2]).astype(np.float32)
