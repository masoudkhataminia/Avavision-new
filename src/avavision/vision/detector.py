"""AvaVision's own trained detector (RF-DETR, Apache-2.0) exported to ONNX.

Expected ONNX interface (RF-DETR export): input ``input`` (1,3,S,S) ImageNet-normalized RGB; outputs ``dets``
(1,Q,4) boxes as normalized (cx, cy, w, h) and ``labels`` (1,Q,C) class logits. The class names come
from the model manifest (``classes``), and their meanings from its ``labels`` map.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..core.geometry import Rect
from ..core.observation import Detection
from .embedder import IMAGENET_MEAN, IMAGENET_STD
from .runtime import create_session


class OnnxDetector:
    def __init__(self, path: Path | str, classes: list[str], input_size: int | None = None, noise_floor: float = 0.25):
        self.session = create_session(path)
        self.classes = classes
        model_input = self.session.get_inputs()[0]
        side = model_input.shape[-1] if model_input.shape else None
        self.input_size = input_size or (side if isinstance(side, int) else 560)
        self.noise_floor = noise_floor
        self.input_name = model_input.name

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Boxes in normalized image coordinates. The image is stretched to the square input, as RF-DETR
        is trained, so normalized coordinates map straight back."""
        rgb = cv2.cvtColor(
            cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
        )
        x = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        boxes, logits = self.session.run(["dets", "labels"], {self.input_name: x.astype(np.float32)})
        return decode(boxes[0], logits[0], self.classes, self.noise_floor)


def decode(boxes: np.ndarray, logits: np.ndarray, classes: list[str], noise_floor: float) -> list[Detection]:
    scores = 1 / (1 + np.exp(-logits))
    best = scores.argmax(axis=1)
    confidence = scores[np.arange(len(best)), best]
    detections = []
    for (cx, cy, w, h), cls, conf in zip(boxes, best, confidence, strict=True):
        if conf < noise_floor or cls >= len(classes):
            continue
        detections.append(
            Detection(
                label=classes[cls],
                confidence=float(conf),
                box=Rect(x=float(cx - w / 2), y=float(cy - h / 2), width=float(w), height=float(h)),
            )
        )
    return detections
