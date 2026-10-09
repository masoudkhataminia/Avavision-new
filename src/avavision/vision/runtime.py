"""ONNX Runtime sessions on the fastest hardware available.

Install the matching package on the station: ``onnxruntime-gpu`` (NVIDIA: TensorRT/CUDA),
``onnxruntime-directml`` (any DirectX 12 GPU, e.g. Surface or AMD/Intel), ``onnxruntime-qnn``
(Snapdragon NPU) or plain ``onnxruntime`` (CPU).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import onnxruntime as ort

#: Fastest first. TensorRT and CUDA need NVIDIA; DirectML runs on any Windows GPU; QNN on Snapdragon.
PREFERRED_PROVIDERS = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "QNNExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]


def available_providers() -> list[str]:
    available = set(ort.get_available_providers())
    return [p for p in PREFERRED_PROVIDERS if p in available]


def create_session(path: Path | str, providers: list[str] | None = None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    chosen = providers or available_providers()
    return ort.InferenceSession(str(path), sess_options=options, providers=chosen)


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
