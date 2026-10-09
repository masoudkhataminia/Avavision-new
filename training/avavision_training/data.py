"""Reads the training data exported by the app (Brain › Export training data).

Layout of an export folder::

    labels.jsonl          one JSON object per pill: crop, medicationID, source, groupID, createdAt
    crops/<uuid>.jpg      single-pill crops

Pills from the same photo or check share a ``groupID``. Splits always keep a whole group on one
side so validation never sees near-identical views of a training pill.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    medication: str
    group: str


def load_samples(export_dir: str | Path) -> list[Sample]:
    root = Path(export_dir)
    samples = []
    with open(root / "labels.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            path = root / row["crop"]
            if path.exists():
                samples.append(Sample(path=path, medication=row["medicationID"], group=row["groupID"]))
    return samples


@dataclass(frozen=True)
class Split:
    train: list[Sample]
    validation: list[Sample]
    #: Medications never seen in training, to measure rejection of unknown pills.
    unseen: list[Sample]


def split(samples: list[Sample], validation_fraction: float = 0.2, unseen_medications: int = 1, seed: int = 0) -> Split:
    """Group-aware split. ``unseen_medications`` medications, chosen at random, are held out entirely."""
    rng = random.Random(seed)
    medications = sorted({s.medication for s in samples})
    held_out = set(rng.sample(medications, min(unseen_medications, max(0, len(medications) - 2))))
    unseen = [s for s in samples if s.medication in held_out]
    known = [s for s in samples if s.medication not in held_out]

    train, validation = [], []
    for medication in sorted({s.medication for s in known}):
        groups = sorted({s.group for s in known if s.medication == medication})
        rng.shuffle(groups)
        count = max(1, round(len(groups) * validation_fraction)) if len(groups) > 1 else 0
        validation_groups = set(groups[:count])
        for sample in known:
            if sample.medication == medication:
                (validation if sample.group in validation_groups else train).append(sample)
    return Split(train=train, validation=validation, unseen=unseen)


def train_transform() -> transforms.Compose:
    # Station lighting is controlled, so colour jitter stays mild: colour is real evidence.
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    # Matches the app: the crop is fitted (not stretched) into a square before embedding.
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class PillDataset(Dataset):
    def __init__(self, samples: list[Sample], classes: list[str], transform: transforms.Compose):
        self.samples = samples
        self.index = {name: i for i, name in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[item]
        image = Image.open(sample.path).convert("RGB")
        return self.transform(image), self.index.get(sample.medication, -1)
