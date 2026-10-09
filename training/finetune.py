"""Fine-tunes AvaVision's appearance embedder on pharmacist-confirmed pill crops.

    python finetune.py --data AvaVision-training-data --out runs/embedder-v1

Metric learning (Sub-center ArcFace) on a DINOv2 backbone. Validation uses whole photos/checks the
model never trained on, plus medications it never saw, and reports the same quantities the app's
brain relies on: nearest-neighbour accuracy and how well unknown pills are rejected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader

from avavision_training.data import PillDataset, eval_transform, load_samples, split, train_transform
from avavision_training.model import DEFAULT_BACKBONE, PillEmbedder


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def embed(model: PillEmbedder, dataset: PillDataset, batch_size: int, target: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    vectors, labels = [], []
    for images, targets in DataLoader(dataset, batch_size=batch_size):
        vectors.append(model(images.to(target)).cpu())
        labels.append(targets)
    if not vectors:
        return torch.empty(0, model.embedding_size), torch.empty(0, dtype=torch.long)
    return torch.cat(vectors), torch.cat(labels)


def evaluate(gallery: tuple[torch.Tensor, torch.Tensor], queries: tuple[torch.Tensor, torch.Tensor], unseen: torch.Tensor) -> dict:
    """1-NN accuracy on known medications, and the best acceptance threshold that keeps every unseen pill
    rejected while naming as many known pills correctly as possible (the app calibrates the same way)."""
    gallery_vectors, gallery_labels = gallery
    query_vectors, query_labels = queries
    if len(gallery_vectors) == 0 or len(query_vectors) == 0:
        return {"queries": 0}
    similarity = query_vectors @ gallery_vectors.T
    best, index = similarity.max(dim=1)
    predicted = gallery_labels[index]
    correct = predicted == query_labels
    report = {"queries": len(query_vectors), "nearest_neighbour_accuracy": correct.float().mean().item()}
    if len(unseen):
        unseen_best = (unseen @ gallery_vectors.T).max(dim=1).values
        threshold = unseen_best.max().item() + 1e-4
        named = best >= threshold
        report["unseen_pills"] = len(unseen)
        report["threshold_rejecting_all_unseen"] = threshold
        report["coverage_at_that_threshold"] = named.float().mean().item()
        report["precision_at_that_threshold"] = (correct & named).sum().item() / max(1, named.sum().item())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Folder exported by the app")
    parser.add_argument("--out", required=True, help="Output folder for the checkpoint and report")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trainable-blocks", type=int, default=2, help="Last backbone blocks to fine-tune")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unseen-medications", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    target = device()
    samples = load_samples(args.data)
    parts = split(samples, unseen_medications=args.unseen_medications, seed=args.seed)
    classes = sorted({s.medication for s in parts.train})
    if len(classes) < 2:
        raise SystemExit("Need pills of at least two medications in the training split.")
    print(f"{len(parts.train)} train, {len(parts.validation)} validation, {len(parts.unseen)} unseen; {len(classes)} medications")

    model = PillEmbedder(args.backbone, args.embedding_size).to(target)
    model.freeze_backbone(args.trainable_blocks)
    criterion = losses.SubCenterArcFaceLoss(num_classes=len(classes), embedding_size=model.embedding_size).to(target)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
            {"params": criterion.parameters(), "lr": args.lr * 10},
        ],
        weight_decay=1e-4,
    )
    loader = DataLoader(
        PillDataset(parts.train, classes, train_transform()), batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    gallery_set = PillDataset(parts.train, classes, eval_transform())
    validation_set = PillDataset(parts.validation, classes, eval_transform())
    unseen_set = PillDataset(parts.unseen, classes, eval_transform())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, labels in loader:
            loss = criterion(model(images.to(target)), labels.to(target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
        report = evaluate(
            embed(model, gallery_set, args.batch_size, target),
            embed(model, validation_set, args.batch_size, target),
            embed(model, unseen_set, args.batch_size, target)[0],
        )
        report.update(epoch=epoch, loss=total / max(1, len(parts.train)))
        history.append(report)
        print(json.dumps(report))
        accuracy = report.get("nearest_neighbour_accuracy", 0.0)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {"state_dict": model.state_dict(), "backbone": args.backbone, "embedding_size": args.embedding_size,
                 "classes": classes, "report": report},
                out / "embedder.pt",
            )
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best validation accuracy {best_accuracy:.4f}; checkpoint at {out / 'embedder.pt'}")


if __name__ == "__main__":
    main()
