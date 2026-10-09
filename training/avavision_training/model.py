"""AvaVision's appearance embedder: a DINOv2 backbone (Apache-2.0 weights) with a projection head."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from .data import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

#: Apache-2.0 licensed DINOv2 ViT-S/14 with registers. Do not swap in DINOv3 (custom licence) or
#: research-only weights without a licence review.
DEFAULT_BACKBONE = "facebook/dinov2-with-registers-small"
BACKBONE_LICENSE = "Apache-2.0 (facebook/dinov2-with-registers-small)"


class PillEmbedder(nn.Module):
    """Image → L2-normalized embedding. Without a head it returns the backbone's CLS feature."""

    def __init__(self, backbone: str = DEFAULT_BACKBONE, embedding_size: int | None = 256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, embedding_size) if embedding_size else nn.Identity()
        self.embedding_size = embedding_size or hidden

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values=pixels).pooler_output
        return nn.functional.normalize(self.head(features), dim=-1)

    def freeze_backbone(self, trainable_blocks: int = 0) -> None:
        """Freezes the backbone except its last ``trainable_blocks`` transformer blocks."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        layers = self.backbone.encoder.layer
        for block in layers[len(layers) - trainable_blocks :] if trainable_blocks else []:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.layernorm.parameters():
            parameter.requires_grad = trainable_blocks > 0


class ExportWrapper(nn.Module):
    """What ships in the app: takes an RGB image with values 0–255 and does all preprocessing itself,
    so the app passes Vision's pixel buffer straight in."""

    def __init__(self, embedder: PillEmbedder):
        super().__init__()
        self.embedder = embedder.eval()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255)
        _freeze_position_embeddings(self.embedder.backbone, IMAGE_SIZE)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.embedder((image - self.mean) / self.std)


def _freeze_position_embeddings(backbone: nn.Module, size: int) -> None:
    """DINOv2 interpolates its position embeddings bicubically for every input size, which Core ML
    cannot convert. The app always uses ``size``×``size``, so the interpolation is done once here."""
    embeddings = backbone.embeddings
    patches = (size // backbone.config.patch_size) ** 2
    with torch.no_grad():
        dummy = torch.zeros(1, 1 + patches, backbone.config.hidden_size)
        fixed = embeddings.interpolate_pos_encoding(dummy, size, size).detach().clone()
    embeddings.register_buffer("fixed_position_embeddings", fixed)
    embeddings.interpolate_pos_encoding = lambda _embeddings, _height, _width: embeddings.fixed_position_embeddings
