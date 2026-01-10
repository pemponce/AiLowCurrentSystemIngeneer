from __future__ import annotations

import torch.nn as nn
import torchvision.models.segmentation as seg


def build_structure_model(num_classes: int):
    """
    Базовый сегментатор: DeepLabV3-ResNet50.
    Выдаёт logits: [B, num_classes, H, W]
    """
    model = seg.deeplabv3_resnet50(weights=None, weights_backbone="DEFAULT")
    # classifier: DeepLabHead(2048 -> 256 -> num_classes)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model
