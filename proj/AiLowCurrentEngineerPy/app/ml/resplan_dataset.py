from __future__ import annotations

import os
from dataclasses import dataclass
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


@dataclass
class AugmentConfig:
    hflip: bool = True
    vflip: bool = True
    rotate90: bool = True


class ResPlanRasterDataset(Dataset):
    """
    Ожидает структуру:
      data/resplan_raster/
        images/*.png   (RGB)
        masks/*.png    (L, labels 0..N-1)
    """

    def __init__(self, root: str, augment: bool = True, aug_cfg: AugmentConfig | None = None):
        self.root = root
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "masks")

        self.ids: List[str] = []
        for p in sorted(glob(os.path.join(self.images_dir, "*.png"))):
            base = os.path.splitext(os.path.basename(p))[0]
            mp = os.path.join(self.masks_dir, f"{base}.png")
            if os.path.exists(mp):
                self.ids.append(base)

        if not self.ids:
            raise RuntimeError(f"No samples found in {self.images_dir} (and matching masks in {self.masks_dir}).")

        self.augment = bool(augment)
        self.aug_cfg = aug_cfg or AugmentConfig()

    def __len__(self) -> int:
        return len(self.ids)

    def _augment(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # img: [3,H,W], mask: [H,W]
        if not self.augment:
            return img, mask

        # flips
        if self.aug_cfg.hflip and torch.rand(()) < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)

        if self.aug_cfg.vflip and torch.rand(()) < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)

        # rotate 0/90/180/270
        if self.aug_cfg.rotate90:
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img = torch.rot90(img, k, dims=(1, 2))
                mask = torch.rot90(mask, k, dims=(0, 1))

        return img, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sid = self.ids[idx]
        ip = os.path.join(self.images_dir, f"{sid}.png")
        mp = os.path.join(self.masks_dir, f"{sid}.png")

        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")

        img_t = TF.to_tensor(img)  # float32 [0..1], [3,H,W]
        mask_np = np.array(mask, dtype=np.uint8)
        mask_t = torch.from_numpy(mask_np).long()  # [H,W]

        img_t, mask_t = self._augment(img_t, mask_t)
        return img_t, mask_t
