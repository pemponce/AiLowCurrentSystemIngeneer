from __future__ import annotations

import argparse
import os
import os.path as osp
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from app.ml.model_structure import build_structure_model


CLASS_NAMES = ["bg", "wall", "door", "window", "front_door"]

PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
}


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _list_ids(images_dir: str, masks_dir: str) -> List[str]:
    ids = []
    for fn in os.listdir(images_dir):
        if not fn.lower().endswith(".png"):
            continue
        sid = os.path.splitext(fn)[0]
        mp = osp.join(masks_dir, sid + ".png")
        if osp.exists(mp):
            ids.append(sid)
    ids.sort()
    return ids


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, bgr in PALETTE.items():
        out[mask == cls_id] = bgr
    return out


def _overlay(img_bgr: np.ndarray, color_mask_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    alpha = float(max(0.0, min(1.0, alpha)))
    return cv2.addWeighted(img_bgr, 1.0 - alpha, color_mask_bgr, alpha, 0)


def _load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = int(ckpt.get("num_classes", len(CLASS_NAMES)))
    model = build_structure_model(num_classes=num_classes)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, num_classes


@torch.no_grad()
def _infer(model, img_pil: Image.Image, device: torch.device) -> np.ndarray:
    x = TF.to_tensor(img_pil).unsqueeze(0).to(device)
    out = model(x)["out"]
    pred = out.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred


def _iou_per_class(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> List[float]:
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        g = (gt == c)
        inter = int(np.logical_and(p, g).sum())
        union = int(np.logical_or(p, g).sum())
        if union == 0:
            ious.append(float("nan"))  # class not present in both
        else:
            ious.append(inter / union)
    return ious


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="models/structure_resplan.pt")
    ap.add_argument("--data", required=True, help="data/resplan_raster")
    ap.add_argument("--out", default="out/structure_eval", help="output folder")
    ap.add_argument("--n", type=int, default=12, help="how many samples to visualize")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    images_dir = osp.join(args.data, "images")
    masks_dir = osp.join(args.data, "masks")

    _ensure_dir(args.out)

    ids = _list_ids(images_dir, masks_dir)
    if not ids:
        raise RuntimeError(f"No samples in {images_dir} / {masks_dir}")

    random.seed(int(args.seed))
    pick = random.sample(ids, k=min(int(args.n), len(ids)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, num_classes = _load_checkpoint(args.ckpt, device)

    all_ious: List[List[float]] = []

    for i, sid in enumerate(pick, start=1):
        img_path = osp.join(images_dir, sid + ".png")
        gt_path = osp.join(masks_dir, sid + ".png")
        out_dir = osp.join(args.out, f"{i:03d}_{sid}")
        _ensure_dir(out_dir)

        img_pil = Image.open(img_path).convert("RGB")
        gt = np.array(Image.open(gt_path).convert("L"), dtype=np.uint8)

        pred = _infer(model, img_pil, device)

        # metrics
        ious = _iou_per_class(pred, gt, num_classes)
        all_ious.append(ious)

        # save visuals
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        gt_color = _colorize_mask(gt)
        pred_color = _colorize_mask(pred)
        overlay = _overlay(img_bgr, pred_color, alpha=float(args.alpha))

        cv2.imwrite(osp.join(out_dir, "input.png"), img_bgr)
        cv2.imwrite(osp.join(out_dir, "gt_color.png"), gt_color)
        cv2.imwrite(osp.join(out_dir, "pred_color.png"), pred_color)
        cv2.imwrite(osp.join(out_dir, "overlay_pred.png"), overlay)
        cv2.imwrite(osp.join(out_dir, "pred_mask.png"), pred)

    # print aggregate IoU
    arr = np.array(all_ious, dtype=np.float32)  # [N,C]
    mean_per_class = np.nanmean(arr, axis=0)
    miou = float(np.nanmean(mean_per_class))

    print("IoU per class:")
    for c in range(num_classes):
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
        v = mean_per_class[c]
        if np.isnan(v):
            print(f"  {c:02d} {name:12s}: n/a (absent)")
        else:
            print(f"  {c:02d} {name:12s}: {float(v):.4f}")

    print(f"mIoU: {miou:.4f}")
    print(f"Saved визуализации в: {args.out}")


if __name__ == "__main__":
    main()
