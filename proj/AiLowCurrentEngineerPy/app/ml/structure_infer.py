from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import Dict, Tuple, List, Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from app.ml.model_structure import build_structure_model


DEFAULT_CLASS_NAMES = ["bg", "wall", "door", "window", "front_door"]

PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),         # bg
    1: (0, 0, 255),       # wall - red
    2: (0, 255, 255),     # door - yellow
    3: (255, 255, 0),     # window - cyan
    4: (255, 0, 255),     # front_door - magenta
}


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


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
    """
    Supports BOTH formats:
      1) checkpoint dict: {"model": state_dict, "num_classes": int, ...}
      2) raw state_dict (what torch.save(model.state_dict()) produces)
    """
    ckpt: Any = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        num_classes = int(ckpt.get("num_classes", len(DEFAULT_CLASS_NAMES)))
        class_names = ckpt.get("class_names", DEFAULT_CLASS_NAMES[:num_classes])
    else:
        state = ckpt
        num_classes = len(DEFAULT_CLASS_NAMES)
        class_names = DEFAULT_CLASS_NAMES

    model = build_structure_model(num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, num_classes, list(class_names)


def _resize_max_side(img_bgr: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_bgr
    scale = max_side / float(m)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def _preprocess(img_bgr: np.ndarray, mode: str, invert: bool) -> np.ndarray:
    if invert:
        img_bgr = 255 - img_bgr

    mode = (mode or "none").strip().lower()
    if mode == "none":
        return img_bgr

    if mode == "binarize":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thr = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            5,
        )
        out = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
        return out

    raise ValueError(f"Unknown --preprocess mode: {mode}")


@torch.no_grad()
def infer_one(model, img_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img_rgb).convert("RGB")
    x = TF.to_tensor(image_pil).unsqueeze(0).to(device)  # [1,3,H,W]
    out = model(x)["out"]
    pred = out.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out/structure_infer")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--preprocess", default="binarize", choices=["none", "binarize"])
    ap.add_argument("--invert", action="store_true")
    args = ap.parse_args()

    _ensure_dir(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, num_classes, class_names = _load_checkpoint(args.ckpt, device)

    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    orig_h, orig_w = img_bgr.shape[:2]
    img_bgr = _resize_max_side(img_bgr, int(args.max_side))
    img_bgr = _preprocess(img_bgr, mode=str(args.preprocess), invert=bool(args.invert))

    pred = infer_one(model, img_bgr, device)

    pred_mask_path = osp.join(args.out, "pred_mask.png")
    pred_color_path = osp.join(args.out, "pred_color.png")
    overlay_path = osp.join(args.out, "overlay.png")
    preproc_path = osp.join(args.out, "input_preprocessed.png")

    cv2.imwrite(preproc_path, img_bgr)
    cv2.imwrite(pred_mask_path, pred)

    pred_color = _colorize_mask(pred)
    cv2.imwrite(pred_color_path, pred_color)

    overlay = _overlay(img_bgr, pred_color, alpha=float(args.alpha))
    cv2.imwrite(overlay_path, overlay)

    uniq = np.unique(pred).tolist()
    used = [class_names[i] if i < len(class_names) else str(i) for i in uniq]

    total = float(pred.size)
    cov_items: List[Tuple[str, float]] = []
    for cid in uniq:
        name = class_names[cid] if cid < len(class_names) else str(cid)
        cov = float((pred == cid).sum()) / total
        cov_items.append((name, cov))

    print("Saved:")
    print(f"  {preproc_path}")
    print(f"  {pred_mask_path}")
    print(f"  {pred_color_path}")
    print(f"  {overlay_path}")
    print(f"Input: {orig_w}x{orig_h}  -> used for infer: {img_bgr.shape[1]}x{img_bgr.shape[0]}")
    print(f"Classes in prediction: {uniq} -> {used} (num_classes={num_classes})")
    print("Coverage:")
    for k, v in cov_items:
        print(f"  {k:12s}: {v*100:.3f}%")

if __name__ == "__main__":
    main()
