# app/ml/train_structure_rf.py
import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.segmentation import deeplabv3_resnet50

try:
    import cv2
except Exception as e:
    raise RuntimeError("opencv-python is required (cv2)") from e

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


CLASS_NAMES = ["bg", "wall", "door", "window", "front_door"]  # ids: 0..4


@dataclass
class TrainCfg:
    data: str
    out: str
    epochs: int
    batch: int
    lr: float
    val_split: float
    num_workers: int
    seed: int
    num_classes: int

    auto_class_weights: bool
    focal_gamma: float
    dice_weight: float

    split_by_plan: bool
    balance_sampler: bool

    # IMPORTANT: training/val preprocess must match infer, otherwise "val_iou good but real looks bad"
    preprocess: str
    invert: bool
    # optional domain-rand: apply infer-like binarize with this probability (in addition to preprocess)
    binarize_prob: float

    drop_last: bool
    amp: bool
    tf32: bool
    cudnn_benchmark: bool

    # DataLoader perf knobs
    persistent_workers: bool
    prefetch_factor: int

    # BatchNorm stability
    freeze_bn: bool

    early_stop_patience: int
    early_stop_min_delta: float

    # "real" holdout evaluator (no labels needed)
    real_dir: str
    real_out: str
    real_max_side: int
    real_alpha: float
    real_minor_target: float
    real_minor_tol: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int) -> nn.Module:
    # CAD-like domain; weights=None as in your setup.
    return deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _atomic_torch_save(obj: dict, path: str) -> None:
    # Avoid partially-written checkpoints.
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def freeze_batchnorm_(model: nn.Module) -> None:
    """
    Freeze all BatchNorm layers:
      - set eval() so they don't update running stats
      - disable grads for affine params
    Useful when batch is small / last batch / ASPP 1x1 problems.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters(recurse=False):
                p.requires_grad = False


def scan_pairs(dataset_root: str) -> List[Tuple[str, str]]:
    images_dir = os.path.join(dataset_root, "images")
    masks_dir = os.path.join(dataset_root, "masks")
    if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
        raise RuntimeError(f"Expected {images_dir} and {masks_dir}")

    out: List[Tuple[str, str]] = []
    for fn in os.listdir(images_dir):
        if not fn.lower().endswith(".png"):
            continue
        img_path = os.path.join(images_dir, fn)
        mask_path = os.path.join(masks_dir, fn)
        if not os.path.exists(mask_path):
            continue
        if "_overlay" in fn.lower():
            continue
        out.append((img_path, mask_path))

    out.sort(key=lambda x: os.path.basename(x[0]))
    if not out:
        raise RuntimeError(f"No image/mask pairs found in {dataset_root}")
    return out


def plan_id_from_filename(path: str) -> str:
    """
    Our preprocess names tiles like: plan_001_0004.png
    plan_id = everything before last '_dddd'
    """
    base = os.path.basename(path)
    name = base[:-4] if base.lower().endswith(".png") else base
    if len(name) >= 5 and name[-5] == "_" and name[-4:].isdigit():
        return name[:-5]
    return name


def split_pairs(
    pairs: List[Tuple[str, str]],
    val_split: float,
    seed: int,
    split_by_plan: bool,
) -> Tuple[List[int], List[int]]:
    n = len(pairs)
    idxs = list(range(n))
    rng = random.Random(seed)

    if not split_by_plan:
        rng.shuffle(idxs)
        n_val = max(1, int(n * val_split))
        val_idx = idxs[:n_val]
        tr_idx = idxs[n_val:]
        return tr_idx, val_idx

    # group by plan_id
    groups: Dict[str, List[int]] = {}
    for i, (ip, _) in enumerate(pairs):
        pid = plan_id_from_filename(ip)
        groups.setdefault(pid, []).append(i)

    plan_ids = list(groups.keys())
    rng.shuffle(plan_ids)

    n_val_plans = max(1, int(len(plan_ids) * val_split))
    val_plans = set(plan_ids[:n_val_plans])

    tr_idx: List[int] = []
    val_idx: List[int] = []
    for pid, ids in groups.items():
        if pid in val_plans:
            val_idx.extend(ids)
        else:
            tr_idx.extend(ids)

    # fallback safety
    if not tr_idx or not val_idx:
        rng.shuffle(idxs)
        n_val = max(1, int(n * val_split))
        val_idx = idxs[:n_val]
        tr_idx = idxs[n_val:]
    return tr_idx, val_idx


def compute_pixel_hist_from_mask_paths(mask_paths: List[str], num_classes: int) -> np.ndarray:
    hs = np.zeros(num_classes, dtype=np.int64)
    for mp in mask_paths:
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        b = np.bincount(m.reshape(-1), minlength=num_classes)
        hs += b[:num_classes]
    return hs


def make_class_weights_from_hist(hist: np.ndarray) -> torch.Tensor:
    """
    Robust weights for heavy imbalance.
    - class absent => weight 0 (do not affect CE)
    """
    hist = hist.astype(np.float64)
    w = np.zeros_like(hist, dtype=np.float64)
    nonzero = hist > 0
    if nonzero.sum() == 0:
        return torch.ones_like(torch.tensor(hist, dtype=torch.float32))

    total = float(hist[nonzero].sum() + 1e-9)
    freq = hist[nonzero] / total
    freq = np.clip(freq, 1e-9, 1.0)

    inv = 1.0 / np.sqrt(freq)      # softer than 1/f
    inv = inv / float(inv.mean())  # normalize on nonzero only
    inv = np.clip(inv, 0.2, 10.0)

    w[nonzero] = inv
    w[~nonzero] = 0.0
    return torch.tensor(w, dtype=torch.float32)


def dice_loss_multiclass_ignore_empty(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
    ignore_background: bool = True,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)  # (N,C,H,W)
    t = torch.zeros_like(probs)
    t.scatter_(1, target.unsqueeze(1), 1.0)

    start_c = 1 if ignore_background else 0
    dices: List[torch.Tensor] = []

    for c in range(start_c, num_classes):
        pc = probs[:, c, :, :]
        tc = t[:, c, :, :]

        # ignore empty target class in this batch
        if float(tc.sum().item()) <= 0.0:
            continue

        inter = (pc * tc).sum(dim=(1, 2))
        denom = pc.sum(dim=(1, 2)) + tc.sum(dim=(1, 2)) + eps
        d = 1.0 - (2.0 * inter + eps) / denom
        dices.append(d.mean())

    if not dices:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(dices).mean()


def confusion_matrix_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    pred = torch.argmax(logits, dim=1).to(torch.int64)
    tgt = target.to(torch.int64)
    pred = pred.reshape(-1)
    tgt = tgt.reshape(-1)
    idx = (tgt * num_classes + pred).to(torch.int64).cpu()
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def iou_from_confusion(cm: torch.Tensor) -> List[float]:
    cmf = cm.to(torch.float64)
    tp = torch.diag(cmf)
    fp = cmf.sum(dim=0) - tp
    fn = cmf.sum(dim=1) - tp
    denom = tp + fp + fn
    out = []
    for c in range(len(tp)):
        if float(denom[c].item()) <= 0.0:
            out.append(float("nan"))
        else:
            out.append(float((tp[c] / denom[c]).item()))
    return out


def fmt_iou(iou: List[float], names: List[str]) -> str:
    parts = []
    for i, v in enumerate(iou):
        nm = names[i] if i < len(names) else f"c{i}"
        if v == v:
            parts.append(f"{nm}={v:.3f}")
        else:
            parts.append(f"{nm}=nan")
    return " ".join(parts)


def focal_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor],
    gamma: float,
) -> torch.Tensor:
    """
    Focal loss on top of CE:
      FL = (1-pt)^gamma * CE
    weight: per-class weights (like CE)
    """
    logp = torch.log_softmax(logits, dim=1)  # (N,C,H,W)
    tgt = target.unsqueeze(1)
    logpt = torch.gather(logp, 1, tgt).squeeze(1)  # (N,H,W)
    pt = torch.exp(logpt)

    ce = -logpt
    if weight is not None:
        w = weight[target]  # (N,H,W)
        ce = ce * w

    fl = ((1.0 - pt) ** gamma) * ce
    return fl.mean()


# -------------------------
# Preprocess (MUST MATCH infer)
# -------------------------

def preprocess_like_infer(img_bgr: np.ndarray, mode: str, invert: bool) -> np.ndarray:
    """
    Keep identical semantics to app/ml/structure_infer.py:
      - optional invert
      - mode: none | binarize (adaptiveThreshold)
    """
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

    raise ValueError(f"Unknown preprocess mode: {mode}")


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


# -------------------------
# Dataset
# -------------------------

class RasterPairsDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        num_classes: int,
        augment: bool,
        preprocess: str,
        invert: bool,
        binarize_prob: float,
        seed: int,
    ) -> None:
        self.pairs = pairs
        self.num_classes = int(num_classes)
        self.augment = bool(augment)
        self.preprocess = str(preprocess)
        self.invert = bool(invert)
        self.binarize_prob = float(binarize_prob)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # deterministic base-preprocess (match infer)
        img_bgr = preprocess_like_infer(img_bgr, mode=self.preprocess, invert=self.invert)

        # optional domain-rand: sometimes binarize even if preprocess=none
        if self.binarize_prob > 0 and self.rng.random() < self.binarize_prob:
            img_bgr = preprocess_like_infer(img_bgr, mode="binarize", invert=False)

        if self.augment:
            # flips (copy to avoid negative strides)
            if self.rng.random() < 0.5:
                img_bgr = img_bgr[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if self.rng.random() < 0.2:
                img_bgr = img_bgr[::-1, :, :].copy()
                mask = mask[::-1, :].copy()

            # 90-deg rotations
            if self.rng.random() < 0.3:
                k = self.rng.choice([1, 2, 3])
                img_bgr = np.rot90(img_bgr, k).copy()
                mask = np.rot90(mask, k).copy()

            # light noise/blur (small)
            if self.rng.random() < 0.25:
                noise = np.random.normal(0, self.rng.uniform(2.0, 8.0), img_bgr.shape).astype(np.float32)
                img_bgr = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            if self.rng.random() < 0.15:
                img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
                img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)

        # FINAL SAFETY: contiguous arrays for torch.from_numpy
        img_bgr = np.ascontiguousarray(img_bgr)
        mask = np.ascontiguousarray(mask)

        # model expects RGB? torchvision uses RGB convention, but it just learns; still keep consistent:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)

        x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        y = torch.from_numpy(mask).long()  # (H,W)
        return x, y


def make_sampler_weights(pairs: List[Tuple[str, str]], idxs: List[int], num_classes: int) -> torch.DoubleTensor:
    """
    Oversample tiles that contain door/window/front_door.
    """
    weights = np.ones(len(idxs), dtype=np.float64)
    for j, i in enumerate(idxs):
        _, mp = pairs[i]
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        has_minor = bool(((m == 2) | (m == 3) | (m == 4)).any())
        if has_minor:
            weights[j] = 3.0
    return torch.tensor(weights, dtype=torch.double)


# -------------------------
# Eval / Train
# -------------------------

@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainCfg,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
) -> Tuple[float, List[float], float]:
    model.eval()
    losses: List[float] = []
    cm_total = torch.zeros((cfg.num_classes, cfg.num_classes), dtype=torch.int64)

    if hasattr(torch, "amp"):
        autocast_ctx = lambda enabled: torch.amp.autocast(device_type="cuda", enabled=enabled)
    else:
        autocast_ctx = lambda enabled: torch.cuda.amp.autocast(enabled=enabled)

    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc="val", leave=False, dynamic_ncols=True)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    for x, y in it:
        x = x.to(device, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)

        use_amp = bool(cfg.amp and device.type == "cuda")
        with autocast_ctx(enabled=use_amp):
            out = model(x)["out"]

            if cfg.focal_gamma > 0:
                loss = focal_ce_loss(out, y, class_weights, cfg.focal_gamma)
            else:
                loss = ce_loss(out, y)

            if cfg.dice_weight > 0:
                loss = loss + cfg.dice_weight * dice_loss_multiclass_ignore_empty(out, y, cfg.num_classes)

        losses.append(float(loss.item()))
        cm_total += confusion_matrix_from_logits(out, y, cfg.num_classes)

    val_loss = float(np.mean(losses)) if losses else 0.0
    iou = iou_from_confusion(cm_total)

    minority_ids = [2, 3, 4]
    minority_vals = [iou[i] for i in minority_ids if i < len(iou) and (iou[i] == iou[i])]
    minor_iou = float(np.mean(minority_vals)) if minority_vals else 0.0
    return val_loss, iou, minor_iou


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler: Optional[object],
    cfg: TrainCfg,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
) -> float:
    model.train()
    # keep BN frozen if requested
    if cfg.freeze_bn:
        freeze_batchnorm_(model)

    losses: List[float] = []

    if hasattr(torch, "amp"):
        autocast_ctx = lambda enabled: torch.amp.autocast(device_type="cuda", enabled=enabled)
    else:
        autocast_ctx = lambda enabled: torch.cuda.amp.autocast(enabled=enabled)

    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    for x, y in it:
        x = x.to(device, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        use_amp = bool(cfg.amp and device.type == "cuda")
        with autocast_ctx(enabled=use_amp):
            out = model(x)["out"]

            if cfg.focal_gamma > 0:
                loss = focal_ce_loss(out, y, class_weights, cfg.focal_gamma)
            else:
                loss = ce_loss(out, y)

            if cfg.dice_weight > 0:
                loss = loss + cfg.dice_weight * dice_loss_multiclass_ignore_empty(out, y, cfg.num_classes)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        lv = float(loss.item())
        losses.append(lv)
        if tqdm is not None:
            it.set_postfix(loss=f"{lv:.4f}")

    return float(np.mean(losses)) if losses else 0.0


# -------------------------
# "Real" holdout evaluator (no labels)
# -------------------------

PALETTE_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
}


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, bgr in PALETTE_BGR.items():
        out[mask == cls_id] = bgr
    return out


def _overlay(img_bgr: np.ndarray, color_mask_bgr: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(max(0.0, min(1.0, alpha)))
    return cv2.addWeighted(img_bgr, 1.0 - alpha, color_mask_bgr, alpha, 0)


@torch.no_grad()
def infer_on_bgr(model: nn.Module, img_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(np.ascontiguousarray(img_rgb)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(device, non_blocking=True)
    out = model(x)["out"]
    pred = out.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred


def real_score_from_pred(pred: np.ndarray, target: float, tol: float) -> float:
    """
    Heuristic "real" score to avoid degenerate solutions:
      - if door/window/front disappear => score ~ 0
      - if they flood the image => also bad
    score in [0, 1]
    """
    total = float(pred.size)
    if total <= 0:
        return 0.0
    minor = float(((pred == 2) | (pred == 3) | (pred == 4)).sum()) / total
    # band-pass around target with tolerance
    lo = max(0.0, target - tol)
    hi = target + tol
    if minor < lo:
        # too few minor: linearly scale from 0..1 across [0..lo]
        return float(minor / (lo + 1e-9))
    if minor > hi:
        # too many minor: penalize
        return float(max(0.0, 1.0 - (minor - hi) / max(hi, 1e-9)))
    # inside band: peak at target
    mid = target
    if mid <= 0:
        return 1.0
    return float(1.0 - abs(minor - mid) / max(tol, 1e-9))


@torch.no_grad()
def eval_real_holdout(
    model: nn.Module,
    cfg: TrainCfg,
    device: torch.device,
    epoch: int,
) -> float:
    if not cfg.real_dir:
        return float("nan")
    if not os.path.isdir(cfg.real_dir):
        print(f"[real] WARNING: real_dir not found: {cfg.real_dir}")
        return float("nan")

    imgs = []
    for fn in sorted(os.listdir(cfg.real_dir)):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            imgs.append(os.path.join(cfg.real_dir, fn))
    if not imgs:
        print(f"[real] WARNING: no images in {cfg.real_dir}")
        return float("nan")

    out_root = cfg.real_out or os.path.join(os.path.dirname(cfg.out) or ".", "real_holdout")
    _ensure_dir(out_root)
    ep_dir = os.path.join(out_root, f"epoch_{epoch:04d}")
    _ensure_dir(ep_dir)

    model.eval()

    scores: List[float] = []
    for p in imgs:
        base = os.path.splitext(os.path.basename(p))[0]
        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        img_bgr = _resize_max_side(img_bgr, int(cfg.real_max_side))
        img_bgr = preprocess_like_infer(img_bgr, mode=cfg.preprocess, invert=cfg.invert)

        pred = infer_on_bgr(model, img_bgr, device)
        s = real_score_from_pred(pred, target=float(cfg.real_minor_target), tol=float(cfg.real_minor_tol))
        scores.append(float(s))

        color = _colorize_mask(pred)
        overlay = _overlay(img_bgr, color, alpha=float(cfg.real_alpha))

        od = os.path.join(ep_dir, base)
        _ensure_dir(od)
        cv2.imwrite(os.path.join(od, "input_preprocessed.png"), img_bgr)
        cv2.imwrite(os.path.join(od, "pred_mask.png"), pred)
        cv2.imwrite(os.path.join(od, "pred_color.png"), color)
        cv2.imwrite(os.path.join(od, "overlay.png"), overlay)

    if not scores:
        return float("nan")
    return float(np.mean(scores))


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-classes", type=int, default=5)

    ap.add_argument("--auto-class-weights", action="store_true")
    ap.add_argument("--focal-gamma", type=float, default=0.0)
    ap.add_argument("--dice-weight", type=float, default=0.6)

    ap.add_argument("--split-by-plan", action="store_true")
    ap.add_argument("--balance-sampler", action="store_true")

    ap.add_argument("--preprocess", default="binarize", choices=["none", "binarize"])
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--binarize-prob", type=float, default=0.0)

    ap.add_argument("--drop-last", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--cudnn-benchmark", action="store_true")

    ap.add_argument("--persistent-workers", action="store_true")
    ap.add_argument("--prefetch-factor", type=int, default=2)

    ap.add_argument("--freeze-bn", action="store_true")

    ap.add_argument("--early-stop", type=int, default=20)
    ap.add_argument("--early-stop-min-delta", type=float, default=1e-3)

    # real holdout
    ap.add_argument("--real-dir", default="")
    ap.add_argument("--real-out", default="")
    ap.add_argument("--real-max-side", type=int, default=1600)
    ap.add_argument("--real-alpha", type=float, default=0.45)
    ap.add_argument("--real-minor-target", type=float, default=0.02)
    ap.add_argument("--real-minor-tol", type=float, default=0.02)

    args = ap.parse_args()

    cfg = TrainCfg(
        data=args.data,
        out=args.out,
        epochs=int(args.epochs),
        batch=int(args.batch),
        lr=float(args.lr),
        val_split=float(args.val_split),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        num_classes=int(args.num_classes),

        auto_class_weights=bool(args.auto_class_weights),
        focal_gamma=float(args.focal_gamma),
        dice_weight=float(args.dice_weight),

        split_by_plan=bool(args.split_by_plan),
        balance_sampler=bool(args.balance_sampler),

        preprocess=str(args.preprocess),
        invert=bool(args.invert),
        binarize_prob=float(args.binarize_prob),

        drop_last=bool(args.drop_last),
        amp=bool(args.amp),
        tf32=bool(args.tf32),
        cudnn_benchmark=bool(args.cudnn_benchmark),

        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=int(args.prefetch_factor),

        freeze_bn=bool(args.freeze_bn),

        early_stop_patience=int(args.early_stop),
        early_stop_min_delta=float(args.early_stop_min_delta),

        real_dir=str(args.real_dir),
        real_out=str(args.real_out),
        real_max_side=int(args.real_max_side),
        real_alpha=float(args.real_alpha),
        real_minor_target=float(args.real_minor_target),
        real_minor_tol=float(args.real_minor_tol),
    )

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device.type} | cuda_available={torch.cuda.is_available()}")

    if cfg.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)

    pairs = scan_pairs(cfg.data)
    print(f"[data] pairs={len(pairs)} from {cfg.data}")

    tr_idx, va_idx = split_pairs(pairs, cfg.val_split, cfg.seed, cfg.split_by_plan)
    print(f"[split]{' by plan' if cfg.split_by_plan else ''}: train={len(tr_idx)} val={len(va_idx)}")

    tr_pairs = [pairs[i] for i in tr_idx]
    va_pairs = [pairs[i] for i in va_idx]

    # class weights from TRAIN only
    class_weights = None
    if cfg.auto_class_weights:
        hist = compute_pixel_hist_from_mask_paths([mp for _, mp in tr_pairs], cfg.num_classes)
        class_weights = make_class_weights_from_hist(hist).to(device)
        print("pixel_hist(train):", hist.tolist())
        print("class_weights    :", class_weights.detach().cpu().numpy().round(4).tolist())

    tr_ds = RasterPairsDataset(
        tr_pairs,
        cfg.num_classes,
        augment=True,
        preprocess=cfg.preprocess,
        invert=cfg.invert,
        binarize_prob=cfg.binarize_prob,
        seed=cfg.seed + 1,
    )
    va_ds = RasterPairsDataset(
        va_pairs,
        cfg.num_classes,
        augment=False,
        preprocess=cfg.preprocess,   # IMPORTANT: same as train/infer
        invert=cfg.invert,
        binarize_prob=0.0,
        seed=cfg.seed + 2,
    )

    sampler = None
    shuffle = True
    if cfg.balance_sampler:
        w = make_sampler_weights(pairs, tr_idx, cfg.num_classes)
        sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)
        shuffle = False
        print("[sampler] enabled: oversampling door/window tiles")

    # DataLoader knobs
    pin = (device.type == "cuda")
    nw = int(cfg.num_workers)
    pw = bool(cfg.persistent_workers and nw > 0)

    dl_common = dict(
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=pw,
    )
    if nw > 0:
        dl_common["prefetch_factor"] = int(cfg.prefetch_factor)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=cfg.batch,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=cfg.drop_last,
        **dl_common,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=max(1, cfg.batch),
        shuffle=False,
        drop_last=False,
        **dl_common,
    )

    model = build_model(cfg.num_classes).to(device)
    if cfg.freeze_bn:
        freeze_batchnorm_(model)
        print("[bn] frozen BatchNorm layers")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    scaler = None
    if cfg.amp and device.type == "cuda":
        if hasattr(torch, "amp"):
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = torch.cuda.amp.GradScaler()

    _ensure_dir(os.path.dirname(cfg.out) or ".")

    best_minor = -1e18
    best_loss = 1e18
    best_real = -1e18
    no_improve = 0

    best_loss_path = cfg.out.replace(".pt", "_bestloss.pt") if cfg.out.lower().endswith(".pt") else (cfg.out + "_bestloss.pt")
    best_real_path = cfg.out.replace(".pt", "_bestreal.pt") if cfg.out.lower().endswith(".pt") else (cfg.out + "_bestreal.pt")
    names = CLASS_NAMES[: cfg.num_classes]

    for ep in range(1, cfg.epochs + 1):
        print(f"\n=== epoch {ep}/{cfg.epochs} ===")
        t0 = time.time()

        tr_loss = train_one_epoch(model, tr_loader, opt, scaler, cfg, device, class_weights)
        va_loss, va_iou, minor_iou = eval_one_epoch(model, va_loader, cfg, device, class_weights)

        # real holdout (saves overlays every epoch by default if real_dir is provided)
        real_score = eval_real_holdout(model, cfg, device, ep)

        dt = (time.time() - t0) / 60.0
        print(f"epoch={ep}/{cfg.epochs} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} time={dt:.1f}min")
        print(f"val_iou: {fmt_iou(va_iou, names)} | minor_iou(door/window/front)={minor_iou:.3f}")
        if real_score == real_score:
            print(f"real_holdout_score={real_score:.3f} (saved overlays to: {cfg.real_out or os.path.join(os.path.dirname(cfg.out) or '.', 'real_holdout')})")

        # save best by val_loss
        if va_loss < best_loss:
            best_loss = va_loss
            ckpt = {
                "model": model.state_dict(),
                "num_classes": cfg.num_classes,
                "class_names": names,
                "best_val_loss": best_loss,
                "best_minor_iou": best_minor,
                "best_real_score": best_real,
                "epoch": ep,
                "cfg": cfg.__dict__,
            }
            _atomic_torch_save(ckpt, best_loss_path)
            print(f"saved best(loss) -> {best_loss_path}")

        # save best by minority IoU (tile-val)
        if minor_iou > (best_minor + cfg.early_stop_min_delta):
            best_minor = minor_iou
            no_improve = 0
            ckpt = {
                "model": model.state_dict(),
                "num_classes": cfg.num_classes,
                "class_names": names,
                "best_val_loss": best_loss,
                "best_minor_iou": best_minor,
                "best_real_score": best_real,
                "epoch": ep,
                "cfg": cfg.__dict__,
            }
            _atomic_torch_save(ckpt, cfg.out)
            print(f"saved best(minor_iou) -> {cfg.out}")
        else:
            no_improve += 1
            print(f"[early-stop] no improve {no_improve}/{cfg.early_stop_patience} (best_minor_iou={best_minor:.3f})")

        # save best by real score (optional)
        if real_score == real_score and real_score > (best_real + 1e-6):
            best_real = real_score
            ckpt = {
                "model": model.state_dict(),
                "num_classes": cfg.num_classes,
                "class_names": names,
                "best_val_loss": best_loss,
                "best_minor_iou": best_minor,
                "best_real_score": best_real,
                "epoch": ep,
                "cfg": cfg.__dict__,
            }
            _atomic_torch_save(ckpt, best_real_path)
            print(f"saved best(real_holdout) -> {best_real_path}")

        if cfg.early_stop_patience > 0 and no_improve >= cfg.early_stop_patience:
            print(f"[early-stop] stop: minor_iou did not improve for {cfg.early_stop_patience} epochs")
            break

    print("\ndone.")
    print(f"best(minor_iou) saved at: {cfg.out}")
    print(f"best(val_loss)  saved at: {best_loss_path}")
    if cfg.real_dir:
        print(f"best(real_holdout) saved at: {best_real_path}")


if __name__ == "__main__":
    main()
