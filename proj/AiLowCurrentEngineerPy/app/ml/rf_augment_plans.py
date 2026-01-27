from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np


def _read_png(path: Path, gray: bool) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    return img


def _write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _binarize_layer(layer_gray: np.ndarray) -> np.ndarray:
    """
    Expect: white background, dark/black strokes (CAD plots).
    Output: 0/255 where 255 = foreground strokes.

    ВАЖНО: не делаем морфологический open, чтобы не убивать тонкие дуги дверей/окон.
    """
    g = layer_gray
    # если внезапно слой "белые линии на тёмном" — инвертнём
    if g.mean() < 127:
        g = 255 - g

    # инверт: линии станут светлыми, фон тёмным
    inv = 255 - g

    # порог достаточно мягкий, чтобы брать антиалиасинг
    # (если слишком много мусора — подними thresh до 20-30)
    _, bw = cv2.threshold(inv, 12, 255, cv2.THRESH_BINARY)

    # минимальный dilate, чтобы тонкие линии не исчезали после affine
    bw = cv2.dilate(bw, np.ones((2, 2), np.uint8), iterations=1)
    return bw


def _unbinarize_layer(bw: np.ndarray) -> np.ndarray:
    # Store back as white background + black strokes (like your exports)
    return 255 - bw


def _apply_affine(img: np.ndarray, M: np.ndarray, is_mask: bool) -> np.ndarray:
    h, w = img.shape[:2]
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    if is_mask:
        border_val = 0
    else:
        border_val = (255, 255, 255) if img.ndim == 3 else 255
    out = cv2.warpAffine(
        img, M, (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )
    return out


def _maybe_hflip(img: np.ndarray, do_flip: bool) -> np.ndarray:
    if not do_flip:
        return img
    return cv2.flip(img, 1)


def _photometric_base(
    img_bgr: np.ndarray,
    rng: np.random.RandomState,
    brightness: int,
    contrast: float,
    noise_sigma: float
) -> np.ndarray:
    out = img_bgr.astype(np.float32)

    if contrast > 0:
        alpha = 1.0 + rng.uniform(-contrast, contrast)
        beta = rng.uniform(-brightness, brightness)
        out = out * alpha + beta

    if noise_sigma > 0 and rng.rand() < 0.5:
        n = rng.normal(0.0, noise_sigma, size=out.shape).astype(np.float32)
        out = out + n

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _find_plan_files(plan_dir: Path) -> Dict[str, Path]:
    # Support both base.png and input.png
    base = plan_dir / "base.png"
    if not base.exists():
        base = plan_dir / "input.png"
    if not base.exists():
        raise RuntimeError(f"Missing base/input in: {plan_dir}")

    out = {"base": base}

    for name in ["wall.png", "door.png", "window.png", "front_door.png"]:
        p = plan_dir / name
        if p.exists():
            out[name] = p

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-plan", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--copy-original", action="store_true")
    ap.add_argument("--rot-deg", type=float, default=2.5)
    ap.add_argument("--scale-jitter", type=float, default=0.03)
    ap.add_argument("--translate-frac", type=float, default=0.02)
    ap.add_argument("--flip-p", type=float, default=0.25)
    ap.add_argument("--brightness", type=int, default=10)
    ap.add_argument("--contrast", type=float, default=0.06)
    ap.add_argument("--noise-sigma", type=float, default=3.0)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(int(args.seed))

    plans = sorted([p for p in src.iterdir() if p.is_dir()])
    if not plans:
        raise RuntimeError(f"No plan folders in {src}")

    if args.copy_original:
        for pd in plans:
            files = _find_plan_files(pd)
            dst = out / pd.name
            dst.mkdir(parents=True, exist_ok=True)
            for _, p in files.items():
                (dst / p.name).write_bytes(p.read_bytes())

    for pd in plans:
        files = _find_plan_files(pd)

        base = _read_png(files["base"], gray=False)
        if base is None:
            raise RuntimeError(f"Cannot read: {files['base']}")
        h, w = base.shape[:2]

        layers_bw: Dict[str, np.ndarray] = {}
        for ln in ["wall.png", "door.png", "window.png", "front_door.png"]:
            if ln not in files:
                continue
            g = _read_png(files[ln], gray=True)
            if g is None:
                continue
            if g.shape[:2] != (h, w):
                raise RuntimeError(f"Layer size mismatch in {pd}: {ln} is {g.shape[:2]} but base is {(h, w)}")
            layers_bw[ln] = _binarize_layer(g)

        for i in range(int(args.n_per_plan)):
            ang = float(rng.uniform(-args.rot_deg, args.rot_deg))
            sc = float(rng.uniform(1.0 - args.scale_jitter, 1.0 + args.scale_jitter))
            cx, cy = w * 0.5, h * 0.5
            M = cv2.getRotationMatrix2D((cx, cy), ang, sc)
            M[0, 2] += float(rng.uniform(-args.translate_frac, args.translate_frac) * w)
            M[1, 2] += float(rng.uniform(-args.translate_frac, args.translate_frac) * h)

            # ВАЖНО: один и тот же флип для base и всех масок
            do_flip = bool(rng.rand() < float(args.flip_p))

            base_aug = _apply_affine(base, M, is_mask=False)
            base_aug = _maybe_hflip(base_aug, do_flip)
            base_aug = _photometric_base(
                base_aug, rng,
                int(args.brightness),
                float(args.contrast),
                float(args.noise_sigma),
            )

            out_plan = out / f"{pd.name}_aug_{i:03d}"
            out_plan.mkdir(parents=True, exist_ok=True)
            _write_png(out_plan / "input.png", base_aug)

            for ln, bw in layers_bw.items():
                m_aug = _apply_affine(bw, M, is_mask=True)
                m_aug = _maybe_hflip(m_aug, do_flip)
                m_aug = ((m_aug > 127).astype(np.uint8) * 255)
                _write_png(out_plan / ln, _unbinarize_layer(m_aug))

        print(f"[ok] {pd.name}: generated {args.n_per_plan} variants")

    print("Done.")


if __name__ == "__main__":
    main()
