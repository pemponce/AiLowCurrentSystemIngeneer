import argparse
import os
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root with masks/")
    ap.add_argument("--num-classes", type=int, default=5)
    args = ap.parse_args()

    p = os.path.join(args.data, "masks")
    hs = np.zeros(args.num_classes, dtype=np.int64)

    fs = [f for f in os.listdir(p) if f.lower().endswith(".png") and "_overlay" not in f.lower()]
    for f in fs:
        m = cv2.imread(os.path.join(p, f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        b = np.bincount(m.reshape(-1), minlength=args.num_classes)
        hs += b[: args.num_classes]

    tot = int(hs.sum())
    print("files:", len(fs))
    print("pixels:", hs.tolist(), "total:", tot)
    if tot > 0:
        print("share:", (hs / tot).round(6).tolist())


if __name__ == "__main__":
    main()
