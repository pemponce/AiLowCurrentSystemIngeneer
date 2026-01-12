from __future__ import annotations

import argparse
import os
import os.path as osp

import fitz  # PyMuPDF


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--out", required=True, help="Output PNG path or output directory")
    ap.add_argument("--page", type=int, default=1, help="Page number (1-based)")
    ap.add_argument("--dpi", type=int, default=300, help="Render DPI (200-400 usually ok)")
    args = ap.parse_args()

    pdf_path = args.pdf
    page_num = int(args.page)
    dpi = int(args.dpi)

    if page_num < 1:
        raise SystemExit("--page must be >= 1")

    doc = fitz.open(pdf_path)
    if page_num > doc.page_count:
        raise SystemExit(f"PDF has only {doc.page_count} pages, but --page={page_num}")

    page = doc.load_page(page_num - 1)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    out = args.out
    if osp.isdir(out) or out.endswith("\\") or out.endswith("/"):
        os.makedirs(out, exist_ok=True)
        base = osp.splitext(osp.basename(pdf_path))[0]
        out_path = osp.join(out, f"{base}_p{page_num}.png")
    else:
        os.makedirs(osp.dirname(out) or ".", exist_ok=True)
        out_path = out

    pix.save(out_path)
    print(f"Saved PNG: {out_path}")
    print(f"Size: {pix.width}x{pix.height}  dpi={dpi}")


if __name__ == "__main__":
    main()
