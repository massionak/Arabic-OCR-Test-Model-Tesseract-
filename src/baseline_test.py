# -*- coding: utf-8 -*-
import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytesseract


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    bw = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY,
                               35, 15)

    black_on_white = bw
    return black_on_white


def ocr_image(img_bgr: np.ndarray, lang: str = "ara", psm: int = 6, dpi: int = 300):
    """
    Run Tesseract on a preprocessed image.
    --oem 1: LSTM only (best for printed Arabic)
    --psm 6: assume a single uniform block of text (good default)
    """
    img_proc = preprocess(img_bgr)
    config = f'--oem 1 --psm {psm} -l {lang} --dpi {dpi} -c preserve_interword_spaces=1'
    text = pytesseract.image_to_string(img_proc, config=config)
    tsv = pytesseract.image_to_data(img_proc, config=config, output_type=pytesseract.Output.DATAFRAME)
    return text, tsv


def main():
    ap = argparse.ArgumentParser(description="Printed Arabic OCR smoke test (Tesseract)")
    ap.add_argument("images", nargs="+", help="Path(s) to PNG/JPG/TIFF")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs"),
                    help="Output root folder (default: ../runs)")
    ap.add_argument("--psm", type=int, default=6, help="Tesseract PSM (default 6)")
    ap.add_argument("--dpi", type=int, default=300, help="Assumed DPI (default 300)")
    args = ap.parse_args()

    out_txt = Path(args.out) / "txt"
    out_tsv = Path(args.out) / "tsv"
    out_txt.mkdir(parents=True, exist_ok=True)
    out_tsv.mkdir(parents=True, exist_ok=True)

    for img_path in args.images:
        p = Path(img_path)
        if not p.exists():
            print(f"[WARN] not found: {p}")
            continue

        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"[WARN] cannot read: {p}")
            continue

        text, tsv_df = ocr_image(bgr, lang="ara", psm=args.psm, dpi=args.dpi)

        stem = p.stem
        txt_path = out_txt / f"{stem}.txt"
        tsv_path = out_tsv / f"{stem}.tsv"

        txt_path.write_text(text, encoding="utf-8")
        if tsv_df is not None:
            tsv_df = tsv_df.fillna("")
            tsv_df.to_csv(tsv_path, index=False, encoding="utf-8-sig")

        print(f"[OK] {p} -> {txt_path.name} | {tsv_path.name}")


if __name__ == "__main__":
    sys.exit(main())
