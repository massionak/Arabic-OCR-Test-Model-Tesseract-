# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import cv2
import pandas as pd


def visualize(image_path: Path, tsv_path: Path, out_path: Path, min_conf: int = 50):
    img = cv2.imread(str(image_path))
    assert img is not None, f"Cannot read image: {image_path}"

    df = pd.read_csv(tsv_path, sep=",") if tsv_path.suffix.lower() == ".csv" else pd.read_csv(tsv_path, sep=",", engine="python")
    if "level" not in df.columns:
        try:
            df = pd.read_csv(tsv_path, sep="\t")
        except Exception:
            raise RuntimeError("Could not parse TSV/CSV; ensure it was written by baseline_test.py")

    cols_needed = {"level","left","top","width","height","conf","text"}
    if not cols_needed.issubset(set(df.columns)):
        df = pd.read_csv(tsv_path, sep="\t")
        if not cols_needed.issubset(set(df.columns)):
            raise RuntimeError(f"Missing columns in {tsv_path}. Found: {df.columns}")

    words = df[(df["level"] == 5) & (df["conf"].astype(str) != "-1")].copy()
    words["conf"] = pd.to_numeric(words["conf"], errors="coerce").fillna(-1).astype(int)
    words = words[words["conf"] >= min_conf]

    # Draw boxes
    for idx, row in words.reset_index(drop=True).iterrows():
        x, y, w, h = int(row["left"]), int(row["top"]), int(row["width"]), int(row["height"])
        conf = int(row["conf"])
        label = f"#{idx}|{conf}"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 180, 0), 2)
        cv2.putText(img, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[OK] Visualization saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Visualize Tesseract word boxes (level=5) with confidences")
    ap.add_argument("image", help="Path to the original image used for OCR")
    ap.add_argument("tsv", help="Path to the TSV/CSV output from baseline_test.py")
    ap.add_argument("--out", default=None, help="Output image path (default: runs/vis/<stem>_vis.png)")
    ap.add_argument("--min-conf", type=int, default=50, help="Minimum confidence to draw (0-100)")
    args = ap.parse_args()

    img_path = Path(args.image)
    tsv_path = Path(args.tsv)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(__file__).resolve().parents[1] / "runs" / "vis" / f"{img_path.stem}_vis.png"

    visualize(img_path, tsv_path, out_path, min_conf=args.min_conf)


if __name__ == "__main__":
    main()
