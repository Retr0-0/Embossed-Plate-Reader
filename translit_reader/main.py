"""CLI for romanized (Devanagari-transliterated) Nepali plates.

    python -m translit_reader.main path/to/plate.jpg
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plate_detector import detect_plates  # noqa: E402
from plate_db import save_reading  # noqa: E402
from translit_reader.reader import read_translit_plate, load_templates, deskew_translit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Read Nepali plates written in English but transliterated from "
                    "Devanagari (e.g. BAGMATI PRADESH-01 / 030 CHA 7911).")
    ap.add_argument("image", help="path to the photo")
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    ap.add_argument("--no-detect", action="store_true", help="treat the whole image as the plate")
    ap.add_argument("--no-deskew", action="store_true", help="skip straightening")
    ap.add_argument("--no-header", action="store_true", help="skip the province band")
    ap.add_argument("--no-db", action="store_true", help="don't log to plates.db")
    ap.add_argument("--out", default="translit_reader/recognized",
                    help="where to write the annotated images")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not open image: {args.image}")
    H, W = img.shape[:2]

    tpl = load_templates()

    if args.no_detect:
        boxes = [(0, 0, W, H, 1.0)]
    else:
        boxes = detect_plates(img, conf=args.conf)
        if not boxes:
            print("No plate detected — reading the whole image instead.")
            boxes = [(0, 0, W, H, 1.0)]

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    annotated = img.copy()
    info = None
    for i, (x1, y1, x2, y2, conf) in enumerate(boxes, 1):
        crop = img[y1:y2, x1:x2]

        angle = 0.0
        if not args.no_deskew:
            crop, angle = deskew_translit(crop)

        info, vis = read_translit_plate(crop, tpl, read_header=not args.no_header)

        print(f"\nPlate {i} (conf {conf:.2f}, deskew {angle:+.1f} deg):")
        print(f"  Province : {info['province'] or '-'}"
              + (f" (code {info['code']:02d})" if info["code"] is not None else ""))
        print(f"  Lot      : {info['lot'] or '-'}")
        print(f"  Syllable : {info['syllable'] or '-'}"
              + (f"  [Devanagari {info['devanagari']}]" if info["devanagari"] else ""))
        print(f"  Serial   : {info['serial'] or '-'}")
        print(f"  Raw rows : {info['number']}")
        print("  =>", info["plate"])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 3)
        cv2.putText(annotated, info["plate"], (x1, max(30, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        cv2.imwrite(os.path.join(args.out, f"{stem}_plate{i}.png"), vis)

    out_path = os.path.join(args.out, stem + ".png")
    cv2.imwrite(out_path, annotated)
    if info and not args.no_db:
        save_reading(args.image, info, out_path)
    print(f"\nSaved: {out_path}  (+ straightened plate crop)")


if __name__ == "__main__":
    main()
