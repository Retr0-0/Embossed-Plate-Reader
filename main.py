import argparse
import os
import cv2
from emb_plate_reader import ensure_font, build_templates, read_plate
from plate_detector import detect_plates, deskew_plate
from plate_db import save_reading


def main():
    ap = argparse.ArgumentParser(description="Detect, straighten, and read FE-Schrift plates.")
    ap.add_argument("image", help="path to the photo")
    ap.add_argument("--color", default="auto", choices=["auto", "red", "dark"])
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    ap.add_argument("--no-detect", action="store_true", help="treat the whole image as the plate")
    ap.add_argument("--no-deskew", action="store_true", help="skip straightening")
    ap.add_argument("--no-header", action="store_true", help="skip the header word")
    ap.add_argument("--save", help="output filename (saved into recognized/)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not open image: {args.image}")
    H, W = img.shape[:2]

    tpl = build_templates(ensure_font())

    # Stage 1: detect
    if args.no_detect:
        boxes = [(0, 0, W, H, 1.0)]
    else:
        boxes = detect_plates(img, conf=args.conf)
        if not boxes:
            print("No plate detected — reading the whole image instead.")
            boxes = [(0, 0, W, H, 1.0)]

    out_dir = "recognized"
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(args.save or os.path.basename(args.image))[0]

    annotated = img.copy()
    for i, (x1, y1, x2, y2, conf) in enumerate(boxes, 1):
        crop = img[y1:y2, x1:x2]

        # Stage 2: straighten
        angle = 0.0
        if not args.no_deskew:
            crop, angle = deskew_plate(crop)

        # Stage 3: read
        info, vis = read_plate(crop, tpl, char_color=args.color, read_header=not args.no_header)

        print(f"\nPlate {i} (conf {conf:.2f}, deskew {angle:+.1f} deg):")
        if info["header"]:
            print("  Header:", info["header"])
        for j, row in enumerate(info["number"], 1):
            print(f"  Number row {j}: {' '.join(row)}")
        print("  =>", info["plate"])

        # full image gets the detection box + recognized string
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 3)
        cv2.putText(annotated, info["plate"], (x1, max(30, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        # straightened, per-character-annotated plate saved separately
        cv2.imwrite(os.path.join(out_dir, f"{stem}_plate{i}.png"), vis)

    out_path = os.path.join(out_dir, stem + ".png")
    cv2.imwrite(out_path, annotated)
    save_reading(args.image, info, out_path)          # log this reading to plates.db
    print(f"\nSaved: {out_path}  (+ straightened plate crop)")


if __name__ == "__main__":
    main()