import argparse
import cv2
from fe_plate_reader import ensure_font, build_templates, read_plate


def main():
    ap = argparse.ArgumentParser(description="Read FE-Schrift license plates.")
    ap.add_argument("image", help="path to the plate image")
    ap.add_argument("--color", default="red", choices=["red", "dark"],
                    help="character colour on the plate")
    ap.add_argument("--save", help="optional path to save the annotated image")
    args = ap.parse_args()

    tpl = build_templates(ensure_font())
    rows, vis = read_plate(args.image, tpl, char_color=args.color)

    for i, row in enumerate(rows, 1):
        print(f"Row {i}: {' '.join(row)}")
    print("Plate:", " ".join(rows))

    if args.save:
        cv2.imwrite(args.save, vis)
        print(f"Saved annotated image to {args.save}")
    else:
        cv2.imshow("Detected characters", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()