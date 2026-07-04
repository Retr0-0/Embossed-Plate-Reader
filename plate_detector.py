import cv2
import numpy as np
from open_image_models import LicensePlateDetector

DEFAULT_MODEL = "yolo-v9-s-608-license-plate-end2end"
_DETECTOR = None


def get_detector(model=DEFAULT_MODEL, conf=0.25):
    """Load the plate detector once. First call downloads the ONNX model
    (~a few MB) to ~/.cache/open-image-models."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = LicensePlateDetector(detection_model=model, conf_thresh=conf)
    return _DETECTOR


def detect_plates(img, model=DEFAULT_MODEL, conf=0.25, pad=0.35):
    """Return [(x1, y1, x2, y2, confidence), ...] for each detected plate,
    padded slightly and sorted by confidence (highest first)."""
    det = get_detector(model, conf)
    H, W = img.shape[:2]
    boxes = []
    for r in det.predict(img):
        bb = r.bounding_box
        pw, ph = int(bb.width * pad), int(bb.height * pad)
        x1 = max(0, bb.x1 - pw); y1 = max(0, bb.y1 - ph)
        x2 = min(W, bb.x2 + pw); y2 = min(H, bb.y2 + ph)
        boxes.append((x1, y1, x2, y2, float(r.confidence)))
    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes

def _plate_text_mask(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_cov = (cv2.inRange(hsv, (0, 90, 60), (12, 255, 255)) |
               cv2.inRange(hsv, (165, 90, 60), (180, 255, 255))).mean() / 255
    if red_cov > 0.04:
        m = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255)) | \
            cv2.inRange(hsv, (165, 80, 60), (180, 255, 255))
    else:
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        m = cv2.threshold(cv2.bilateralFilter(g, 11, 60, 60), 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _rotate_expand(im, angle, border=(255, 255, 255)):
    H, W = im.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(H * s + W * c), int(H * c + W * s)
    M[0, 2] += (nW - W) / 2
    M[1, 2] += (nH - H) / 2
    return cv2.warpAffine(im, M, (nW, nH), flags=cv2.INTER_CUBIC, borderValue=border)


def _recrop_to_text(im):
    m = _plate_text_mask(im)
    H, W = im.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = [stats[i] for i in range(1, n)
            if 0.06 * H < stats[i][3] < 0.9 * H and stats[i][2] < 0.5 * W]
    if not keep:
        return im
    x0 = min(s[0] for s in keep); y0 = min(s[1] for s in keep)
    x1 = max(s[0] + s[2] for s in keep); y1 = max(s[1] + s[3] for s in keep)
    mx = int(0.06 * (x1 - x0)) + 8
    my = int(0.10 * (y1 - y0)) + 8
    return im[max(0, y0 - my):min(H, y1 + my), max(0, x0 - mx):min(W, x1 + mx)]


def deskew_plate(crop, max_angle=30, step=0.5, min_apply=1.0):
    """Rotate a plate crop so its text is horizontal, then re-crop tight.
    Returns (straightened_crop, angle_applied)."""
    mask = _plate_text_mask(crop)
    H, W = mask.shape
    best_a, best_s = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + step, step):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0)
        r = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST)
        score = float(np.sum(np.diff(r.sum(axis=1, dtype=np.float64)) ** 2))
        if score > best_s:
            best_s, best_a = score, a
    rotated = crop if abs(best_a) < min_apply else _rotate_expand(crop, best_a)
    return _recrop_to_text(rotated), best_a