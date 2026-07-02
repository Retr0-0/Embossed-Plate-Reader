import os
import urllib.request
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
FONT_URL = "https://raw.githubusercontent.com/ZosoV/platesGenerator/master/fonts/FE-FONT.ttf"


def ensure_font(path="assets/FE-FONT.ttf"):
    """Use a committed font if present, otherwise download and validate it."""
    if os.path.exists(path) and os.path.getsize(path) > 5000:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    req = urllib.request.Request(FONT_URL, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=30).read()
    if data[:4] not in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"ttcf") or len(data) < 5000:
        raise RuntimeError("Downloaded font is invalid — check your connection or the URL.")
    with open(path, "wb") as f:
        f.write(data)
    return path


def build_templates(font_path, H=96):
    font = ImageFont.truetype(font_path, 200)
    tpl = {}
    for ch in CHARS:
        im = Image.new("L", (320, 340), 255)
        d = ImageDraw.Draw(im)
        bb = d.textbbox((0, 0), ch, font=font)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((320 - w) / 2 - bb[0], (340 - h) / 2 - bb[1]), ch, fill=0, font=font)
        a = cv2.threshold(np.array(im), 128, 255, cv2.THRESH_BINARY)[1]
        ys, xs = np.where(a < 128)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        nw = max(1, int(a.shape[1] * H / a.shape[0]))
        tpl[ch] = cv2.threshold(cv2.resize(a, (nw, H), interpolation=cv2.INTER_AREA),
                                128, 255, cv2.THRESH_BINARY)[1]
    return tpl


def _norm(g, H=96):
    nw = max(1, int(g.shape[1] * H / g.shape[0]))
    return cv2.threshold(cv2.resize(g, (nw, H), interpolation=cv2.INTER_AREA),
                         128, 255, cv2.THRESH_BINARY)[1]


def _match(q, tpl):
    best, best_score = "?", -1
    for c, t in tpl.items():
        W = max(q.shape[1], t.shape[1]) + 8
        def place(x):
            canvas = np.full((96, W), 255, np.uint8)
            o = (W - x.shape[1]) // 2
            canvas[:, o:o + x.shape[1]] = x
            return canvas
        A, B = place(q) < 128, place(t) < 128
        union = (A | B).sum()
        score = (A & B).sum() / union if union else 0
        if score > best_score:
            best_score, best = score, c
    return best, best_score


def read_plate(image_path, tpl, char_color="red", row_tol=0.10, min_score=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    h, w = img.shape[:2]
    if w < 1600:
        s = 1600 / w
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if char_color == "red":
        mask = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255)) | \
               cv2.inRange(hsv, (165, 80, 60), (180, 255, 255))
    else:
        g = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 11, 60, 60)
        mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    blue = cv2.inRange(hsv, (95, 70, 40), (135, 255, 255))

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        ar = bw / float(bh)
        if 0.30 * H < bh < 0.95 * H and 0.02 * W < bw < 0.30 * W and 0.12 < ar < 1.15:
            if blue[y:y + bh, x:x + bw].mean() / 255.0 < 0.05:
                boxes.append((x, y, bw, bh))
    if not boxes:
        return [], img

    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    rows, tol = [], row_tol * H
    for b in boxes:
        cy = b[1] + b[3] / 2
        for r in rows:
            if abs(cy - np.mean([bb[1] + bb[3] / 2 for bb in r])) < tol:
                r.append(b)
                break
        else:
            rows.append([b])
    rows.sort(key=lambda r: np.mean([bb[1] for bb in r]))
    for r in rows:
        r.sort(key=lambda b: b[0])

    vis = img.copy()
    result = []
    for r in rows:
        chars = []
        for (x, y, bw, bh) in r:
            q = _norm(cv2.bitwise_not(mask[y:y + bh, x:x + bw]))
            ch, sc = _match(q, tpl)
            chars.append(ch)
            col = (0, 180, 0) if sc >= min_score else (0, 0, 255)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), col, 3)
            cv2.putText(vis, ch, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        result.append("".join(chars))
    return result, vis