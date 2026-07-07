import os
import urllib.request
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from glyph_classifier import classify

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
FONT_URL = "https://raw.githubusercontent.com/ZosoV/platesGenerator/master/fonts/FE-FONT.ttf"

import difflib
PROVINCES = ["KOSHI", "MADHESH", "BAGMATI", "GANDAKI", "LUMBINI", "KARNALI", "SUDURPASHCHIM"]

def snap_province(text, threshold=0.55):
    """Snap a (possibly mangled) header OCR result to the nearest Nepali province
    name. Leaves non-province text (junk, NPL/NEP, empty) untouched."""
    letters = "".join(ch for ch in text.upper() if ch.isalpha())
    if not letters:
        return text
    best, score = text, 0.0
    for p in PROVINCES:
        r = difflib.SequenceMatcher(None, letters, p).ratio()
        if r > score:
            score, best = r, p
    return best if score >= threshold else text

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
    """Render every FE-Schrift glyph once, tight-cropped and height-normalised."""
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


def _is_flag(hsv_sub, plate_is_red=False):
    """Reject the Nepal flag / emblem. Normally it has BOTH red and blue. On low-res
    crops the blue can wash out, so on a non-red-text plate we also reject a strongly
    red-dominant blob (no real character is red unless the whole plate is red-text)."""
    red = (cv2.inRange(hsv_sub, (0, 90, 50), (12, 255, 255)) |
           cv2.inRange(hsv_sub, (165, 90, 50), (180, 255, 255))).mean() / 255
    blue = cv2.inRange(hsv_sub, (95, 90, 50), (135, 255, 255)).mean() / 255
    if red > 0.08 and blue > 0.02:
        return True
    if red > 0.20 and not plate_is_red:      # red blob on a non-red plate = flag
        return True
    return False


def _detect_color(hsv):
    red = (cv2.inRange(hsv, (0, 90, 60), (12, 255, 255)) |
           cv2.inRange(hsv, (165, 90, 60), (180, 255, 255))).mean() / 255
    return "red" if red > 0.04 else "dark"


def _largest_cluster(boxes, gap_factor=2.2):
    """Keep the biggest group of characters that sit close together, then drop any
    that fall outside the main vertical band (rocks/debris below or above the plate)."""
    if len(boxes) <= 2:
        return boxes
    hs = sorted(b[3] for b in boxes)
    thr = gap_factor * hs[len(hs) // 2]
    ctr = [(b[0] + b[2] / 2, b[1] + b[3] / 2) for b in boxes]
    n = len(boxes)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if ((ctr[i][0] - ctr[j][0]) ** 2 + (ctr[i][1] - ctr[j][1]) ** 2) ** 0.5 < thr:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])
    kept = max(groups.values(), key=len)

    # junk guard: group boxes into rows; keep every multi-box row, but keep a LONE
    # box only if it's char-sized and within the number's x-span (a real stacked
    # row like a single top-row 'A'). Drops debris: wrong-size or off-to-the-side.
    med_h = sorted(b[3] for b in kept)[len(kept) // 2]
    ys = sorted(kept, key=lambda b: b[1] + b[3] / 2)
    bands, gap = [[ys[0]]], 0.6 * med_h
    for b in ys[1:]:
        prev_cy = bands[-1][-1][1] + bands[-1][-1][3] / 2
        if (b[1] + b[3] / 2) - prev_cy < gap:
            bands[-1].append(b)
        else:
            bands.append([b])
    main_band = max(bands, key=len)
    x0 = min(b[0] for b in main_band)
    x1 = max(b[0] + b[2] for b in main_band)
    out = []
    for band in bands:
        if len(band) >= 2:
            out += band
        else:
            b = band[0]
            cx = b[0] + b[2] / 2
            if 0.7 * med_h < b[3] < 1.4 * med_h and x0 - 8 <= cx <= x1 + 8:
                out += band
    return out

def read_plate(image, tpl, char_color="auto", row_tol=0.10, min_score=0.4, read_header=True):
    """
    Returns (info, annotated_image) where info = {"header", "number", "plate"}.
    image      : a file path OR an already-loaded BGR image (numpy array).
    char_color : "auto" (default), "red", or "dark".
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(image)
    else:
        img = image

    h, w = img.shape[:2]
    if w < 1600:
        s = 1600 / w
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if char_color == "auto":
        char_color = _detect_color(hsv)

    # isolate the characters
    if char_color == "red":
        mask = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255)) | \
               cv2.inRange(hsv, (165, 80, 60), (180, 255, 255))
    else:
        mask = cv2.threshold(cv2.bilateralFilter(gray, 11, 60, 60), 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # components: drop frame, tiny noise, and the flag/emblem
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if bh > 0.85 * H or bw > 0.85 * W or bh < 0.03 * H:
            continue
        if _is_flag(hsv[y:y + bh, x:x + bh], plate_is_red=(char_color == "red")):
            continue
        comps.append((x, y, bw, bh))
    glyphs = [b for b in comps if b[3] > 0.12 * H and 0.12 < b[2] / b[3] < 1.3]
    if not glyphs:
        return {"header": "", "number": [], "plate": ""}, img

    # drop over-tall frame/chrome slivers so they don't skew the height reference
    med_h = float(np.median([b[3] for b in glyphs]))
    glyphs = [b for b in glyphs if b[3] < 1.8 * med_h]

    # MAIN number = the tallest cluster, kept spatially compact
    max_h = max(b[3] for b in glyphs)
    main = [b for b in glyphs if b[3] > 0.55 * max_h]
    main = _largest_cluster(main)
    print("DEBUG y-centers:", sorted(round((b[1] + b[3] / 2) / img.shape[0], 2) for b in main))

    # match candidates; keep ONLY confident matches (drops chrome/frame strays,
    # which match FE glyphs poorly and score low)
    kept = []
    for (x, y, bw, bh) in main:
        q = _norm(cv2.bitwise_not(mask[y:y + bh, x:x + bw]))
        ch, sc = _match(q, tpl)
        if sc < 0.35:                      # template says "not a character" -> junk, reject
            continue                        # (don't let the CNN rescue a screw/rock/hole)
        if sc < 0.55:                      # template unsure but plausible -> CNN for worn glyphs
            cnn = classify(q)
            if cnn is not None:
                ch, sc = cnn
        if sc >= min_score:
            kept.append((x, y, bw, bh, ch))    
        if not kept:
            return {"header": "", "number": [], "plate": ""}, img
    main = [(x, y, bw, bh) for (x, y, bw, bh, _) in kept]
    char_of = {b[:4]: b[4] for b in kept}

    # group into rows by vertical alignment, sort each row left -> right
    main.sort(key=lambda b: b[1] + b[3] / 2)
    rows, tol = [], row_tol * H
    for b in main:
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

    # draw + assemble (every kept glyph is a confident match -> green)
    vis = img.copy()
    number_rows = []
    for r in rows:
        chars = []
        for (x, y, bw, bh) in r:
            ch = char_of[(x, y, bw, bh)]
            chars.append(ch)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 180, 0), 3)
            cv2.putText(vis, ch, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 180, 0), 3)
        number_rows.append("".join(chars))

    number_rows = tidy_number(number_rows)

    # HEADER = the smaller text band above the (clean) number block
    header = ""
    if read_header and main:
        top = min(b[1] for b in main)
        mx0 = min(b[0] for b in main)
        mx1 = max(b[0] + b[2] for b in main)
        hc = [b for b in comps if b[1] + b[3] <= top and b[3] < 0.55 * max_h
              and b[0] >= mx0 - 30 and b[0] + b[2] <= mx1 + 30]
        hc = _largest_cluster(hc)          # drop stray screws/holes that stretch the header crop
        if hc:
            hx0 = min(b[0] for b in hc); hy0 = min(b[1] for b in hc)
            hx1 = max(b[0] + b[2] for b in hc); hy1 = max(b[1] + b[3] for b in hc)
            px = int(0.06 * (hx1 - hx0)) + 15
            py = 12
            band = gray[max(0, hy0 - py):hy1 + py, max(0, hx0 - px):min(W, hx1 + px)]
            big = cv2.resize(band, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            bt = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            bt = cv2.copyMakeBorder(bt, 20, 40, 20, 40, cv2.BORDER_CONSTANT, value=255)
            header = pytesseract.image_to_string(
                bt, config="--oem 3 --psm 8 -c tessedit_char_whitelist="
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZ ").strip()
            cv2.rectangle(vis, (hx0 - px, hy0 - py), (hx1 + px, hy1 + py), (200, 120, 0), 2)
            cv2.putText(vis, header, (hx0, max(24, hy0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 120, 0), 2)
    header = snap_province(header)
    plate = (header + " | " if header else "") + " ".join(number_rows)
    return {"header": header, "number": number_rows, "plate": plate}, vis


# FE-Schrift look-alikes: letter-shape -> digit, and digit-shape -> letter
_L2D = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1",
        "Z": "7", "T": "7", "S": "5", "G": "0", "C": "0", "B": "8"}
_D2L = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}


def tidy_number(rows, ndigits=4):
    """Fix letter/digit look-alikes using the plate structure:
    leading letters then exactly `ndigits` trailing digits (e.g. AAB + 7880)."""
    joined = "".join(rows)
    if len(joined) < ndigits:
        return rows
    head, tail = joined[:-ndigits], joined[-ndigits:]
    head = "".join(_D2L.get(c, c) if c.isdigit() else c for c in head)
    tail = "".join(_L2D.get(c, c) if c.isalpha() else c for c in tail)
    fixed = head + tail
    out, i = [], 0
    for r in rows:
        out.append(fixed[i:i + len(r)])
        i += len(r)
    return out