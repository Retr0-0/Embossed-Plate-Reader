"""Reader for romanized (Devanagari-transliterated) Nepali plates.

Difference from ../emb_plate_reader.py, which this reuses the glyph engine from:

* those plates have coloured characters on a light field, so the mask hunts for
  red/dark ink. The plates here are the reverse -- LIGHT characters embossed on
  a coloured field (red private, black public, ...), so the polarity is picked
  per image instead of assumed.
* the number is <lot digits><class syllable><4-digit serial>, not
  <letters><digits>, so parsing goes through translit_tokens.parse_number.
"""
import os
import sys

import cv2
import numpy as np
import pytesseract

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emb_plate_reader import ensure_font, build_templates, _norm, _largest_cluster  # noqa: E402
from plate_detector import _rotate_expand  # noqa: E402
from glyph_classifier import classify  # noqa: E402
from translit_reader.translit_tokens import (  # noqa: E402
    snap_province, parse_number, format_plate,
)

HEADER_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- "
NUMBER_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# How far a glyph may be re-stretched horizontally before being compared to a
# template. On these plates the serial line is often set in a wider, bolder face
# than the header line, and the root _match compares at fixed aspect -- so a wide
# "7" overlaps the FE template "7" poorly and loses to a wide letter like "Z".
MAX_STRETCH = 1.45


def _match_flex(q, tpl):
    """Aspect-tolerant version of emb_plate_reader._match.

    Each template is compared against the glyph rescaled toward that template's
    width (bounded by MAX_STRETCH), so a horizontally stretched digit still
    matches its own template. At a ratio of 1 this degrades to the plain match.
    """
    best, best_score = "?", -1.0
    for c, t in tpl.items():
        r = min(max(t.shape[1] / q.shape[1], 1 / MAX_STRETCH), MAX_STRETCH)
        nw = max(1, int(round(q.shape[1] * r)))
        qs = cv2.threshold(cv2.resize(q, (nw, 96), interpolation=cv2.INTER_AREA),
                           128, 255, cv2.THRESH_BINARY)[1]
        W = max(nw, t.shape[1]) + 8

        def place(x):
            canvas = np.full((96, W), 255, np.uint8)
            o = (W - x.shape[1]) // 2
            canvas[:, o:o + x.shape[1]] = x
            return canvas

        A, B = place(qs) < 128, place(t) < 128
        union = (A | B).sum()
        score = (A & B).sum() / union if union else 0.0
        if score > best_score:
            best_score, best = score, c
    return best, best_score


def _charlike(mask, H, W):
    """Components that could plausibly be plate characters."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if bh > 0.85 * H or bw > 0.85 * W or bh < 0.05 * H:
            continue
        if not 0.12 < bw / bh < 1.5:
            continue
        out.append((x, y, bw, bh))
    return out


def _ink_mask(img):
    """Pick the polarity that actually contains the text.

    Embossed transliterated plates are light glyphs on a coloured field, but
    older painted ones are dark glyphs on white -- so build both masks and keep
    whichever yields more character-shaped components.
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(gray, 11, 60, 60)
    light = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    dark = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    best, best_mask, best_polarity = -1, light, "light"
    for m, name in ((light, "light"), (dark, "dark")):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        score = len(_charlike(m, H, W))
        if score > best:
            best, best_mask, best_polarity = score, m, name
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return best_mask, best_polarity


def _recrop_to_text(im):
    """Tighten a crop onto its text, using the auto-detected ink polarity."""
    H, W = im.shape[:2]
    mask, _ = _ink_mask(im)
    boxes = _charlike(mask, H, W)
    if not boxes:
        return im
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    mx = int(0.06 * (x1 - x0)) + 8
    my = int(0.10 * (y1 - y0)) + 8
    return im[max(0, y0 - my):min(H, y1 + my), max(0, x0 - mx):min(W, x1 + mx)]


def deskew_translit(crop, max_angle=30, step=0.5, min_apply=1.0):
    """Straighten a plate crop, then re-crop tight. Returns (crop, angle).

    plate_detector.deskew_plate cannot be reused here: it masks RED as the ink,
    but on these plates red is the BACKGROUND and the glyphs are light, so it
    straightens and crops to the wrong thing.
    """
    mask, _ = _ink_mask(crop)
    H, W = mask.shape
    best_a, best_s = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + step, step):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0)
        r = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST)
        score = float(np.sum(np.diff(r.sum(axis=1, dtype=np.float64)) ** 2))
        if score > best_s:
            best_s, best_a = score, a
    rotated = crop if abs(best_a) < min_apply else _rotate_expand(crop, best_a, border=(0, 0, 0))
    return _recrop_to_text(rotated), best_a


def _rows_of(boxes, tol):
    """Group boxes into text rows by vertical centre, each sorted left->right."""
    rows = []
    for b in sorted(boxes, key=lambda b: b[1] + b[3] / 2):
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
    return rows


def _row_text(mask, row, target_h=110):
    """OCR a whole number row.

    These plates are painted in an ordinary bold face rather than FE-Schrift, so
    per-glyph template matching is weak on them while Tesseract reads the row
    outright -- but only once the glyphs are scaled DOWN to roughly target_h;
    at full plate resolution the strokes are too heavy and it drops characters.
    """
    x0 = min(b[0] for b in row); y0 = min(b[1] for b in row)
    x1 = max(b[0] + b[2] for b in row); y1 = max(b[1] + b[3] for b in row)
    pad = 20
    band = cv2.bitwise_not(mask[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad])

    med_h = float(np.median([b[3] for b in row]))
    s = target_h / med_h
    if s < 1.0:
        band = cv2.resize(band, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    band = cv2.copyMakeBorder(band, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)

    for psm in (8, 13, 7):
        t = pytesseract.image_to_string(
            band, config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist={NUMBER_WHITELIST}")
        t = "".join(c for c in t.upper() if c.isalnum())
        if len(t) == len(row):        # agrees with the segmentation -> trust it
            return t
    return ""


def read_translit_plate(image, tpl, row_tol=0.10, min_score=0.4, read_header=True):
    """Read one plate crop.

    Returns (info, annotated_image) with info =
      {"header", "province", "code", "lot", "syllable", "devanagari",
       "serial", "number", "plate"}.
    "number" is the raw recognised rows, so plate_db.save_reading still works.
    """
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        raise FileNotFoundError(image)

    if img.shape[1] < 1600:
        s = 1600 / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask, polarity = _ink_mask(img)
    comps = _charlike(mask, H, W)
    blank = {"header": "", "province": "", "code": None, "lot": "", "syllable": "",
             "devanagari": "", "serial": "", "number": [], "plate": ""}
    if not comps:
        return blank, img

    # The number block is the tall text; the header band is the short text above it.
    max_h = max(b[3] for b in comps)
    main = _largest_cluster([b for b in comps if b[3] > 0.55 * max_h])
    if not main:
        return blank, img

    # Per-glyph pass: rejects junk (rivets, debris) and gives a per-character
    # fallback label. FE templates first, CNN for the ones they are unsure about.
    kept = []
    for (x, y, bw, bh) in main:
        q = _norm(cv2.bitwise_not(mask[y:y + bh, x:x + bw]))
        ch, sc = _match_flex(q, tpl)
        if sc < 0.30:                       # not a character -- screw, rivet, debris
            continue
        if sc < 0.55:
            cnn = classify(q)
            if cnn is not None:
                ch, sc = cnn
        kept.append((x, y, bw, bh, ch))
    if not kept:
        return blank, img

    char_of = {b[:4]: b[4] for b in kept}
    rows = _rows_of([b[:4] for b in kept], row_tol * H)

    vis = img.copy()
    number_rows = []
    for r in rows:
        # Row-level OCR is the primary reader here; the per-glyph labels stand in
        # only when Tesseract disagrees with the segmentation (or reads nothing).
        text = _row_text(mask, r)
        chars = list(text) if len(text) == len(r) else [char_of[b] for b in r]
        for ch, box in zip(chars, r):
            x, y, bw, bh = box
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 180, 0), 3)
            cv2.putText(vis, ch, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 180, 0), 3)
        number_rows.append("".join(chars))

    # Header band: short text sitting above the number, within its x-span.
    header = ""
    if read_header:
        top = min(b[1] for b in main)
        mx0 = min(b[0] for b in main)
        mx1 = max(b[0] + b[2] for b in main)
        hc = [b for b in comps if b[1] + b[3] <= top and b[3] < 0.6 * max_h
              and b[0] >= mx0 - 30 and b[0] + b[2] <= mx1 + 30]
        hc = _largest_cluster(hc)
        if hc:
            hx0 = min(b[0] for b in hc); hy0 = min(b[1] for b in hc)
            hx1 = max(b[0] + b[2] for b in hc); hy1 = max(b[1] + b[3] for b in hc)
            px, py = int(0.06 * (hx1 - hx0)) + 15, 12
            band = gray[max(0, hy0 - py):hy1 + py, max(0, hx0 - px):min(W, hx1 + px)]
            band = cv2.resize(band, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            flag = cv2.THRESH_BINARY_INV if polarity == "light" else cv2.THRESH_BINARY
            bt = cv2.threshold(band, 0, 255, flag + cv2.THRESH_OTSU)[1]
            bt = cv2.copyMakeBorder(bt, 20, 40, 20, 40, cv2.BORDER_CONSTANT, value=255)
            header = pytesseract.image_to_string(
                bt, config="--oem 3 --psm 7 -c tessedit_char_whitelist=" + HEADER_WHITELIST
            ).strip()
            cv2.rectangle(vis, (hx0 - px, hy0 - py), (hx1 + px, hy1 + py), (200, 120, 0), 2)

    province, code, _ = snap_province(header)
    num = parse_number(number_rows)
    plate = format_plate(province, code, num)

    info = {"header": (province + f" PRADESH-{code:02d}" if province and code is not None
                       else province), "province": province, "code": code,
            "number": number_rows, "plate": plate, **num}
    cv2.putText(vis, plate, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 120, 0), 2)
    return info, vis


def load_templates():
    return build_templates(ensure_font(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "FE-FONT.ttf")))
