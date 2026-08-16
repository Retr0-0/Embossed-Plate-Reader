"""Vocabulary and grammar for romanized (Devanagari-transliterated) Nepali plates.

These plates are written in English letters, but the letters spell out Devanagari
sounds:  BAGMATI PRADESH-01 / 030 CHA 7911   where CHA is the roman spelling of
the Devanagari class letter च.  The number therefore is NOT the
"letters-then-digits" shape the FE-Schrift reader assumes -- it is

    <lot digits> <class syllable> <4-digit serial>

so it needs its own vocabulary and its own confusion repair.
"""
import difflib

# Province name -> its number, as printed in the header band (BAGMATI PRADESH-01).
PROVINCES = {
    "KOSHI": 1, "MADHESH": 2, "BAGMATI": 3, "GANDAKI": 4,
    "LUMBINI": 5, "KARNALI": 6, "SUDURPASHCHIM": 7,
}

# Roman spelling -> the Devanagari letter it transliterates. This is the whole
# point of the module: the plate is English text standing in for Devanagari.
SYLLABLES = {
    "KA": "क", "KHA": "ख", "GA": "ग", "GHA": "घ", "NGA": "ङ",
    "CHA": "च", "CHHA": "छ", "JA": "ज", "JHA": "झ",
    "TA": "त", "THA": "थ", "DA": "द", "DHA": "ध", "NA": "न",
    "PA": "प", "PHA": "फ", "BA": "ब", "BHA": "भ", "MA": "म",
    "YA": "य", "RA": "र", "LA": "ल", "WA": "व",
    "SHA": "श", "SA": "स", "HA": "ह",
    "SE": "से", "ME": "मे", "PRA": "प्र", "SU": "सु", "KO": "को",
}

# Look-alikes between the FE glyph shapes. Applied in opposite directions
# depending on which slot of the plate a character landed in.
_D2L = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
_L2D = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "7",
        "T": "7", "S": "5", "G": "6", "B": "8", "A": "4"}

SERIAL_LEN = 4


def snap_province(text, threshold=0.55):
    """Snap mangled header letters to the nearest province name.
    Returns (province, code, score); province is "" when nothing matches."""
    letters = "".join(c for c in text.upper() if c.isalpha())
    letters = letters.replace("PRADESH", "")
    digits = "".join(c for c in text if c.isdigit())
    code = int(digits[-2:]) if digits else None
    if not letters:
        return "", code, 0.0
    best, score = "", 0.0
    for p in PROVINCES:
        r = difflib.SequenceMatcher(None, letters, p).ratio()
        if r > score:
            score, best = r, p
    return (best if score >= threshold else ""), code, score


def snap_syllable(text, threshold=0.5):
    """Snap a candidate letter group to the closest romanized Devanagari
    syllable. Returns (syllable, score); syllable is "" below threshold."""
    q = "".join(c for c in text.upper() if c.isalpha())
    if not q:
        return "", 0.0
    best, score = "", 0.0
    for s in SYLLABLES:
        r = difflib.SequenceMatcher(None, q, s).ratio()
        if r > score:
            score, best = r, s
    return (best if score >= threshold else q), score


def parse_number(rows):
    """Turn the recognised glyph rows into {lot, syllable, devanagari, serial}.

    The rows are read top-to-bottom, left-to-right and concatenated, so the
    plate in the photo arrives as "030CHA" + "7911". The last SERIAL_LEN
    characters are the serial (force digits), and the remaining prefix splits
    into lot digits + class syllable -- the split point is chosen as whichever
    tail spells a real syllable best.
    """
    joined = "".join(rows).upper()
    if len(joined) <= SERIAL_LEN:
        return {"lot": "", "syllable": "", "devanagari": "", "serial": joined}

    serial = "".join(_L2D.get(c, c) if c.isalpha() else c
                     for c in joined[-SERIAL_LEN:])
    prefix = joined[:-SERIAL_LEN]

    # Longest first, so a tie goes to the longer syllable: "030CHA" must split as
    # 030 + CHA, not 030C + HA (both spell a real syllable, both score 1.0).
    best_k, best_score, best_syl = 0, 0.0, ""
    for k in range(min(4, len(prefix)), 1, -1):
        cand = "".join(_D2L.get(c, c) if c.isdigit() else c for c in prefix[-k:])
        syl, score = snap_syllable(cand)
        if syl in SYLLABLES and score > best_score:
            best_k, best_score, best_syl = k, score, syl

    if best_k:
        lot_raw, syllable = prefix[:-best_k], best_syl
    else:                       # no syllable recognised -- treat it all as lot
        lot_raw, syllable = prefix, ""
    lot = "".join(_L2D.get(c, c) if c.isalpha() else c for c in lot_raw)

    return {"lot": lot, "syllable": syllable,
            "devanagari": SYLLABLES.get(syllable, ""), "serial": serial}


def format_plate(province, code, num):
    """Human-readable plate string, e.g. 'BAGMATI PRADESH-01 | 030 CHA 7911'."""
    head = ""
    if province:
        head = province + " PRADESH" + (f"-{code:02d}" if code is not None else "")
    body = " ".join(p for p in (num["lot"], num["syllable"], num["serial"]) if p)
    return (head + " | " + body) if head else body
