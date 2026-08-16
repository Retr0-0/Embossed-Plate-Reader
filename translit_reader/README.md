# Transliterated (Devanagari → English) Plate Reader

Reads the newer Nepali embossed plates whose text is printed in English letters
but spells out Devanagari sounds:

```
BAGMATI PRADESH-01
   030 CHA
     7911
```

`CHA` is the roman spelling of the Devanagari class letter `च`. The reader
returns the syllable *and* the Devanagari letter it stands for.

## How it differs from the root reader

| | root `emb_plate_reader.py` | this folder |
|---|---|---|
| glyph colour | coloured ink on a light field | light ink embossed on a coloured field (polarity auto-detected per image) |
| number shape | `<letters><4 digits>` | `<lot digits><class syllable><4-digit serial>` |
| header | province word only | province word **+** province code (`PRADESH-01`) |

Detection (`plate_detector.py`), deskewing, the FE-Schrift glyph templates, the
CNN fallback (`glyph_cnn.pt`) and the SQLite log (`plates.db`) are all reused
from the root project — only the masking and the number grammar are new.

## Usage

Run from the repository root:

```bash
python -m translit_reader.main path/to/plate.jpg
python -m translit_reader.main plate.jpg --no-detect      # image is already a plate crop
python -m translit_reader.main plate.jpg --no-db          # don't log the reading
```

Annotated output lands in `translit_reader/recognized/`.

## Vocabulary

`translit_tokens.py` holds the two lookups the grammar leans on: the seven
provinces with their numbers, and the roman → Devanagari syllable table
(`KA क`, `KHA ख`, `CHA च`, `BA ब`, `PA प`, …). OCR output is snapped to the
nearest entry in each, so a misread `0HA` still resolves to `CHA`.
