# FE-Schrift(Nepali Embossed) License Plate Reader

Reads embossed license plates that use the FE-Schrift font (e.g. Nepali plates)
by matching each character against rendered FE-Schrift glyph templates instead
of relying on general-purpose OCR.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python main.py path/to/plate.png            # red characters (default)
python main.py path/to/plate.png --color dark --save out.png
```
