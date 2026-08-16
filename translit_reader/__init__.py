"""Reader for Nepali plates written in English but transliterated from Devanagari."""
from translit_reader.reader import read_translit_plate, load_templates
from translit_reader.translit_tokens import (
    PROVINCES, SYLLABLES, snap_province, snap_syllable, parse_number, format_plate,
)

__all__ = ["read_translit_plate", "load_templates", "PROVINCES", "SYLLABLES",
           "snap_province", "snap_syllable", "parse_number", "format_plate"]
