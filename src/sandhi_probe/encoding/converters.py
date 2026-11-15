"""Convert between IAST and SLP1, the canonical encoding.

Two source encodings are accepted by `to_canonical`:
- "iast": strict IAST. `from_canonical` writes this too.
- "dcs_iast": the transcription in the DCS Rigveda file. It is IAST except that it writes some
  letters as ASCII capitals (`T`, `L`, `Q`, `W`) and uses `ṁ`, which the library would read as the
  wrong sound (`T` -> t, `L` -> l). `normalize_dcs_iast` first rewrites them as the IAST letters
  they stand for. A strict-IAST string with a real capital `T`, `L`, `Q` or `W` would be rewritten
  too, so use "iast" for anything that is not DCS.

Two letters need care with `indic-transliteration`:
- Vedic ḷ (ळ, DCS `L`) must become `ḻ` (U+1E3B). With `ḷ` (U+1E37) the library reads the vowel ऌ
  and `īḷe` comes out as `Ixe`.
- `ṁ` is left unconverted by the library, so DCS's `ṁ` is rewritten as `ṃ`.

Known gap: 7 DCS lines end a word with `*`. It has no SLP1 equivalent, so it is passed through;
`non_slp1_chars` reports it.
"""

from __future__ import annotations

import string
import unicodedata

from indic_transliteration import sanscript

CANONICAL = "slp1"

ENCODINGS = {"iast": sanscript.IAST}
# What to_canonical accepts. "dcs_iast" is a source only: nothing writes DCS's markers back out.
SOURCES = (*ENCODINGS, "dcs_iast")

# DCS marker -> the IAST letter it stands for. Digraphs come before their first letter.
DCS_MARKERS = (
    ("Th", "ṭh"),
    ("T", "ṭ"),
    ("Lh", "ḻh"),
    ("L", "ḻ"),
    ("Q", "ḷ"),
    ("W", "ṃ"),
    ("ṁ", "ṃ"),
)

# ASCII letters, `|` (SLP1 for ळ्ह), the avagraha and the word separator.
SLP1_CHARS = frozenset(string.ascii_letters + "|' ")


def get_scheme(name: str) -> str:
    try:
        return ENCODINGS[name]
    except KeyError:
        raise ValueError(f"unsupported encoding {name!r}; supported: {sorted(ENCODINGS)}") from None


def normalize_dcs_iast(s: str) -> str:
    """Rewrite DCS's ASCII letter markers as the IAST letters they stand for (NFC)."""
    s = unicodedata.normalize("NFC", s)
    for marker, letter in DCS_MARKERS:
        s = s.replace(marker, letter)
    return unicodedata.normalize("NFC", s)


def to_canonical(s: str, src: str) -> str:
    """Convert `s` from encoding `src` ("iast" or "dcs_iast") to SLP1."""
    if src not in SOURCES:
        raise ValueError(f"unsupported encoding {src!r}; supported: {sorted(SOURCES)}")
    if src == "dcs_iast":
        s, src = normalize_dcs_iast(s), "iast"
    return sanscript.transliterate(unicodedata.normalize("NFC", s), get_scheme(src), sanscript.SLP1)


def from_canonical(s: str, dst: str) -> str:
    """Convert SLP1 text `s` to encoding `dst`, in NFC."""
    return unicodedata.normalize("NFC", sanscript.transliterate(s, sanscript.SLP1, get_scheme(dst)))


def non_slp1_chars(s: str) -> set[str]:
    """Characters in `s` that are not SLP1 letters, `|`, `'` or a space (unconverted input)."""
    return {c for c in s if c not in SLP1_CHARS}
