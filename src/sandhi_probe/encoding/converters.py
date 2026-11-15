"""IAST <-> SLP1 (the canonical encoding), for the DCS Rigveda text.

"iast" here means the transcription DCS ships, not strict IAST: it writes some letters as ASCII
capitals, and the library would turn those into the wrong sound (`T` -> t, `L` -> l, ...). So
`to_canonical` first rewrites them as the IAST letters they stand for (see `normalize_dcs_iast`).
A strict-IAST string with a legitimate capital `T`, `L`, `Q` or `W` would be rewritten too.

Two letters need care with `indic-transliteration`:
- Vedic ḷ (ळ, DCS `L`) must become `ḻ` (U+1E3B). With `ḷ` (U+1E37) the library reads the vowel ऌ
  and `īḷe` comes out as `Ixe`.
- `ṁ` is left unconverted by the library, so it is rewritten as `ṃ`.

Known gap: 7 DCS lines end a word with `*`. It has no SLP1 equivalent, so it is passed through;
`non_slp1_chars` reports it.
"""

from __future__ import annotations

import string
import unicodedata

from indic_transliteration import sanscript

CANONICAL = "slp1"

_ENCODINGS = {"iast": sanscript.IAST}

# DCS marker -> the IAST letter it stands for. Digraphs come before their first letter.
_DCS_MARKERS = (
    ("Th", "ṭh"),
    ("T", "ṭ"),
    ("Lh", "ḻh"),
    ("L", "ḻ"),
    ("Q", "ḷ"),
    ("W", "ṃ"),
    ("ṁ", "ṃ"),
)

# ASCII letters, `|` (SLP1 for ळ्ह), the avagraha and the word separator.
_SLP1_CHARS = frozenset(string.ascii_letters + "|' ")


def _scheme(name: str) -> str:
    try:
        return _ENCODINGS[name]
    except KeyError:
        raise ValueError(
            f"unsupported encoding {name!r}; supported: {sorted(_ENCODINGS)}"
        ) from None


def normalize_dcs_iast(s: str) -> str:
    """Rewrite DCS's ASCII letter markers as the IAST letters they stand for (NFC)."""
    s = unicodedata.normalize("NFC", s)
    for marker, letter in _DCS_MARKERS:
        s = s.replace(marker, letter)
    return unicodedata.normalize("NFC", s)


def to_canonical(s: str, src: str) -> str:
    """Convert `s` from encoding `src` to SLP1."""
    scheme = _scheme(src)
    if src == "iast":
        s = normalize_dcs_iast(s)
    return sanscript.transliterate(s, scheme, sanscript.SLP1)


def from_canonical(s: str, dst: str) -> str:
    """Convert SLP1 text `s` to encoding `dst`, in NFC."""
    return unicodedata.normalize("NFC", sanscript.transliterate(s, sanscript.SLP1, _scheme(dst)))


def non_slp1_chars(s: str) -> set[str]:
    """Characters in `s` that are not SLP1 letters, `|`, `'` or a space (unconverted input)."""
    return {c for c in s if c not in _SLP1_CHARS}
