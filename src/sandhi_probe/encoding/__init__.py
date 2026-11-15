"""Conversion between Devanagari / IAST / SLP1 / HK / WX and the canonical encoding.

Every other module takes canonical-encoding text only. Converters: #20 · choice: ADR-001 (#19)
"""

from sandhi_probe.encoding.converters import (
    CANONICAL,
    from_canonical,
    non_slp1_chars,
    normalize_dcs_iast,
    to_canonical,
)

__all__ = ["CANONICAL", "from_canonical", "non_slp1_chars", "normalize_dcs_iast", "to_canonical"]
