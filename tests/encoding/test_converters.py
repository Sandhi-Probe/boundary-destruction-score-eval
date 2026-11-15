"""IAST <-> SLP1 converters on the DCS Rigveda text (#20)."""

import csv
import unicodedata

import pytest

from sandhi_probe import paths
from sandhi_probe.encoding import (
    from_canonical,
    non_slp1_chars,
    normalize_dcs_iast,
    to_canonical,
)

DCS_FILE = paths.RAW_DIR / "dcs" / "rigveda" / "pada-and-analysis.dat"

# Source line (DCS transcription) -> expected SLP1.
KNOWN_LINES = [
    ("agnim īLe purohitam", "agnim ILe purohitam"),  # RV 1.1.1a
    ("yajñam vaṣTu dhiyāvasuḥ", "yajYam vazwu DiyAvasuH"),  # RV 1.3.10c
]

# One case per DCS marker. The Th, Lh, Q and W inputs are made up; the others are from the corpus.
MARKERS = [
    ("kṛṣTayaḥ", "kfzwayaH"),  # T  -> ṭ -> w
    ("aThā", "aWA"),  # Th -> ṭh -> W
    ("īLe", "ILe"),  # L  -> Vedic ḷ -> L (not the vowel, which would be x)
    ("aLha", "a|a"),  # Lh -> Vedic ḷh -> |
    ("kQptaḥ", "kxptaH"),  # Q  -> vocalic ḷ -> x
    ("aWsa", "aMsa"),  # W  -> ṃ -> M
    ("araṁkṛtāḥ", "araMkftAH"),  # ṁ  -> ṃ -> M
]


@pytest.mark.parametrize(("source", "expected"), KNOWN_LINES + MARKERS)
def test_to_canonical(source: str, expected: str) -> None:
    assert to_canonical(source, "dcs_iast") == expected


@pytest.mark.parametrize(("source", "expected"), KNOWN_LINES + MARKERS)
def test_output_is_only_slp1(source: str, expected: str) -> None:
    assert non_slp1_chars(to_canonical(source, "dcs_iast")) == set()


@pytest.mark.parametrize(("source", "_"), KNOWN_LINES + MARKERS)
def test_iast_round_trip(source: str, _: str) -> None:
    """Back to IAST gives the source with its markers written as the letters they stand for."""
    assert from_canonical(to_canonical(source, "dcs_iast"), "iast") == normalize_dcs_iast(source)


@pytest.mark.parametrize("slp1", [expected for _, expected in KNOWN_LINES + MARKERS])
def test_slp1_round_trip(slp1: str) -> None:
    assert to_canonical(from_canonical(slp1, "iast"), "iast") == slp1


def test_strict_iast() -> None:
    assert to_canonical("yajñasya", "iast") == "yajYasya"
    assert to_canonical("agnim īḻe purohitam", "iast") == "agnim ILe purohitam"


@pytest.mark.parametrize(
    ("src", "composed"), [("iast", "yajñasya kṛṣṇaḥ"), ("dcs_iast", "kṛṣTayaḥ īLe")]
)
def test_decomposed_input_gives_the_same_result(src: str, composed: str) -> None:
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed
    assert to_canonical(decomposed, src) == to_canonical(composed, src)


def test_outputs_are_nfc() -> None:
    assert unicodedata.is_normalized("NFC", from_canonical("yajYasya kfzRaH ILe", "iast"))
    decomposed = unicodedata.normalize("NFD", "kṛṣTayaḥ")
    assert unicodedata.is_normalized("NFC", normalize_dcs_iast(decomposed))


def test_vedic_l_is_not_the_vowel() -> None:
    assert from_canonical("ILe", "iast") == "īḻe"
    assert "x" not in to_canonical("īLe", "dcs_iast")


def test_unconverted_characters_are_reported() -> None:
    assert non_slp1_chars(to_canonical("rodasī*", "dcs_iast")) == {"*"}


@pytest.mark.parametrize(
    ("fn", "name"), [(to_canonical, "devanagari"), (from_canonical, "dcs_iast")]
)
def test_unsupported_encoding(fn, name: str) -> None:
    with pytest.raises(ValueError, match="unsupported encoding"):
        fn("agnim", name)


@pytest.mark.skipif(not DCS_FILE.exists(), reason="DCS raw file not downloaded")
def test_dcs_sample_round_trips() -> None:
    """Every 8th DCS line (about 5k, all ten books): the round trip is exact, and the only
    unconverted character is the `*` that ends a word on a few lines."""
    with DCS_FILE.open(encoding="utf-8", newline="") as f:
        lines = [r["text"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    sample = lines[::8]
    assert len(sample) > 4_900

    failures = [
        s
        for s in sample
        if from_canonical(to_canonical(s, "dcs_iast"), "iast") != normalize_dcs_iast(s)
    ]
    assert failures == []

    unconverted = {c for s in sample for c in non_slp1_chars(to_canonical(s, "dcs_iast"))}
    assert unconverted <= {"*"}
