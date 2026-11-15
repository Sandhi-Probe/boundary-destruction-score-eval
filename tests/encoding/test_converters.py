"""IAST <-> SLP1 converters on the DCS Rigveda text (#20)."""

import csv

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
    assert to_canonical(source, "iast") == expected


@pytest.mark.parametrize(("source", "expected"), KNOWN_LINES + MARKERS)
def test_output_is_only_slp1(source: str, expected: str) -> None:
    assert non_slp1_chars(to_canonical(source, "iast")) == set()


@pytest.mark.parametrize(("source", "_"), KNOWN_LINES + MARKERS)
def test_iast_round_trip(source: str, _: str) -> None:
    """Back to IAST gives the source with its markers written as the letters they stand for."""
    assert from_canonical(to_canonical(source, "iast"), "iast") == normalize_dcs_iast(source)


@pytest.mark.parametrize("slp1", [expected for _, expected in KNOWN_LINES + MARKERS])
def test_slp1_round_trip(slp1: str) -> None:
    assert to_canonical(from_canonical(slp1, "iast"), "iast") == slp1


def test_vedic_l_is_not_the_vowel() -> None:
    assert from_canonical("ILe", "iast") == "īḻe"
    assert "x" not in to_canonical("īLe", "iast")


def test_unconverted_characters_are_reported() -> None:
    assert non_slp1_chars(to_canonical("rodasī*", "iast")) == {"*"}


@pytest.mark.parametrize("fn", [to_canonical, from_canonical])
def test_unsupported_encoding(fn) -> None:
    with pytest.raises(ValueError, match="unsupported encoding"):
        fn("agnim", "devanagari")


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
        if from_canonical(to_canonical(s, "iast"), "iast") != normalize_dcs_iast(s)
    ]
    assert failures == []

    unconverted = {c for s in sample for c in non_slp1_chars(to_canonical(s, "iast"))}
    assert unconverted <= {"*"}
