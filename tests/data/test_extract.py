"""The #23 extraction test run: fixture rows are valid pairs and reruns are stable.

The committed fixture is the first 25 rows of the Kumārasaṃbhava chapter 1
extraction, in file order. When the raw DCS dump is on disk (data/ is not
committed), the full extraction also runs and must reproduce it byte for byte.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sandhi_probe.data.extract import (
    FIXTURE_PATH,
    RAW_DIR,
    RELATIVE_RAW,
    Tok,
    classify_rule,
    extract_rows,
    has_midword_change,
    junction_for_pair,
    junction_from_parts,
    parse_conllu,
)
from sandhi_probe.data.schema import Pair, RuleClass, RuleId

FULL_RUN = RAW_DIR / RELATIVE_RAW


def rows_from(path: Path) -> list[Pair]:
    return [
        Pair.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


@pytest.fixture(name="fixture_rows")
def fixture_rows() -> list[Pair]:
    return rows_from(FIXTURE_PATH)


def test_fixture_exists() -> None:
    assert FIXTURE_PATH.is_file(), "run the extraction once to (re)generate the fixture"


def test_fixture_rows_validate(fixture_rows: list[Pair]) -> None:
    assert len(fixture_rows) == 25
    assert len({row.id for row in fixture_rows}) == 25
    assert all(row.encoding == "slp1" for row in fixture_rows)
    assert all(row.alt_splits == [] for row in fixture_rows)  # DCS gives one analysis per token


def rebuild(row: Pair) -> str:
    """The #24 check: fused must be its parts with only its own junctions applied."""
    out = ""
    for i, part in enumerate(row.split_parts):
        if i > 0:
            part = part.removeprefix(row.junctions[i - 1].underlying_right)
        if i < len(row.junctions):
            junction = row.junctions[i]
            part = part.removesuffix(junction.underlying_left) + junction.surface
        out += part
    return out


def test_fixture_invariants(fixture_rows: list[Pair]) -> None:
    """The checks #24 will run, applied to the committed rows."""
    for row in fixture_rows:
        assert len(row.junctions) == len(row.split_parts) - 1
        for junction in row.junctions:
            start, end = junction.span
            assert 0 <= start <= end <= len(row.fused)
            assert row.fused[start:end] == junction.surface
            if junction.cut is not None:
                assert start <= junction.cut <= end
                assert 0 < junction.cut < len(row.fused)
        assert rebuild(row).replace(" ", "") == row.fused.replace(" ", "")
        assert row.is_control == all(j.boundary_level == "intact" for j in row.junctions)
        assert row.split_assignment in ("train", "dev", "test")
        assert row.frequency >= 1


def test_fixture_known_rows(fixture_rows: list[Pair]) -> None:
    """Hand-checked rows from verse 1 (devatātmā, himAlayo nāma, ...)."""
    by_id = {row.id: row for row in fixture_rows}

    row = by_id["dcs-2737695"]  # devatā + ātmā -> devatātmā
    assert row.fused == "devatAtmA"
    assert row.split_parts == ["devatA", "AtmA"]
    junction = row.junctions[0]
    assert junction.boundary_level == "merged"
    assert junction.cut is None and junction.surface == "A"
    assert (junction.underlying_left, junction.underlying_right) == ("A", "A")

    row = by_id["dcs-2737699"]  # naga + adhirājaḥ -> nagādhirājaḥ
    junction = row.junctions[0]
    assert junction.boundary_level == "merged"
    assert junction.rule_ids == [RuleId.SAVARNA_DIRGHA]  # a + a -> ā

    row = by_id["dcs-2737697"]  # himālayaḥ + nāma -> himAlayo nAma
    assert row.fused == "himAlayo nAma"
    assert row.split_parts == ["himAlayaH", "nAma"]
    junction = row.junctions[0]
    assert junction.rule_class == RuleClass.VISARGA
    assert junction.rule_ids == [RuleId.VISARGA_O]  # aH -> o before a voiced consonant

    # sampAdayitrIm + SiKarEH -> sampAdayitrIM SiKarEH: m -> M is consonant,
    # and the r that SiKarEH gets from the *next* word is not in this row
    row = by_id["dcs-2738216"]
    assert row.fused == "sampAdayitrIM SiKarEH"
    assert row.split_parts == ["sampAdayitrIm", "SiKarEH"]
    junction = row.junctions[0]
    assert junction.rule_class == RuleClass.CONSONANT
    assert (junction.underlying_left, junction.surface) == ("m", "M")
    assert rebuild(row).replace(" ", "") == row.fused.replace(" ", "")


def test_visarga_rules() -> None:
    """#47's keys, including the ones that only appear outside the fixture sample."""
    assert classify_rule("aH", "", "o", 1) == (RuleClass.VISARGA, [RuleId.VISARGA_O])
    assert classify_rule("aH", "a", "o '", 1) == (
        RuleClass.VISARGA,
        [RuleId.VISARGA_O_AVAGRAHA],
    )
    assert classify_rule("H", "", "", 1) == (RuleClass.VISARGA, [RuleId.VISARGA_DROP])
    assert classify_rule("H", "", "s", 1) == (RuleClass.VISARGA, [RuleId.VISARGA_S])
    # H -> r has no key in the table yet: classed, but left untagged on purpose
    assert classify_rule("H", "", "r", 1) == (RuleClass.VISARGA, [])


def test_anusvara_is_consonant() -> None:
    """m -> M is a consonant change, not a vowel one (review on #54)."""
    assert classify_rule("m", "", "M", 1) == (RuleClass.CONSONANT, [])


def test_far_end_change_is_reverted() -> None:
    """A word's change caused by its other neighbour stays out of this row."""
    left = Tok(no=1, form="sampAdayitrIM", unsandhied="sampAdayitrIm", reconstructed=True, occ_id=1)
    right = Tok(no=2, form="SiKarEr", unsandhied="SiKarEH", reconstructed=True, occ_id=2)
    found = junction_for_pair(left, right)
    assert found is not None
    junction, fused = found
    assert fused == "sampAdayitrIM SiKarEH"
    assert (junction.underlying_left, junction.surface) == ("m", "M")
    assert not has_midword_change(left, right)


def test_midword_change_is_skipped() -> None:
    """DCS's uttarAyAm vs printed uttarasyAM cannot be rebuilt: the pair is refused."""
    left = Tok(no=1, form="asty", unsandhied="asti", reconstructed=True, occ_id=1)
    right = Tok(no=2, form="uttarasyAM", unsandhied="uttarAyAm", reconstructed=True, occ_id=2)
    assert has_midword_change(left, right)


def test_merged_vowel_coalescence() -> None:
    """sītā + udakam -> sītodakam: one surface vowel shared by both words."""
    junctions = junction_from_parts(["sItA", "udakam"], "sItodakam")
    assert junctions is not None and len(junctions) == 1
    junction = junctions[0]
    assert junction.boundary_level == "merged"
    assert junction.cut is None and junction.surface == "o"
    assert (junction.underlying_left, junction.underlying_right) == ("A", "u")
    assert junction.rule_ids == [RuleId.GUNA]  # a + u -> o


def test_word_final_voicing() -> None:
    """tat + api -> tadapi: unvoiced stop becomes voiced at the join."""
    junctions = junction_from_parts(["tat", "api"], "tadapi")
    assert junctions is not None and len(junctions) == 1
    junction = junctions[0]
    assert junction.boundary_level == "changed"
    assert junction.cut == 3 and junction.surface == "d"
    assert junction.rule_ids == [RuleId.WORD_FINAL_VOICING]
    assert junction.rule_class == RuleClass.CONSONANT


def test_refuses_adjacent_changes() -> None:
    """Two joins changing back to back cannot be attributed: the fusion is skipped."""
    assert junction_from_parts(["a", "dfzwvA", "api"], "adfzwvApi") is None


@pytest.mark.skipif(
    not FULL_RUN.is_file(), reason="raw DCS dump not on disk (data/ is not committed)"
)
def test_full_run_matches_fixture(tmp_path: Path) -> None:
    """The committed fixture is the first 25 rows of a rerun, byte for byte."""

    def run(out: Path) -> str:
        sents, stats = parse_conllu(FULL_RUN)
        rows = extract_rows(sents, stats)
        out.write_text(
            "".join(json.dumps(row.model_dump(), ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return out.read_text(encoding="utf-8")

    first = run(tmp_path / "first.jsonl")
    second = run(tmp_path / "second.jsonl")
    assert first == second, "extraction is not deterministic"
    first_lines = first.splitlines()
    assert len(first_lines) == 278
    fixture_lines = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
    assert first_lines[: len(fixture_lines)] == fixture_lines
