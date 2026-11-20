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
    extract_rows,
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


def test_fixture_invariants(fixture_rows: list[Pair]) -> None:
    """The checks #24 will run, applied to the committed rows."""
    for row in fixture_rows:
        assert len(row.junctions) == len(row.split_parts) - 1
        for junction in row.junctions:
            start, end = junction.span
            assert 0 <= start <= end <= len(row.fused)
            assert row.fused[start:end] == junction.surface
        assert row.is_control == all(j.boundary_level == "intact" for j in row.junctions)
        assert row.split_assignment in ("train", "dev", "test")


def test_fixture_known_rows(fixture_rows: list[Pair]) -> None:
    """Hand-checked rows from verse 1 (asty uttarasyāṃ ...)."""
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

    row = by_id["dcs-2737692"]  # asti + uttarāyām -> asty uttarasyāṃ
    assert row.fused == "asty uttarasyAM"
    assert row.split_parts == ["asti", "uttarAyAm"]
    junction = row.junctions[0]
    assert junction.boundary_level == "changed"
    assert junction.rule_ids == [RuleId.YAN]  # i -> y
    assert row.junctions[0].span == (3, 4) and row.junctions[0].cut == 4


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
    assert len(first_lines) == 283
    fixture_lines = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
    assert first_lines[: len(fixture_lines)] == fixture_lines
