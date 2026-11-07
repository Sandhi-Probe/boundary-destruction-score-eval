"""Checks the minimal-pair schema on the worked examples from docs/data-format.md."""

import pytest
from pydantic import ValidationError

from sandhi_probe.data.schema import RULES, BoundaryLevel, Junction, Pair, RuleId


def make_pair(fused: str, split_parts: list[str], junction: Junction, **overrides) -> Pair:
    fields = dict(
        id="dcs-000001",
        fused=fused,
        split_parts=split_parts,
        junctions=[junction],
        source_corpus="DCS",
        source_ref="dcs-2022/text-1/sent-1/tok-1",
        sentence_id="dcs-2022/text-1/sent-1",
        split_provenance="gold",
        frequency=1,
        split_assignment="train",
    )
    return Pair(**(fields | overrides))


@pytest.mark.parametrize(
    ("span", "cut", "left", "right", "surface", "rule_class", "expected"),
    [
        ((4, 4), 4, "", "", "", "none", BoundaryLevel.INTACT),  # rAma + gfham -> rAmagfham
        ((2, 3), 3, "t", "", "d", "consonant", BoundaryLevel.CHANGED),  # tat + api -> tadapi
        ((4, 4), 4, "H", "", "", "visarga", BoundaryLevel.CHANGED),  # rAmaH + udakam
        ((3, 4), None, "A", "u", "o", "vowel", BoundaryLevel.MERGED),  # sItA + udakam
    ],
)
def test_boundary_level(
    span: tuple[int, int],
    cut: int | None,
    left: str,
    right: str,
    surface: str,
    rule_class: str,
    expected: BoundaryLevel,
) -> None:
    junction = Junction(
        span=span,
        cut=cut,
        underlying_left=left,
        underlying_right=right,
        surface=surface,
        rule_class=rule_class,
        type="external",
    )
    assert junction.boundary_level == expected


def test_every_rule_id_is_in_rules_table() -> None:
    assert set(RULES) == set(RuleId)


def test_rejects_unknown_split() -> None:
    junction = Junction(
        span=(2, 3),
        cut=3,
        underlying_left="t",
        underlying_right="",
        surface="d",
        rule_class="consonant",
        rule_ids=["word_final_voicing"],
        type="external",
    )
    assert make_pair("tadapi", ["tat", "api"], junction).split_assignment == "train"
    with pytest.raises(ValidationError):
        make_pair("tadapi", ["tat", "api"], junction, split_assignment="holdout")
