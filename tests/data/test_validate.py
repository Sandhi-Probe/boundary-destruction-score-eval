"""Each check in the validation suite fires on a broken row and stays quiet on a valid fixture."""

import pytest

from sandhi_probe.data.schema import Junction, Pair
from sandhi_probe.data.validate import validate

TADAPI = Junction(
    span=(2, 3),
    cut=3,
    underlying_left="t",
    underlying_right="",
    surface="d",
    rule_class="consonant",
    rule_ids=["word_final_voicing"],
    type="external",
)

INTACT = Junction(
    span=(4, 4),
    cut=4,
    underlying_left="",
    underlying_right="",
    surface="",
    rule_class="none",
    type="compound",
)

MERGED = Junction(
    span=(3, 4),
    cut=None,
    underlying_left="A",
    underlying_right="u",
    surface="o",
    rule_class="vowel",
    rule_ids=["guna"],
    type="external",
)


def make_pair(
    fused: str, split_parts: list[str], junction: Junction, n: int = 1, **overrides
) -> Pair:
    fields = dict(
        id=f"dcs-{n:06d}",
        fused=fused,
        split_parts=split_parts,
        junctions=[junction],
        source_corpus="DCS",
        source_ref=f"dcs-2022/text-1/sent-{n}/tok-1",
        sentence_id=f"dcs-2022/text-1/sent-{n}",
        split_provenance="gold",
        frequency=1,
        split_assignment="train",
    )
    return Pair(**(fields | overrides))


BASE = make_pair("tadapi", ["tat", "api"], TADAPI)
VALID = [
    BASE,
    make_pair("rAmagfham", ["rAma", "gfham"], INTACT, 2, is_control=True),
    make_pair("sItodakam", ["sItA", "udakam"], MERGED, 3),
    make_pair("rAma gfham", ["rAma", "gfham"], INTACT, 4, is_control=True),
]

SAME_SENTENCE = {"id": "dcs-000002", "source_ref": "dcs-2022/text-1/sent-1/tok-2"}
OTHER_SENTENCE = {
    "id": "dcs-000002",
    "sentence_id": "dcs-2022/text-1/sent-2",
    "source_ref": "dcs-2022/text-1/sent-2/tok-1",
}


def hits(pairs: list[Pair]) -> set[tuple[int, str]]:
    return {(issue.check, issue.level) for issue in validate(pairs)}


def test_valid_fixture_has_no_errors() -> None:
    assert not [issue for issue in validate(VALID) if issue.level == "error"]


@pytest.mark.parametrize(
    ("check", "changes"),
    [
        (1, {"split_parts": ["tat", ""]}),
        (2, {"junctions": []}),
        (3, {"junctions": [TADAPI.model_copy(update={"span": (5, 7)})]}),
        (4, {"junctions": [TADAPI.model_copy(update={"span": (1, 2)})]}),
        (5, {"junctions": [TADAPI.model_copy(update={"span": (5, 6)}), TADAPI]}),
        (6, {"junctions": [TADAPI.model_copy(update={"underlying_left": "s"})]}),
        (7, {"split_parts": ["tat", "iti"]}),
        (8, {"junctions": [TADAPI.model_copy(update={"cut": 5})]}),
        (
            9,
            {
                "fused": "rAmagfham",
                "split_parts": ["rAma", "gfham"],
                "junctions": [INTACT.model_copy(update={"cut": None})],
            },
        ),
        (10, {"junctions": [TADAPI.model_copy(update={"rule_class": "none"})]}),
        (11, {"junctions": [TADAPI.model_copy(update={"rule_ids": ["guna"]})]}),
        (12, {"is_control": True}),
        (13, {"alt_splits": [["tat", "api"]]}),
        (14, {"frequency": 0}),
        (16, {"source_ref": "uoh/text-1/sent-1"}),
        (23, {"fused": "tadapi§"}),
    ],
)
def test_row_check_fires(check: int, changes: dict) -> None:
    assert (check, "error") in hits([BASE.model_copy(update=changes)])


def test_check_7_ignores_spaces_in_fused() -> None:
    assert (7, "error") not in hits([VALID[3]])


@pytest.mark.parametrize(
    ("check", "level", "other_changes"),
    [
        (17, "error", {"source_ref": "dcs-2022/text-1/sent-1/tok-2", "fused": "tadapi2"}),
        (18, "error", {"id": "dcs-000002"}),
        (19, "error", {**SAME_SENTENCE, "split_assignment": "test"}),
        (20, "warning", {**OTHER_SENTENCE, "split_assignment": "test"}),
    ],
)
def test_dataset_check_fires(check: int, level: str, other_changes: dict) -> None:
    assert (check, level) in hits([BASE, BASE.model_copy(update=other_changes)])


def test_split_size_and_rule_class_coverage_warn() -> None:
    assert {(21, "warning"), (22, "warning")} <= hits([BASE])
