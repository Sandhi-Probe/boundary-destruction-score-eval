"""Validation suite for the minimal-pair dataset (#24).

Not implemented yet: 15 (frequency_bucket vs frequency) waits for the cutoffs, and
24 (Devanagari round trip) is a stub until the encoding converters land (#20).
"""

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from itertools import pairwise
from typing import Literal, NamedTuple

from sandhi_probe.data.schema import RULES, BoundaryLevel, Pair, RuleClass

SLP1 = set("aAiIuUfFxXeEoOMH~kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzshL '")
DCS_SOURCE_REF = re.compile(r"dcs-\d{4}/text-\d+/sent-\d+/tok-\d+")
SPLIT_SHARE = {"train": 0.8, "dev": 0.1, "test": 0.1}
SPLIT_TOLERANCE = 0.05

Level = Literal["error", "warning"]


class Issue(NamedTuple):
    check: int
    level: Level
    message: str


def aligned(pair: Pair) -> bool:
    return len(pair.junctions) == len(pair.split_parts) - 1


# builds `fused` back from the parts (check 7)
def rebuild(pair: Pair) -> str:
    out = ""
    for i, part in enumerate(pair.split_parts):
        if i > 0:
            part = part.removeprefix(pair.junctions[i - 1].underlying_right)
        if i < len(pair.junctions):
            junction = pair.junctions[i]
            part = part.removesuffix(junction.underlying_left) + junction.surface
        out += part
    return out


# lists every text value for the SLP1 check
def text_fields(pair: Pair) -> Iterator[tuple[str, str]]:
    yield "fused", pair.fused
    for part in pair.split_parts:
        yield "split_parts", part
    for alt in pair.alt_splits:
        for part in alt:
            yield "alt_splits", part
    for junction in pair.junctions:
        yield "underlying_left", junction.underlying_left
        yield "underlying_right", junction.underlying_right
        yield "surface", junction.surface


# finds values that appear in more than one split
def split_spread(pairs: list[Pair], key: Callable[[Pair], str]) -> Iterator[tuple[str, list[str]]]:
    splits: defaultdict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        splits[key(pair)].add(pair.split_assignment)
    for value, found in splits.items():
        if len(found) > 1:
            yield value, sorted(found)


# Row checks


def parts_not_empty(pair: Pair) -> Iterator[str]:
    if any(not part.strip() for part in pair.split_parts):
        yield "split_parts has an empty or whitespace-only part"


def junction_count(pair: Pair) -> Iterator[str]:
    if not aligned(pair):
        yield f"{len(pair.junctions)} junctions for {len(pair.split_parts)} split_parts"


def span_inside_fused(pair: Pair) -> Iterator[str]:
    for junction in pair.junctions:
        start, end = junction.span
        if not 0 <= start <= end <= len(pair.fused):
            yield f"span {junction.span} is outside fused (length {len(pair.fused)})"


def span_matches_surface(pair: Pair) -> Iterator[str]:
    for junction in pair.junctions:
        start, end = junction.span
        if pair.fused[start:end] != junction.surface:
            found = pair.fused[start:end]
            yield f"fused[{start}:{end}] is {found!r}, surface is {junction.surface!r}"


def junctions_in_order(pair: Pair) -> Iterator[str]:
    for a, b in pairwise(pair.junctions):
        if a.span[1] > b.span[0]:
            yield f"spans {a.span} and {b.span} are out of order or overlap"


def underlying_matches_parts(pair: Pair) -> Iterator[str]:
    if not aligned(pair):
        return
    for i, junction in enumerate(pair.junctions):
        if not pair.split_parts[i].endswith(junction.underlying_left):
            yield f"{pair.split_parts[i]!r} does not end with {junction.underlying_left!r}"
        if not pair.split_parts[i + 1].startswith(junction.underlying_right):
            yield f"{pair.split_parts[i + 1]!r} does not start with {junction.underlying_right!r}"


def rebuilds_fused(pair: Pair) -> Iterator[str]:
    # spaces are dropped from fused: the parts do not say where a join crosses a space
    if aligned(pair) and rebuild(pair) != pair.fused.replace(" ", ""):
        yield f"parts rebuild to {rebuild(pair)!r}, fused is {pair.fused!r}"


def cut_inside_junction(pair: Pair) -> Iterator[str]:
    for junction in pair.junctions:
        start, end = junction.span
        if junction.cut is not None and not (
            start <= junction.cut <= end and 0 < junction.cut < len(pair.fused)
        ):
            yield f"cut {junction.cut} is not inside span {junction.span}"


def cut_set_when_intact(pair: Pair) -> Iterator[str]:
    for junction in pair.junctions:
        unchanged = junction.underlying_left + junction.underlying_right == junction.surface
        if junction.cut is None and unchanged:
            yield f"cut is null but nothing changed at span {junction.span}"


def nothing_changed_agrees(pair: Pair) -> Iterator[str]:
    for junction in pair.junctions:
        no_class = junction.rule_class == RuleClass.NONE
        no_rules = not junction.rule_ids
        intact = junction.boundary_level == BoundaryLevel.INTACT
        if not no_class == no_rules == intact:
            yield (
                f"at span {junction.span}: rule_class {junction.rule_class}, "
                f"rule_ids {[str(r) for r in junction.rule_ids]}, "
                f"boundary_level {junction.boundary_level}"
            )


def rule_ids_match_class(pair: Pair) -> Iterator[str]:
    for junction in pair.junctions:
        for rule_id in junction.rule_ids:
            group = RULES[rule_id].group
            if group != junction.rule_class:
                yield f"{rule_id} is a {group} rule, rule_class is {junction.rule_class}"


def control_is_intact(pair: Pair) -> Iterator[str]:
    if pair.is_control and any(
        j.boundary_level != BoundaryLevel.INTACT or j.rule_ids for j in pair.junctions
    ):
        yield "control row has a changed junction or rule ids"


def alt_splits_well_formed(pair: Pair) -> Iterator[str]:
    seen: list[list[str]] = []
    for alt in pair.alt_splits:
        if len(alt) < 2 or any(not part.strip() for part in alt):
            yield f"alt split {alt} needs at least 2 non-empty parts"
        if alt == pair.split_parts:
            yield f"alt split {alt} repeats split_parts"
        if alt in seen:
            yield f"alt split {alt} is listed twice"
        seen.append(alt)


def frequency_at_least_one(pair: Pair) -> Iterator[str]:
    if pair.frequency < 1:
        yield f"frequency is {pair.frequency}"


def ids_match_corpus(pair: Pair) -> Iterator[str]:
    prefix = pair.source_corpus.lower()
    if not pair.id.startswith(prefix):
        yield f"id does not start with {prefix!r}"
    if not pair.source_ref.startswith(prefix):
        yield f"source_ref {pair.source_ref!r} does not start with {prefix!r}"
    if pair.source_corpus == "DCS" and not DCS_SOURCE_REF.fullmatch(pair.source_ref):
        yield f"source_ref {pair.source_ref!r} is not dcs-YYYY/text-N/sent-N/tok-N"
    if not pair.source_ref.startswith(pair.sentence_id):
        yield f"source_ref {pair.source_ref!r} does not start with sentence_id {pair.sentence_id!r}"


def slp1_only(pair: Pair) -> Iterator[str]:
    for name, text in text_fields(pair):
        if bad := sorted(set(text) - SLP1):
            yield f"{name} {text!r} has non-SLP1 characters {bad}"


def devanagari_round_trip(pair: Pair) -> Iterator[str]:
    # stub until the converters land (#20): each text field must survive SLP1 -> Devanagari -> SLP1
    yield from ()


# Whole-dataset checks


def unique_ids(pairs: list[Pair]) -> Iterator[str]:
    for id_, count in Counter(p.id for p in pairs).items():
        if count > 1:
            yield f"id {id_} appears {count} times"


def no_duplicate_rows(pairs: list[Pair]) -> Iterator[str]:
    rows = Counter((p.fused, tuple(p.split_parts), p.source_ref) for p in pairs)
    for (fused, _, source_ref), count in rows.items():
        if count > 1:
            yield f"{fused} from {source_ref} appears {count} times"


def sentence_in_one_split(pairs: list[Pair]) -> Iterator[str]:
    for sentence_id, splits in split_spread(pairs, lambda p: p.sentence_id):
        yield f"sentence {sentence_id} is in splits {splits}"


def fused_in_one_split(pairs: list[Pair]) -> Iterator[str]:
    for fused, splits in split_spread(pairs, lambda p: p.fused):
        yield f"{fused} is in splits {splits}"


def split_sizes(pairs: list[Pair]) -> Iterator[str]:
    if not pairs:
        return
    counts = Counter(p.split_assignment for p in pairs)
    for split, target in SPLIT_SHARE.items():
        share = counts[split] / len(pairs)
        if abs(share - target) > SPLIT_TOLERANCE:
            yield f"{split} is {share:.0%} of rows, expected about {target:.0%}"


def rule_classes_in_train_and_test(pairs: list[Pair]) -> Iterator[str]:
    for split in ("train", "test"):
        seen = {j.rule_class for p in pairs if p.split_assignment == split for j in p.junctions}
        for rule_class in RuleClass:
            if rule_class != RuleClass.NONE and rule_class not in seen:
                yield f"no {rule_class} rows in {split}"


ROW_CHECKS: list[tuple[int, Callable[[Pair], Iterator[str]]]] = [
    (1, parts_not_empty),
    (2, junction_count),
    (3, span_inside_fused),
    (4, span_matches_surface),
    (5, junctions_in_order),
    (6, underlying_matches_parts),
    (7, rebuilds_fused),
    (8, cut_inside_junction),
    (9, cut_set_when_intact),
    (10, nothing_changed_agrees),
    (11, rule_ids_match_class),
    (12, control_is_intact),
    (13, alt_splits_well_formed),
    (14, frequency_at_least_one),
    (16, ids_match_corpus),
    (23, slp1_only),
    (24, devanagari_round_trip),
]

DatasetCheck = tuple[int, Level, Callable[[list[Pair]], Iterator[str]]]

DATASET_CHECKS: list[DatasetCheck] = [
    (17, "error", unique_ids),
    (18, "error", no_duplicate_rows),
    (19, "error", sentence_in_one_split),
    (20, "warning", fused_in_one_split),
    (21, "warning", split_sizes),
    (22, "warning", rule_classes_in_train_and_test),
]


def validate(pairs: list[Pair]) -> list[Issue]:
    issues = [
        Issue(number, "error", f"{pair.id}: {message}")
        for pair in pairs
        for number, check in ROW_CHECKS
        for message in check(pair)
    ]
    issues += [
        Issue(number, level, message)
        for number, level, check in DATASET_CHECKS
        for message in check(pairs)
    ]
    return issues
