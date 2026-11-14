"""Issue #23 test run: extract minimal-pair rows from one DCS CoNLL-U file.

Reads the DCS Kumārasaṃbhava chapter-1 dump (``data/raw/dcs/conllu/kumarasambhava-01.conllu``,
checksummed in ``data/manifest.yaml``) and writes Pair rows (docs/data-format.md)
as JSONL into ``data/processed/``. Wired into ``make data``.

Deterministic: no RNG anywhere. The only seed is SALT, a fixed string hashed with
the sentence id to pick train/dev/test, so re-running produces byte-identical
output and a sentence never changes split.

Rows come from two places in the CoNLL-U file:

- Multi-word tokens (``4-5<TAB>devatātmā``): DCS's own sandhi-fused forms; the
  sub-token lines after them carry the pre-sandhi words in their ``Unsandhied``
  field. The fused form is written as one word in the text, so junctions inside
  it are typed "compound".
- Adjacent token pairs across a space where sandhi touched the boundary
  (``asty uttarasyāṃ``: asti + uttarasyām, i -> y). The space stays in ``fused``
  (team decision on #22) and those junctions are typed "external".

``UnsandhiedReconstructed=True`` maps to split_provenance "reconstructed";
otherwise "gold". "derived" is unused here: DCS provides every split.

Known limits, each counted and reported at the end, never silently dropped
(the #20 rule):

- junctions whose change window spans two joins of a multi-part fusion are
  skipped as ambiguous;
- pairs where the change sits inside one word and not at the boundary
  (e.g. stem-form ``dogdhari`` from ``dogdhri`` before an untouched neighbour)
  are skipped: no sandhi acts at that join;
- pairs where either word is itself part of a fused token are skipped: the row's
  ``fused`` string would misrepresent the source line;
- rules outside the four in the schema's table get ``rule_ids=[]`` and are
  counted as unmapped, so the rules table can grow (#22);
- frequency is counted within this file only; corpus-wide counts need the full
  DCS download (#21).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from sandhi_probe.data.schema import Junction, Pair, RuleClass, RuleId
from sandhi_probe.paths import PROCESSED_DIR, RAW_DIR, REPO_ROOT

SALT = "sandhi-probe/extract/v1"  # change this and every split reshuffles
DCS_COMMIT = "8aeed5a"  # short sha of the pinned DCS dump (data/manifest.yaml)
RELATIVE_RAW = Path("dcs/conllu/kumarasambhava-01.conllu")
DEFAULT_OUT = PROCESSED_DIR / "dcs-kumarasambhava-01-testrun.jsonl"
FIXTURE_PATH = REPO_ROOT / "tests" / "data" / "fixtures" / "kumarasambhava-01-sample.jsonl"
FIXTURE_ROWS = 25

# SLP1 as produced by indic-transliteration 2.3.82 (verified by round trip).
SLP1_VOWELS = frozenset("aAiIuUfFxXeEoO")
SLP1_CONSONANTS = frozenset("kKgGncCjJYtTdDwWqQpPbBmyrlvzSshRNL")
SLP1_MARKS = frozenset("MH")  # anusvāra, visarga
ALLOWED_CHARS = SLP1_VOWELS | SLP1_CONSONANTS | SLP1_MARKS | {" ", "'"}

UNVOICED_STOPS = {"k": "g", "c": "j", "t": "d", "w": "q", "p": "b"}  # word-final voicing
YAN_MAP = {"i": "y", "I": "y", "u": "v", "U": "v", "f": "r", "F": "r", "x": "l", "X": "l"}
GUNA_MAP = {"i": "e", "I": "e", "u": "o", "U": "o"}  # a/ā + i/ī -> e, a/ā + u/ū -> o
SAVARNA_MAP = {"a": "A", "i": "I", "u": "U", "f": "F"}  # same short vowel twice -> long


@dataclass
class Tok:
    """One token occurrence, converted to SLP1."""

    no: int  # token number inside the sentence, as printed by DCS
    form: str  # surface form, SLP1; "" when DCS gives none
    unsandhied: str  # pre-sandhi form, SLP1; "" when DCS gives none
    reconstructed: bool  # DCS flagged the unsandhied form as reconstructed
    occ_id: int | None  # DCS's globally unique occurrence id


@dataclass
class Fusion:
    """One multi-word token: a fused form plus the token numbers of its parts."""

    form: str  # fused form, SLP1
    part_nos: list[int]  # token numbers of the parts, in order


@dataclass
class Sent:
    """One CoNLL-U sentence (a verse half)."""

    text: str  # raw DCS sentence text
    sent_id: int
    verse: int  # sent_counter: both halves of a verse share it
    toks: list[Tok]
    fusions: list[Fusion]


@dataclass
class PendingRow:
    """A row waiting for its frequency and split hash, in file order."""

    id: str
    fused: str
    provenance: str  # "gold" or "reconstructed"
    sentence_id: str
    is_control: bool
    junctions: list[Junction]
    sent_id: int
    tok_ref: str
    parts: list[str]


@dataclass
class Stats:
    sentences: int = 0
    skipped_sentences: int = 0
    bad_sentence_examples: list[str] = field(default_factory=list)
    tokens: int = 0
    fused_tokens: int = 0
    rows: int = 0
    controls: int = 0
    boundary_counts: Counter[str] = field(default_factory=Counter)
    rule_class_counts: Counter[str] = field(default_factory=Counter)
    unmapped_rules: int = 0
    skipped_missing_form: int = 0
    skipped_missing_unsandhied: int = 0
    skipped_missing_occid: int = 0
    skipped_ambiguous: int = 0
    skipped_internal_only: int = 0
    skipped_adjacent_fusion: int = 0
    hiatus_pairs: int = 0
    text_mismatches: int = 0
    duplicate_ids: int = 0


def to_slp1(text: str) -> str:
    """IAST -> SLP1. The ṁ->ṃ pre-map covers the one gap #20 documents."""
    return transliterate(text.replace("\u1e41", "\u1e43"), sanscript.IAST, sanscript.SLP1)


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def common_suffix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[len(a) - 1 - i] != b[len(b) - 1 - i]:
            return i
    return n


def parse_misc(misc: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in misc.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


def parse_conllu(path: Path) -> tuple[list[Sent], Stats]:
    """Parse the file, convert every Sanskrit field to SLP1, drop bad sentences."""
    stats = Stats()
    sents: list[Sent] = []
    text: str | None = None
    sent_id: int | None = None
    verse: int | None = None
    toks: list[Tok] = []
    fusions: list[Fusion] = []
    bad_chars: list[str] = []

    def bad_example() -> str:
        head = ",".join(f"{c!r}" for c in bad_chars[:3])
        return f"sent {sent_id}: {head}"

    def flush() -> None:
        nonlocal text, sent_id, verse, toks, fusions, bad_chars
        if sent_id is None:
            return
        if bad_chars:
            stats.skipped_sentences += 1
            if len(stats.bad_sentence_examples) < 5:
                stats.bad_sentence_examples.append(bad_example())
        else:
            assert text is not None and verse is not None
            stats.sentences += 1
            stats.tokens += len(toks)
            stats.fused_tokens += len(fusions)
            starts = {fusion.part_nos[0]: fusion for fusion in fusions if fusion.form}
            spans = {no for fusion in fusions if fusion.form for no in fusion.part_nos[1:]}
            pieces = []
            for tok in toks:
                if tok.no in spans:
                    continue  # inside a fusion: its form shows up via the fusion
                fusion = starts.get(tok.no)
                pieces.append(fusion.form if fusion is not None else tok.form)
            if " ".join(pieces) != to_slp1(text):
                stats.text_mismatches += 1
            sents.append(Sent(text, sent_id, verse, toks, fusions))
        text, sent_id, verse, toks, fusions, bad_chars = None, None, None, [], [], []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            flush()
            continue
        if line.startswith("##"):
            continue
        if line.startswith("#"):
            key, _, value = line.removeprefix("#").partition("=")
            key, value = key.strip(), value.strip()
            if key == "text":
                text = value
            elif key == "sent_id":
                sent_id = int(value)
            elif key == "sent_counter":
                verse = int(value)
            continue
        cols = line.split("\t")
        if len(cols) < 10:
            flush()
            continue
        token_no, form_raw, misc_raw = cols[0], cols[1], cols[9]
        misc = parse_misc(misc_raw) if misc_raw != "_" else {}
        if "-" in token_no:  # multi-word token: the fused form, parts follow
            start, end = (int(x) for x in token_no.split("-"))
            fused_form = to_slp1(form_raw) if form_raw != "_" else ""
            fusions.append(Fusion(fused_form, list(range(start, end + 1))))
            continue
        form = to_slp1(form_raw) if form_raw not in ("_", "") else ""
        unsandhied = (
            to_slp1(misc["Unsandhied"]) if misc.get("Unsandhied") not in (None, "_") else ""
        )
        occ_id = int(misc["OccId"]) if "OccId" in misc else None
        for out in (form, unsandhied):
            bad_chars.extend(c for c in out if c not in ALLOWED_CHARS)
        toks.append(
            Tok(
                no=int(token_no),
                form=form,
                unsandhied=unsandhied,
                reconstructed=misc.get("UnsandhiedReconstructed") == "True",
                occ_id=occ_id,
            )
        )
    flush()
    return sents, stats


def classify_rule(
    underlying_left: str, underlying_right: str, surface: str, cut: int | None
) -> tuple[RuleClass, list[RuleId]]:
    """rule_class plus any of the four rules the schema's table knows so far."""
    if underlying_left + underlying_right == surface:
        return RuleClass.NONE, []
    if "H" in underlying_left + underlying_right or "H" in surface:
        return RuleClass.VISARGA, []
    touched = (
        (underlying_left[-1:] + underlying_right[:1] + surface).replace(" ", "").replace("'", "")
    )
    is_consonant = touched and all(c in SLP1_CONSONANTS for c in touched)
    rule_class = RuleClass.CONSONANT if is_consonant else RuleClass.VOWEL

    ids: list[RuleId] = []
    if cut is None and underlying_left and underlying_right:
        left, right, out = underlying_left[-1], underlying_right[0], surface[:1]
        if left in "aA" and right in GUNA_MAP and out == GUNA_MAP[right]:
            ids.append(RuleId.GUNA)
        elif left in SAVARNA_MAP and right == left and out == SAVARNA_MAP[left]:
            ids.append(RuleId.SAVARNA_DIRGHA)
    if cut is not None and underlying_left:
        left, out = underlying_left[-1], surface[:1]
        if (
            rule_class == RuleClass.CONSONANT
            and left in UNVOICED_STOPS
            and out == UNVOICED_STOPS[left]
        ):
            ids.append(RuleId.WORD_FINAL_VOICING)
        elif rule_class == RuleClass.VOWEL and left in YAN_MAP and out == YAN_MAP[left]:
            ids.append(RuleId.YAN)
    return rule_class, ids


def junction_from_parts(parts: list[str], fused: str) -> Junction | None:
    """One junction between two neighbouring parts of a fused form, or None if ambiguous.

    Aligns the unsandhied concatenation with the fused form from both ends; the
    middle stretch is what sandhi replaced. The junction must not reach into a
    neighbouring part, so multi-part fusions with adjacent changes are refused.
    """
    cum = [0]
    for part in parts:
        cum.append(cum[-1] + len(part))
    whole = "".join(parts)
    junctions: list[Junction] = []
    for j in range(len(parts) - 1):
        left_start, join, right_end = cum[j], cum[j + 1], cum[j + 2]
        x_left = common_prefix_len(whole[:join], fused)
        x_right = common_suffix_len(whole[join:], fused)
        if x_left < left_start or x_right < len(whole) - right_end:
            return None  # the change window reaches past this junction's own two parts
        under_left = parts[j][x_left - left_start :]
        covered = max(0, min(len(parts[j + 1]), right_end - (len(whole) - x_right)))
        under_right = parts[j + 1][: len(parts[j + 1]) - covered]
        surface_mid = fused[x_left : len(fused) - x_right]
        overlap = x_left + x_right - len(fused)
        if overlap > 0:  # one letter is claimed by both the prefix and the suffix run
            if surface_mid or overlap > len(parts[j]) or overlap > len(parts[j + 1]):
                return None
            span_start = x_left - overlap
            surface = fused[span_start:x_left]
            under_left = parts[j][len(parts[j]) - overlap :]
            under_right = parts[j + 1][:overlap]
            if under_left + under_right == surface:
                return None  # would come out as intact but with no cut: not a real fusion
            rule_class, rule_ids = classify_rule(under_left, under_right, surface, None)
            junctions.append(
                Junction(
                    span=(span_start, x_left),
                    cut=None,
                    underlying_left=under_left,
                    underlying_right=under_right,
                    surface=surface,
                    rule_class=rule_class,
                    rule_ids=rule_ids,
                    type="compound",
                )
            )
            continue
        if not surface_mid:
            if under_left and under_right:
                return None  # both sides lost letters with no surface: unclear
            cut = x_left
            rule_class, rule_ids = classify_rule(under_left, under_right, "", cut)
            junctions.append(
                Junction(
                    span=(x_left, x_left),
                    cut=cut,
                    underlying_left=under_left,
                    underlying_right=under_right,
                    surface="",
                    rule_class=rule_class,
                    rule_ids=rule_ids,
                    type="compound",
                )
            )
            continue
        if not (under_left or under_right):
            return None  # surface material neither word claims: misaligned window
        if under_left and under_right:
            # a single surface vowel shared by both words: the merged case
            if (
                len(surface_mid) == 1
                and surface_mid in SLP1_VOWELS
                and under_right[0] in SLP1_VOWELS
                and (under_left[-1] in SLP1_VOWELS or under_left[-1] in SLP1_MARKS)
            ):
                rule_class, rule_ids = classify_rule(under_left, under_right, surface_mid, None)
                junctions.append(
                    Junction(
                        span=(x_left, x_left + 1),
                        cut=None,
                        underlying_left=under_left,
                        underlying_right=under_right,
                        surface=surface_mid,
                        rule_class=rule_class,
                        rule_ids=rule_ids,
                        type="compound",
                    )
                )
                continue
            return None  # both sides changed with no way to attribute the surface
        cut = x_left + len(surface_mid) if not under_right else x_left
        rule_class, rule_ids = classify_rule(under_left, under_right, surface_mid, cut)
        junctions.append(
            Junction(
                span=(x_left, x_left + len(surface_mid)),
                cut=cut,
                underlying_left=under_left,
                underlying_right=under_right,
                surface=surface_mid,
                rule_class=rule_class,
                rule_ids=rule_ids,
                type="compound",
            )
        )
    return junctions


def junction_for_pair(left: Tok, right: Tok) -> Junction | None:
    """The junction between two adjacent words separated by a space, or None.

    A junction exists only when a change touches the boundary itself: the left
    word's final letters differ from its surface, or the right word's initial
    letters do. A change inside one word away from the join (stem forms) is not
    a junction. The space stays in fused, so spans may include it and cut is
    always the space position.
    """
    fused = f"{left.form} {right.form}"
    space = len(left.form)
    surface_left, surface_right = fused[:space], fused[space + 1 :]
    x_left = common_prefix_len(left.unsandhied, surface_left)
    s_left = common_suffix_len(left.unsandhied, surface_left)
    x_right = common_prefix_len(right.unsandhied, surface_right)
    s_right = common_suffix_len(right.unsandhied, surface_right)
    touched_left = left.unsandhied != surface_left and s_left == 0
    touched_right = right.unsandhied != surface_right and x_right == 0
    if not (touched_left or touched_right):
        return None
    under_left = left.unsandhied[x_left : len(left.unsandhied) - s_left]
    over_left = surface_left[x_left : len(surface_left) - s_left]
    under_right = right.unsandhied[x_right : len(right.unsandhied) - s_right]
    over_right = surface_right[x_right : len(surface_right) - s_right]
    if not touched_left:  # the change sits inside the word, not at this join
        under_left = over_left = ""
    if not touched_right:
        under_right = over_right = ""
    span_start = x_left if touched_left and over_left else space
    span_end = space + 1 + len(over_right) if touched_right and over_right else space
    surface = fused[span_start:span_end]
    rule_class, rule_ids = classify_rule(under_left, under_right, surface, space)
    return Junction(
        span=(span_start, span_end),
        cut=space,
        underlying_left=under_left,
        underlying_right=under_right,
        surface=surface,
        rule_class=rule_class,
        rule_ids=rule_ids,
        type="external",
    )


def assign_split(sentence_id: str) -> str:
    """train/dev/test from a stable hash of the sentence id (80/10/10)."""
    digest = hashlib.sha256(f"{SALT}/{sentence_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def extract_rows(sents: list[Sent], stats: Stats) -> list[Pair]:
    """Turn every fusion and every boundary-touching adjacent pair into a Pair row."""
    pending: list[PendingRow] = []
    for sent in sents:
        by_no = {tok.no: tok for tok in sent.toks}
        in_fusion = {no for fusion in sent.fusions for no in fusion.part_nos}
        fusion_of = {no: fusion for fusion in sent.fusions for no in fusion.part_nos}
        sentence_id = f"dcs-{DCS_COMMIT}/chapter-7305/verse-{sent.verse}"

        for fusion in sent.fusions:
            parts = [by_no[no] for no in fusion.part_nos if no in by_no]
            if len(parts) != len(fusion.part_nos) or any(not p.unsandhied for p in parts):
                stats.skipped_missing_unsandhied += 1
                continue
            if any(p.occ_id is None for p in parts):
                stats.skipped_missing_occid += 1
                continue
            if not fusion.form:
                stats.skipped_missing_form += 1
                continue
            junctions = junction_from_parts([p.unsandhied for p in parts], fusion.form)
            if junctions is None:
                stats.skipped_ambiguous += 1
                continue
            provenance = "gold" if not any(p.reconstructed for p in parts) else "reconstructed"
            pending.append(
                PendingRow(
                    id=f"dcs-{parts[0].occ_id}",
                    fused=fusion.form,
                    provenance=provenance,
                    sentence_id=sentence_id,
                    is_control=all(j.boundary_level == "intact" for j in junctions),
                    junctions=junctions,
                    sent_id=sent.sent_id,
                    tok_ref=f"tok-{fusion.part_nos[0]}-{fusion.part_nos[-1]}",
                    parts=[p.unsandhied for p in parts],
                )
            )

        for a, b in zip(sent.toks, sent.toks[1:], strict=False):
            if a.no in in_fusion and b.no in in_fusion and fusion_of[a.no] is fusion_of[b.no]:
                continue  # inside one fusion: covered by the fusion row
            if not a.form or not b.form or not a.unsandhied or not b.unsandhied:
                stats.skipped_missing_form += 1
                continue
            if a.no in in_fusion or b.no in in_fusion:
                if junction_for_pair(a, b) is not None:
                    stats.skipped_adjacent_fusion += 1
                continue
            if a.occ_id is None or b.occ_id is None:
                stats.skipped_missing_occid += 1
                continue
            junction = junction_for_pair(a, b)
            if junction is None:
                if a.form != a.unsandhied or b.form != b.unsandhied:
                    stats.skipped_internal_only += 1
                else:
                    stats.hiatus_pairs += 1
                continue
            provenance = "gold" if not (a.reconstructed or b.reconstructed) else "reconstructed"
            pending.append(
                PendingRow(
                    id=f"dcs-{a.occ_id}",
                    fused=f"{a.form} {b.form}",
                    provenance=provenance,
                    sentence_id=sentence_id,
                    is_control=junction.boundary_level == "intact",
                    junctions=[junction],
                    sent_id=sent.sent_id,
                    tok_ref=f"tok-{a.no}",
                    parts=[a.unsandhied, b.unsandhied],
                )
            )

    forms = Counter()
    pair_forms = Counter()
    for sent in sents:
        for tok in sent.toks:
            if tok.form:
                forms[tok.form] += 1
        for a, b in zip(sent.toks, sent.toks[1:], strict=False):
            if a.form and b.form:
                pair_forms[(a.form, b.form)] += 1
        for fusion in sent.fusions:
            forms[fusion.form] += 1

    rows: list[Pair] = []
    seen: set[str] = set()
    for entry in pending:
        row_id = entry.id
        if row_id in seen:  # a token can anchor two rows when it sits in a fusion and changes twice
            stats.duplicate_ids += 1
            row_id = f"{row_id}m"
        seen.add(row_id)
        head, _, tail = entry.fused.partition(" ")
        frequency = pair_forms[(head, tail)] if tail else forms[head]
        rows.append(
            Pair(
                id=row_id,
                fused=entry.fused,
                split_parts=entry.parts,
                junctions=entry.junctions,
                source_corpus="DCS",
                source_ref=f"dcs-{DCS_COMMIT}/chapter-7305/sent-{entry.sent_id}/{entry.tok_ref}",
                sentence_id=entry.sentence_id,
                split_provenance=entry.provenance,
                frequency=frequency,
                is_control=entry.is_control,
                split_assignment=assign_split(entry.sentence_id),
            )
        )
    stats.rows = len(rows)
    stats.controls = sum(1 for row in rows if row.is_control)
    for row in rows:
        for junction in row.junctions:
            stats.boundary_counts[junction.boundary_level.value] += 1
            stats.rule_class_counts[junction.rule_class.value] += 1
            if junction.rule_class != RuleClass.NONE and not junction.rule_ids:
                stats.unmapped_rules += 1
    return rows


def check_row(row: Pair) -> None:
    """The invariants #24 will check, applied to every row as it is built."""
    assert len(row.junctions) == len(row.split_parts) - 1, row.id
    for junction in row.junctions:
        start, end = junction.span
        assert 0 <= start <= end <= len(row.fused), row.id
        assert row.fused[start:end] == junction.surface, row.id
        if junction.boundary_level == "intact":
            assert junction.rule_class == RuleClass.NONE, row.id
        else:
            assert junction.rule_class != RuleClass.NONE, row.id
    assert row.is_control == all(j.boundary_level == "intact" for j in row.junctions), row.id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=RAW_DIR / RELATIVE_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixture-out", type=Path, default=None)
    parser.add_argument("--fixture-rows", type=int, default=FIXTURE_ROWS)
    args = parser.parse_args()

    sents, stats = parse_conllu(args.raw)
    rows = extract_rows(sents, stats)
    for row in rows:
        check_row(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row.model_dump(), ensure_ascii=False) + "\n" for row in rows)
    args.out.write_text(payload, encoding="utf-8")
    print(f"extract: wrote {len(rows)} rows to {args.out}")

    if args.fixture_out is not None:
        args.fixture_out.parent.mkdir(parents=True, exist_ok=True)
        head_rows = rows[: args.fixture_rows]
        args.fixture_out.write_text(
            "".join(json.dumps(row.model_dump(), ensure_ascii=False) + "\n" for row in head_rows),
            encoding="utf-8",
        )
        print(f"extract: wrote first {len(head_rows)} rows to {args.fixture_out}")

    print(
        f"extract: {stats.sentences} sentences, {stats.tokens} tokens,"
        f" {stats.fused_tokens} fused forms"
    )
    print(
        f"extract: boundaries {dict(stats.boundary_counts)}"
        f" rule_classes {dict(stats.rule_class_counts)}"
    )
    print(
        "extract: skipped:"
        f" ambiguous={stats.skipped_ambiguous}"
        f" internal_only={stats.skipped_internal_only}"
        f" adjacent_fusion={stats.skipped_adjacent_fusion}"
        f" missing_form={stats.skipped_missing_form}"
        f" missing_unsandhied={stats.skipped_missing_unsandhied}"
        f" missing_occid={stats.skipped_missing_occid}"
        f" bad_sentences={stats.skipped_sentences}"
    )
    print(
        "extract: other:"
        f" controls={stats.controls}"
        f" hiatus_pairs={stats.hiatus_pairs}"
        f" unmapped_rules={stats.unmapped_rules}"
        f" duplicate_ids={stats.duplicate_ids}"
        f" text_mismatches={stats.text_mismatches}"
    )
    for example in stats.bad_sentence_examples:
        print(f"extract: skipped sentence: {example}")


if __name__ == "__main__":
    main()
