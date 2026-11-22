"""Tests for the tokenizer harness (#45).

Everything but the last section runs offline: small tokenizers built inside the test, and offset
mappings recorded from the real tokenizers. The last section uses the real tokenizers only if
they are already in your Hugging Face cache, and skips otherwise.
"""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, ByT5Tokenizer, PreTrainedTokenizerFast

from sandhi_probe.tokenizers import harness as h
from sandhi_probe.tokenizers import inventory as inv
from sandhi_probe.tokenizers.harness import CharCut, HarnessError, Tokenization

SLP1 = ["tadapi", "rAma udakam", "rAmo 'tra", "sItodakam"]
DEVANAGARI = ["तदपि", "राम उदकम्", "सीतोदकम्"]


def assert_valid(r: Tokenization, n_tokens: int) -> None:
    """The guarantees, checked from outside (the harness also checks them itself)."""
    assert len(r.spans) == n_tokens
    assert r.spans[0][0] == 0 and r.spans[-1][1] == len(r.text)
    for (_, end), (start, _) in zip(r.spans, r.spans[1:], strict=False):
        assert end == start  # in order, no gap, no overlap
    assert "".join(r.text[s:e] for s, e in r.spans) == r.text


# --- small offline tokenizers ------------------------------------------------------------------


def byt5() -> ByT5Tokenizer:
    return ByT5Tokenizer()  # no vocabulary file: ByT5 is just bytes


def tiny_bytelevel() -> PreTrainedTokenizerFast:
    """A GPT-2-style tokenizer: byte-level alphabet plus a couple of merges."""
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {c: i for i, c in enumerate(sorted(alphabet))}
    merges = [("a", "d"), ("a", "p"), ("ap", "i")]
    for a, b in merges:
        vocab[a + b] = len(vocab)
    t = Tokenizer(models.BPE(vocab, merges))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return PreTrainedTokenizerFast(tokenizer_object=t)


def n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)


# --- byte path: GPT-2 and ByT5 -----------------------------------------------------------------


@pytest.mark.parametrize("text", SLP1 + DEVANAGARI)
@pytest.mark.parametrize(("make", "family"), [(byt5, "bytes"), (tiny_bytelevel, "bpe-bytelevel")])
def test_byte_path_spans_are_valid(make, family: str, text: str) -> None:
    tok = make()
    assert_valid(h.tokenize_with(tok, family, text), n_tokens(tok, text))


@pytest.mark.parametrize("text", SLP1)
@pytest.mark.parametrize(("make", "family"), [(byt5, "bytes"), (tiny_bytelevel, "bpe-bytelevel")])
def test_slp1_is_never_cut_inside_a_character(make, family: str, text: str) -> None:
    assert h.tokenize_with(make(), family, text).inside_char_cuts == ()


def test_cut_inside_a_character_devanagari() -> None:
    # त is 3 bytes, so ByT5 gives it 3 tokens: the first owns the character, the other two are
    # zero-width, and both boundaries inside it are reported.
    r = h.tokenize_with(byt5(), "bytes", "तद")
    assert r.spans == ((0, 1), (1, 1), (1, 1), (1, 2), (2, 2), (2, 2))
    assert r.inside_char_cuts == (
        CharCut(token=0, char=0),
        CharCut(token=1, char=0),
        CharCut(token=3, char=1),
        CharCut(token=4, char=1),
    )


def test_cut_inside_a_character_iast() -> None:
    r = h.tokenize_with(byt5(), "bytes", "rāma")  # ā is 2 bytes
    assert r.spans == ((0, 1), (1, 2), (2, 2), (2, 3), (3, 4))
    assert r.inside_char_cuts == (CharCut(token=1, char=1),)


def test_bytelevel_leading_space_belongs_to_the_next_word() -> None:
    r = h.tokenize_with(tiny_bytelevel(), "bpe-bytelevel", "a d")
    pieces = ["a", "Ġ", "d"]  # no "Ġd" merge in the tiny vocab, so the space is its own token
    assert len(r.spans) == len(pieces)
    assert [r.text[s:e] for s, e in r.spans] == ["a", " ", "d"]


def test_spans_from_bytes_rejects_bytes_that_dont_spell_the_text() -> None:
    with pytest.raises(HarnessError, match="byte for byte"):
        h.spans_from_bytes("tad", [b"t", b"a"])


def test_bytelevel_table_covers_every_byte_once() -> None:
    table = h._bytelevel_decoder()
    assert sorted(table.values()) == list(range(256))
    assert set(table) == set(pre_tokenizers.ByteLevel.alphabet())
    assert table["Ġ"] == ord(" ")


# --- offset path: Gemma and mT5 ----------------------------------------------------------------
# Raw (pieces, offsets) recorded from the real tokenizers at the revisions pinned in
# configs/tokenizers.yaml, on 2026-11-21. They pin down exactly the quirks the repair rules handle.

RECORDED = {
    "mt5 bare ▁ at start and before a word": (
        "rAma udakam",
        [
            ("▁", (0, 1)),
            ("r", (0, 1)),
            ("Ama", (1, 4)),
            ("▁", (5, 6)),
            ("uda", (5, 8)),
            ("kam", (8, 11)),
        ],
        [(0, 0), (0, 1), (1, 4), (4, 5), (5, 8), (8, 11)],
    ),
    "mt5 leading space": (
        " rAma",
        [("▁", (1, 2)), ("r", (1, 2)), ("Ama", (2, 5))],
        [(0, 1), (1, 2), (2, 5)],
    ),
    "mt5 drops the trailing space": (
        "rAma ",
        [("▁", (0, 1)), ("r", (0, 1)), ("Ama", (1, 4))],
        [(0, 0), (0, 1), (1, 5)],
    ),
    "mt5 collapses a double space": (
        "rAma  udakam",
        [
            ("▁", (0, 1)),
            ("r", (0, 1)),
            ("Ama", (1, 4)),
            ("▁", (6, 7)),
            ("uda", (6, 9)),
            ("kam", (9, 12)),
        ],
        [(0, 0), (0, 1), (1, 4), (4, 6), (6, 9), (9, 12)],
    ),
    "mt5 NFKC: nukta letter stays one original character": (
        "क़",
        [("▁", (0, 1)), ("क़", (0, 1))],
        [(0, 0), (0, 1)],
    ),
    "gemma space marker inside a token": (
        "rAma udakam",
        [("r", (0, 1)), ("Ama", (1, 4)), ("▁ud", (4, 7)), ("akam", (7, 11))],
        [(0, 1), (1, 4), (4, 7), (7, 11)],
    ),
    "gemma trailing bare ▁": (
        "rAma ",
        [("r", (0, 1)), ("Ama", (1, 4)), ("▁", (4, 5))],
        [(0, 1), (1, 4), (4, 5)],
    ),
    "gemma devanagari": (
        "तदपि",
        [("त", (0, 1)), ("द", (1, 2)), ("पि", (2, 4))],
        [(0, 1), (1, 2), (2, 4)],
    ),
}


@pytest.mark.parametrize("case", RECORDED)
def test_offset_repair_on_recorded_output(case: str) -> None:
    text, raw, expected = RECORDED[case]
    pieces = [p for p, _ in raw]
    r = h.spans_from_offsets(text, [o for _, o in raw], pieces)
    assert list(r.spans) == expected
    assert r.inside_char_cuts == ()
    assert_valid(r, len(raw))


def test_offset_path_reports_byte_fallback_cuts() -> None:
    # Gemma has no piece for ᳐ (U+1CD0, a Vedic sign), so it falls back to its 3 UTF-8 bytes.
    raw = [("<0xE1>", (0, 1)), ("<0xB3>", (0, 1)), ("<0x90>", (0, 1)), ("x", (1, 2))]
    r = h.spans_from_offsets("᳐x", [o for _, o in raw], [p for p, _ in raw])
    assert r.spans == ((0, 1), (1, 1), (1, 1), (1, 2))
    assert r.inside_char_cuts == (CharCut(token=0, char=0), CharCut(token=1, char=0))


def test_whitespace_only_text_may_have_no_tokens() -> None:
    assert h.spans_from_offsets("  ", [], []).spans == ()  # what mT5 does
    with pytest.raises(HarnessError, match="no tokens"):
        h.spans_from_offsets("rAma", [], [])


def test_empty_text() -> None:
    assert h.tokenize_with(byt5(), "bytes", "") == Tokenization("", ())


def test_unknown_family() -> None:
    with pytest.raises(HarnessError, match="no harness path"):
        h.tokenize_with(byt5(), "morse", "a")


# --- tokenize(name, text) ----------------------------------------------------------------------


def test_tokenize_loads_the_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = []

    def fake_load(spec: inv.TokenizerSpec):
        loaded.append(spec)
        return byt5()

    monkeypatch.setattr(inv, "load", fake_load)
    h._load.cache_clear()
    try:
        h.tokenize("byt5", "tadapi")
        h.tokenize("byt5", "rAma")  # second call is cached, not reloaded
    finally:
        h._load.cache_clear()
    assert len(loaded) == 1
    assert loaded[0].revision == inv.get("byt5").revision


# --- the real tokenizers, if cached ------------------------------------------------------------


def cached(name: str):
    spec = inv.get(name)
    try:
        tok = AutoTokenizer.from_pretrained(
            spec.hf_id, revision=spec.revision, local_files_only=True
        )
    except OSError:
        pytest.skip(f"{name} isn't in the Hugging Face cache; run tokenize({name!r}, ...) online")
    return tok, spec.family


@pytest.mark.parametrize("text", SLP1 + DEVANAGARI + ["rāma", " rAma ", "rAma  udakam", "᳐x"])
@pytest.mark.parametrize("name", ["gpt2", "gemma4", "mt5", "byt5"])
def test_real_tokenizers(name: str, text: str) -> None:
    tok, family = cached(name)
    r = h.tokenize_with(tok, family, text)
    assert_valid(r, n_tokens(tok, text))
    if text in SLP1:
        assert r.inside_char_cuts == ()


@pytest.mark.parametrize(
    ("name", "text", "n_cuts"),
    [("gpt2", "तदपि", 4), ("byt5", "तदपि", 8), ("gemma4", "᳐x", 2), ("mt5", "तदपि", 0)],
)
def test_real_inside_char_cuts(name: str, text: str, n_cuts: int) -> None:
    tok, family = cached(name)
    assert len(h.tokenize_with(tok, family, text).inside_char_cuts) == n_cuts
