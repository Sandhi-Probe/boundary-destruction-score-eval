"""Tokenizer harness: the character span of every token, for every tokenizer we study (#45).

    from sandhi_probe.tokenizers.harness import tokenize

    r = tokenize("gpt2", "tadapi")
    r.spans             # ((0, 1), (1, 3), (3, 6)): one (start, end) per token, into the text
    r.inside_char_cuts  # (): token boundaries that fall inside one character

Every result is checked before it is returned, and a broken one raises HarnessError:

- one span per token (special tokens like BOS/EOS are left out: they cover no text);
- spans are half-open [start, end), in order, with no gaps and no overlaps, from 0 to len(text);
- spans index the string you passed in, even when the tokenizer normalizes it or adds a
  leading-space marker (GPT-2's "Ġ", SentencePiece's "▁").

The only exception: text that is empty or only whitespace can give no tokens at all (mT5 drops
whitespace), and then `spans` is empty.

A "character" is one Python str index, i.e. one Unicode code point. In Devanagari "ति" is two
(त + the vowel sign ि), so a cut between them is a boundary between characters, not inside one.

How spans are worked out depends on the tokenizer's `family` in configs/tokenizers.yaml, so a new
tokenizer from a family we already handle needs only a YAML entry.

Byte path: `bpe-bytelevel` (GPT-2) and `bytes` (ByT5).
    These cut UTF-8 bytes, so we rebuild each token's exact bytes (ByT5: id - 3; GPT-2: its
    printable stand-ins mapped back to bytes) and check that together they spell the text.
    A token boundary that lands inside a multi-byte character (every Devanagari letter is 3
    bytes, IAST ā is 2) is rounded up to the end of that character: a character belongs to the
    token holding its first byte. The tokens holding only its later bytes get zero-width spans,
    and each such boundary is reported in `inside_char_cuts`. Rounding (rather than merging the
    tokens) keeps exactly one span per token, which fertility and scoring count on.

Offset path: `bpe-sentencepiece` (Gemma), `unigram` (mT5), `wordpiece`.
    Hugging Face's offset mapping already points into the original string, normalization
    included, but has gaps and overlaps. One pass repairs it:
    1. Gaps go to the next token (these tokenizers attach a space to the word after it); a gap
       at the very end goes to the last token (mT5 drops trailing spaces).
    2. A bare "▁" token covers the text between the previous token and the next token's start:
       the space it stands for, or nothing for the made-up prefix at the start of a string.
       (mT5 reports a bare "▁" at the next token's position, so without this rule it would take
       that token's first letter.)
    3. Overlaps go to the earlier token.
    Byte-fallback pieces like "<0xB3>" (Gemma, for characters outside its vocabulary) are cut
    inside a character exactly when the piece is a UTF-8 continuation byte (0x80-0xBF); rule 3
    then gives them zero-width spans, the same convention as the byte path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import Any, NamedTuple

from sandhi_probe.tokenizers import inventory

OFFSET_FAMILIES = frozenset({"bpe-sentencepiece", "unigram", "wordpiece"})

_BYTE_PIECE = re.compile(r"<0x([0-9A-Fa-f]{2})>")  # SentencePiece byte fallback, e.g. <0xE0>
_SPACE_MARKER = "▁"  # U+2581, SentencePiece's "a space was here"
_BYT5_OFFSET = 3  # ByT5 ids 0-2 are pad, </s>, <unk>; byte b is id b + 3


class HarnessError(Exception):
    """A tokenizer gave output the harness can't turn into valid spans."""


class CharCut(NamedTuple):
    """A token boundary that falls inside one character."""

    token: int  # the cut is right after this token
    char: int  # index of the character it cuts through


@dataclass(frozen=True)
class Tokenization:
    """The tokens of `text` as character spans. See the module docstring for the guarantees."""

    text: str
    spans: tuple[tuple[int, int], ...]
    inside_char_cuts: tuple[CharCut, ...] = ()


# --- public entry points ------------------------------------------------------------------------


def tokenize(name: str, text: str) -> Tokenization:
    """Tokenize `text` with the inventory tokenizer `name` (e.g. "gpt2") at its pinned revision."""
    spec = inventory.get(name)
    return tokenize_with(_load(spec.name), spec.family, text)


def tokenize_with(tok: Any, family: str, text: str) -> Tokenization:
    """Same as `tokenize`, for a tokenizer you already loaded. `family` picks the path."""
    if family == "bytes":
        ids = tok(text, add_special_tokens=False).input_ids
        result = spans_from_bytes(text, [_byt5_byte(i) for i in ids])
    elif family == "bpe-bytelevel":
        ids = tok(text, add_special_tokens=False).input_ids
        pieces = tok.convert_ids_to_tokens(ids)
        result = spans_from_bytes(text, [_bytelevel_bytes(p) for p in pieces])
    elif family in OFFSET_FAMILIES:
        if not getattr(tok, "is_fast", False):
            raise HarnessError(f"family {family!r} needs a fast tokenizer for offsets")
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        pieces = tok.convert_ids_to_tokens(enc.input_ids)
        result = spans_from_offsets(text, enc.offset_mapping, pieces)
    else:
        raise HarnessError(f"no harness path for family {family!r}")
    return result


@cache
def _load(name: str) -> Any:
    """Load once per process: the sweep calls tokenize() per pair, and loading takes seconds."""
    return inventory.load(inventory.get(name))


# --- byte path ----------------------------------------------------------------------------------


def spans_from_bytes(text: str, token_bytes: list[bytes]) -> Tokenization:
    """Character spans for tokens given as their exact UTF-8 bytes (the byte path)."""
    data = text.encode("utf-8")
    if b"".join(token_bytes) != data:
        raise HarnessError(f"tokens don't spell the text back byte for byte: {text!r}")
    if any(not b for b in token_bytes):
        raise HarnessError(f"a token has no bytes: {text!r}")

    # For each byte offset: the character boundary it rounds up to, and, if it is inside a
    # character, which one.
    round_up = [0] * (len(data) + 1)
    inside: dict[int, int] = {}
    pos = 0
    for i, ch in enumerate(text):
        width = len(ch.encode("utf-8"))
        round_up[pos] = i
        for k in range(1, width):
            round_up[pos + k] = i + 1
            inside[pos + k] = i
        pos += width
    round_up[len(data)] = len(text)

    spans, cuts, pos = [], [], 0
    for t, b in enumerate(token_bytes):
        start = round_up[pos]
        pos += len(b)
        spans.append((start, round_up[pos]))
        if pos in inside:
            cuts.append(CharCut(t, inside[pos]))
    return _check(Tokenization(text, tuple(spans), tuple(cuts)))


def _byt5_byte(token_id: int) -> bytes:
    byte = token_id - _BYT5_OFFSET
    if not 0 <= byte < 256:
        raise HarnessError(f"ByT5 id {token_id} is not a byte (special token in the text?)")
    return bytes([byte])


@cache
def _bytelevel_decoder() -> dict[str, int]:
    """GPT-2's printable stand-in character for each byte, inverted (stand-in -> byte).

    Byte-level BPE shows every byte as a printable character (the space byte as "Ġ"). This is
    the published table from GPT-2's encoder.py, rebuilt here so we don't import it from
    transformers internals that move between versions.
    """
    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    stand_in = {b: chr(b) for b in printable}
    extra = 0
    for b in range(256):
        if b not in stand_in:
            stand_in[b] = chr(256 + extra)
            extra += 1
    return {c: b for b, c in stand_in.items()}


def _bytelevel_bytes(piece: str) -> bytes:
    table = _bytelevel_decoder()
    try:
        return bytes(table[c] for c in piece)
    except KeyError:
        msg = f"token {piece!r} is not byte-level (special token in the text?)"
        raise HarnessError(msg) from None


# --- offset path --------------------------------------------------------------------------------


def spans_from_offsets(
    text: str, offsets: list[tuple[int, int]], pieces: list[str]
) -> Tokenization:
    """Repair a Hugging Face offset mapping into valid spans (the offset path)."""
    if len(offsets) != len(pieces):
        raise HarnessError("offsets and pieces differ in length")
    n = len(pieces)
    spans, cuts, prev = [], [], 0
    for t, ((_, raw_end), piece) in enumerate(zip(offsets, pieces, strict=True)):
        if piece and set(piece) == {_SPACE_MARKER}:
            # rule 2: up to where the next token starts (or the end of the text)
            raw_end = offsets[t + 1][0] if t + 1 < n else len(text)
        end = max(prev, raw_end)  # rule 3: an overlap stays with the earlier token
        if t == n - 1:
            end = len(text)  # rule 1: a trailing gap goes to the last token
        spans.append((prev, end))  # rule 1: starting at `prev` gives any gap to this token
        prev = end
        if t + 1 < n and _is_continuation_byte(pieces[t + 1]):
            cuts.append(CharCut(t, offsets[t + 1][0]))
    return _check(Tokenization(text, tuple(spans), tuple(cuts)))


def _is_continuation_byte(piece: str) -> bool:
    m = _BYTE_PIECE.fullmatch(piece)
    return m is not None and 0x80 <= int(m.group(1), 16) <= 0xBF


# --- the guarantees -----------------------------------------------------------------------------


def _check(r: Tokenization) -> Tokenization:
    """Raise HarnessError unless `r` meets every guarantee in the module docstring."""
    if not r.spans:
        if r.text.strip():
            raise HarnessError(f"no tokens for non-blank text {r.text!r}")
        return r
    prev = 0
    for i, (start, end) in enumerate(r.spans):
        if start != prev or end < start:
            raise HarnessError(f"span {i} is {(start, end)}, expected it to start at {prev}")
        prev = end
    if prev != len(r.text):
        raise HarnessError(f"spans end at {prev}, text has {len(r.text)} characters")
    for cut in r.inside_char_cuts:
        if not (0 <= cut.token < len(r.spans) - 1 and 0 <= cut.char < len(r.text)):
            raise HarnessError(f"{cut} is out of range")
    return r
