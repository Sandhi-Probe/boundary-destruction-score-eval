"""Tests for configs/tokenizers.yaml and its loader. No network: the Hub check is
`python -m sandhi_probe.tokenizers.inventory --verify`, run by hand."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers import ByT5Tokenizer, PreTrainedTokenizerFast

from sandhi_probe.tokenizers import inventory as inv

GOOD = {
    "name": "gpt2",
    "hf_id": "openai-community/gpt2",
    "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "family": "bpe-bytelevel",
    "vocab_size": 50257,
    "gated": False,
    "licence": "mit",
    "offsets": True,
    "normalizer": "none",
}


def write(tmp_path: Path, data: object) -> Path:
    p = tmp_path / "tokenizers.yaml"
    p.write_text(data if isinstance(data, str) else yaml.safe_dump(data), encoding="utf-8")
    return p


# --- the committed file ------------------------------------------------------------------------


# Agreed with the TPMs on 2026-11-08: one tokenizer per family.
AGREED = {"gpt2": "bpe-bytelevel", "gemma4": "bpe-sentencepiece", "mt5": "unigram", "byt5": "bytes"}


def test_committed_inventory_is_valid() -> None:
    assert inv.load_inventory()


def test_has_the_agreed_tokenizers() -> None:
    families = {s.name: s.family for s in inv.load_inventory()}
    for name, family in AGREED.items():
        assert families.get(name) == family, f"configs/tokenizers.yaml: no {name} ({family})"


def test_get_by_name() -> None:
    assert inv.get("byt5").hf_id == "google/byt5-small"
    with pytest.raises(KeyError, match="have: gpt2"):
        inv.get("no-such-tokenizer")


# --- validation --------------------------------------------------------------------------------


def test_minimal_valid_file(tmp_path: Path) -> None:
    [spec] = inv.load_inventory(write(tmp_path, {"tokenizers": [GOOD]}))
    assert spec.name == "gpt2" and spec.notes == ""


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"family": "sentencepiece"}, "family"),
        ({"revision": "main"}, "40-character commit SHA"),
        ({"revision": "607a30d"}, "40-character commit SHA"),
        ({"hf_id": "gpt2"}, "owner/repo"),
        ({"name": "GPT-2"}, "lowercase"),
        ({"vocab_size": 0}, "vocab_size"),
        ({"normalizer": "NFKC"}, "normalizer"),
        ({"license": "mit"}, "license"),  # the field is spelled `licence`; typos must not pass
    ],
)
def test_rejects_bad_fields(tmp_path: Path, change: dict, message: str) -> None:
    with pytest.raises(inv.InventoryError, match=message):
        inv.load_inventory(write(tmp_path, {"tokenizers": [{**GOOD, **change}]}))


def test_rejects_missing_field(tmp_path: Path) -> None:
    entry = {k: v for k, v in GOOD.items() if k != "offsets"}
    with pytest.raises(inv.InventoryError, match="gpt2: offsets"):
        inv.load_inventory(write(tmp_path, {"tokenizers": [entry]}))


@pytest.mark.parametrize("key", ["name", "hf_id"])
def test_rejects_duplicates(tmp_path: Path, key: str) -> None:
    other = {**GOOD, "name": "other", "hf_id": "someone/other", key: GOOD[key]}
    with pytest.raises(inv.InventoryError, match="more than once"):
        inv.load_inventory(write(tmp_path, {"tokenizers": [GOOD, other]}))


@pytest.mark.parametrize(
    "content",
    ["", "tokenizers: []\n", "tokenizers: gpt2\n", "- gpt2\n", "models: []\n", "tokenizers: [\n"],
)
def test_rejects_bad_shape(tmp_path: Path, content: str) -> None:
    with pytest.raises(inv.InventoryError):
        inv.load_inventory(write(tmp_path, content))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(inv.InventoryError, match="can't read"):
        inv.load_inventory(tmp_path / "nope.yaml")


def test_reports_every_problem_at_once(tmp_path: Path) -> None:
    bad = [{**GOOD, "family": "x"}, {**GOOD, "name": "b", "hf_id": "a/b", "revision": "main"}]
    with pytest.raises(inv.InventoryError) as e:
        inv.load_inventory(write(tmp_path, {"tokenizers": bad}))
    assert "gpt2: family" in str(e.value) and "b: revision" in str(e.value)


# --- what --verify compares, on small tokenizers built locally ---------------------------------


def fast(model, normalizer=None, pre_tokenizer=None) -> PreTrainedTokenizerFast:
    t = Tokenizer(model)
    if normalizer is not None:
        t.normalizer = normalizer
    if pre_tokenizer is not None:
        t.pre_tokenizer = pre_tokenizer
    return PreTrainedTokenizerFast(tokenizer_object=t)


VOCAB = {"r": 0, "A": 1, "m": 2, "e": 3, "t": 4, "i": 5, "[UNK]": 6}


@pytest.mark.parametrize(
    ("tok", "family"),
    [
        (fast(models.BPE(VOCAB, []), pre_tokenizer=pre_tokenizers.ByteLevel()), "bpe-bytelevel"),
        (fast(models.BPE(VOCAB, [], byte_fallback=True, unk_token="[UNK]")), "bpe-sentencepiece"),
        (fast(models.Unigram([(k, -1.0) for k in VOCAB], unk_id=6)), "unigram"),
        (fast(models.WordPiece(VOCAB, unk_token="[UNK]")), "wordpiece"),
        (ByT5Tokenizer(), "bytes"),
    ],
)
def test_family_of(tok, family: str) -> None:
    assert inv.family_of(tok) == family


@pytest.mark.parametrize(
    ("normalizer", "expected"),
    [
        (None, "none"),
        (normalizers.NFC(), "nfc"),
        (normalizers.NFKC(), "nfkc"),
        (normalizers.BertNormalizer(lowercase=False, strip_accents=False), "none"),
    ],
)
def test_normalizer_of(normalizer, expected: str) -> None:
    assert (
        inv.normalizer_of(fast(models.WordPiece(VOCAB, unk_token="[UNK]"), normalizer)) == expected
    )


def test_has_offsets() -> None:
    assert inv.has_offsets(fast(models.WordPiece(VOCAB, unk_token="[UNK]")))
    assert not inv.has_offsets(ByT5Tokenizer())


def test_mismatches() -> None:
    spec = inv.TokenizerSpec.model_validate(GOOD)
    assert inv.mismatches(spec, {"vocab_size": 50257, "gated": False}) == []
    assert inv.mismatches(spec, {"vocab_size": 50000}) == [
        "vocab_size: file says 50257, Hub says 50000"
    ]


# --- command line ------------------------------------------------------------------------------


def test_main_prints_the_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert inv.main([]) == 0
    assert "google/byt5-small" in capsys.readouterr().out


def test_main_fails_on_invalid_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert inv.main(["--file", str(write(tmp_path, "tokenizers: []\n"))]) == 1
    assert "non-empty list" in capsys.readouterr().err
