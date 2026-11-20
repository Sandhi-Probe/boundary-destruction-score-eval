"""Tokenizer inventory: the tokenizers we evaluate, read from configs/tokenizers.yaml.

The harness and the sweep get their tokenizer list from `load_inventory()` and load each one
with `load(spec)`, so no script hard-codes a model id or forgets the pinned revision.

    uv run python -m sandhi_probe.tokenizers.inventory           # validate the file, print it
    uv run python -m sandhi_probe.tokenizers.inventory --verify  # also check every field on the Hub
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError, field_validator

from sandhi_probe import paths

INVENTORY = paths.CONFIGS_DIR / "tokenizers.yaml"

Family = Literal["bpe-bytelevel", "bpe-sentencepiece", "unigram", "wordpiece", "bytes"]
Normalizer = Literal["none", "nfc", "nfkc"]

_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
_HF_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_PROBE = "rAmeti"  # rāma + iti in SLP1, our main encoding


class InventoryError(Exception):
    """configs/tokenizers.yaml is missing, unreadable or invalid."""


class TokenizerSpec(BaseModel):
    """One tokenizer under study. Field meanings are in the header of configs/tokenizers.yaml."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    hf_id: str
    revision: str
    family: Family
    vocab_size: PositiveInt
    gated: bool
    licence: str
    offsets: bool
    normalizer: Normalizer
    notes: str = ""

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _NAME.fullmatch(v):
            raise ValueError(f"{v!r}: use lowercase a-z, 0-9, '.', '_' and '-', e.g. gemma4")
        return v

    @field_validator("hf_id")
    @classmethod
    def _check_hf_id(cls, v: str) -> str:
        if not _HF_ID.fullmatch(v):
            raise ValueError(
                f"{v!r}: use the full Hugging Face id, owner/repo, e.g. google/mt5-small"
            )
        return v

    @field_validator("revision")
    @classmethod
    def _check_revision(cls, v: str) -> str:
        if not _COMMIT.fullmatch(v):
            raise ValueError(
                f"{v!r}: pin a full 40-character commit SHA (the 'main' branch moves);"
                " it is the `sha` field of https://huggingface.co/api/models/<hf_id>"
            )
        return v


def load_inventory(path: Path = INVENTORY) -> list[TokenizerSpec]:
    """Every tokenizer in the inventory, in file order. Raises InventoryError if invalid."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        raise InventoryError(f"{path.name}: can't read it: {e}") from None
    if not isinstance(data, dict) or set(data) != {"tokenizers"}:
        raise InventoryError(f"{path.name}: expected one top-level key, `tokenizers:`")
    entries = data["tokenizers"]
    if not isinstance(entries, list) or not entries:
        raise InventoryError(f"{path.name}: `tokenizers:` must be a non-empty list")

    specs, problems = [], []
    for i, entry in enumerate(entries, 1):
        label = entry.get("name", f"entry {i}") if isinstance(entry, dict) else f"entry {i}"
        try:
            specs.append(TokenizerSpec.model_validate(entry))
        except ValidationError as e:
            for err in e.errors():
                field = ".".join(str(p) for p in err["loc"]) or "entry"
                problems.append(f"  {label}: {field}: {err['msg']}")
    for key in ("name", "hf_id"):
        seen: set[str] = set()
        for spec in specs:
            value = getattr(spec, key)
            if value in seen:
                problems.append(f"  {key} {value!r} appears more than once")
            seen.add(value)
    if problems:
        raise InventoryError(f"{path.name} is invalid:\n" + "\n".join(problems))
    return specs


def get(name: str, path: Path = INVENTORY) -> TokenizerSpec:
    """The inventory entry called `name`, e.g. get("byt5")."""
    specs = load_inventory(path)
    for spec in specs:
        if spec.name == name:
            return spec
    raise KeyError(
        f"no tokenizer {name!r} in {path.name}; have: {', '.join(s.name for s in specs)}"
    )


def load(spec: TokenizerSpec) -> Any:
    """The Hugging Face tokenizer for `spec`, at its pinned revision."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)


# --- checking the file against the Hub ---------------------------------------------------------


def family_of(tok: Any) -> str:
    """Which `family` value a loaded tokenizer belongs to."""
    if not getattr(tok, "is_fast", False):
        return "bytes" if type(tok).__name__ == "ByT5Tokenizer" else "unknown"
    j = json.loads(tok.backend_tokenizer.to_str())
    model = j.get("model") or {}
    kind = model.get("type")
    if kind == "Unigram":
        return "unigram"
    if kind == "WordPiece":
        return "wordpiece"
    if kind == "BPE":
        if "ByteLevel" in json.dumps(j.get("pre_tokenizer")):
            return "bpe-bytelevel"
        if model.get("byte_fallback"):
            return "bpe-sentencepiece"
    return "unknown"


def normalizer_of(tok: Any) -> str:
    """Which Unicode normalization a loaded tokenizer applies, judged by what it does."""
    norm = tok.backend_tokenizer.normalizer if getattr(tok, "is_fast", False) else None
    if norm is None:
        return "none"
    if norm.normalize_str("ﬁ") == "fi":  # the "fi" ligature: only NFKC splits it
        return "nfkc"
    if norm.normalize_str("é") == "é":  # e + combining acute: NFC composes it
        return "nfc"
    return "none"


def has_offsets(tok: Any) -> bool:
    """Whether the tokenizer returns an offset_mapping."""
    try:
        out = tok(_PROBE, add_special_tokens=False, return_offsets_mapping=True)
    except (NotImplementedError, ValueError, TypeError):
        return False
    return "offset_mapping" in out


def observe(spec: TokenizerSpec) -> dict[str, Any]:
    """What the Hub and the loaded tokenizer report, keyed like TokenizerSpec fields.

    Raises an error caused by huggingface_hub's GatedRepoError if you lack access to a gated repo.
    """
    from huggingface_hub import HfApi

    info = HfApi().model_info(spec.hf_id, revision=spec.revision)
    seen: dict[str, Any] = {
        "gated": bool(info.gated),
        "licence": getattr(info.card_data, "license", None),
    }
    tok = load(spec)
    seen.update(
        family=family_of(tok),
        vocab_size=tok.vocab_size,
        offsets=has_offsets(tok),
        normalizer=normalizer_of(tok),
    )
    return seen


def mismatches(spec: TokenizerSpec, seen: dict[str, Any]) -> list[str]:
    """`field: file says X, Hub says Y` for every observed field that disagrees with the file."""
    return [
        f"{field}: file says {getattr(spec, field)!r}, Hub says {value!r}"
        for field, value in seen.items()
        if getattr(spec, field) != value
    ]


def verify(specs: list[TokenizerSpec]) -> int:
    """Check every spec against the Hub. Returns the number of entries that disagree."""
    from huggingface_hub.errors import GatedRepoError

    def gated(e: BaseException | None) -> bool:  # transformers re-raises it as an OSError
        while e is not None and not isinstance(e, GatedRepoError):
            e = e.__cause__ or e.__context__
        return e is not None

    bad = skipped = 0
    for spec in specs:
        try:
            problems = mismatches(spec, observe(spec))
        except Exception as e:
            if not gated(e):
                raise
            skipped += 1
            print(
                f"SKIP {spec.name}: gated and you don't have access yet. Request it at"
                f" https://huggingface.co/{spec.hf_id}, then run `hf auth login`."
            )
            continue
        if problems:
            bad += 1
            print(f"FAIL {spec.name} ({spec.hf_id})")
            for p in problems:
                print(f"     {p}")
        else:
            print(f"ok   {spec.name}")
    print(f"\n{len(specs) - bad - skipped} match, {bad} differ, {skipped} skipped (gated)")
    return bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--verify", action="store_true", help="check every field on the Hub")
    parser.add_argument("--file", type=Path, default=INVENTORY, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        specs = load_inventory(args.file)
    except InventoryError as e:
        print(e, file=sys.stderr)
        return 1
    if args.verify:
        return 1 if verify(specs) else 0
    print(f"{'name':10} {'family':18} {'vocab':>8}  gated  offsets  hf_id")
    for s in specs:
        print(
            f"{s.name:10} {s.family:18} {s.vocab_size:>8}  {'yes' if s.gated else 'no':5}"
            f"  {'yes' if s.offsets else 'no':7}  {s.hf_id}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
