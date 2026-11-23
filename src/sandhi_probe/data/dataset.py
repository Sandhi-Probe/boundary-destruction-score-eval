"""Versioned dataset: the extracted rows, checked and written to data/processed/v0.1.parquet (#5).

`make dataset` runs this module. It reads the minimal-pair rows that extraction (#23) writes
to data/interim/*.jsonl, checks every row against the schema and the validation suite (#24),
and writes data/processed/<name>.parquet. The bytes depend only on the rows: they are sorted
by id and the Parquet layout is fixed below, so the same rows always give the same file. The
file's SHA-256 has to match data/datasets.yaml, which is committed, so a fresh clone either
rebuilds the dataset byte for byte or fails.

    uv run python -m sandhi_probe.data.dataset                    # build v0.1, check it
    uv run python -m sandhi_probe.data.dataset --rows a.jsonl ... # build from other row files
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from sandhi_probe import paths
from sandhi_probe.data.schema import Pair
from sandhi_probe.data.validate import Issue, validate

DEFAULT_NAME = "v0.1"
MAX_SHOWN = 20  # errors or warnings printed per kind; the rest are counted

_NAME = re.compile(r"v\d+\.\d+")
_SHA256 = re.compile(r"[0-9a-f]{64}")

# One column per Pair field, in the model's order, plus the computed boundary_level on each
# junction. Fixed here rather than inferred from the rows, so the file layout can't drift
# with the data. tests/data/test_dataset.py checks it against the model.
JUNCTION_TYPE = pa.struct(
    [
        pa.field("span", pa.list_(pa.int32(), 2), nullable=False),
        pa.field("cut", pa.int32()),
        pa.field("underlying_left", pa.string(), nullable=False),
        pa.field("underlying_right", pa.string(), nullable=False),
        pa.field("surface", pa.string(), nullable=False),
        pa.field("rule_class", pa.string(), nullable=False),
        pa.field("rule_ids", pa.list_(pa.string()), nullable=False),
        pa.field("type", pa.string(), nullable=False),
        pa.field("boundary_level", pa.string(), nullable=False),
    ]
)
SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("fused", pa.string(), nullable=False),
        pa.field("split_parts", pa.list_(pa.string()), nullable=False),
        pa.field("alt_splits", pa.list_(pa.list_(pa.string())), nullable=False),
        pa.field("junctions", pa.list_(JUNCTION_TYPE), nullable=False),
        pa.field("encoding", pa.string(), nullable=False),
        pa.field("source_corpus", pa.string(), nullable=False),
        pa.field("source_ref", pa.string(), nullable=False),
        pa.field("sentence_id", pa.string(), nullable=False),
        pa.field("split_provenance", pa.string(), nullable=False),
        pa.field("frequency", pa.int64(), nullable=False),
        pa.field("frequency_bucket", pa.string()),
        pa.field("is_control", pa.bool_(), nullable=False),
        pa.field("split_assignment", pa.string(), nullable=False),
    ]
)

# Every write option that affects the bytes, spelled out so a pyarrow default changing can't
# change the file. The created_by string inside the file still names the pyarrow version,
# so upgrading pyarrow in uv.lock means a new checksum in data/datasets.yaml.
WRITE_OPTIONS = {
    "version": "2.6",
    "data_page_version": "1.0",
    "compression": "zstd",
    "compression_level": 3,
    "use_dictionary": True,
    "write_statistics": True,
    "row_group_size": 1 << 20,
}


class DatasetError(Exception):
    """datasets.yaml, a row file or a built dataset is missing or invalid."""


# --- data/datasets.yaml ----------------------------------------------------------------------


class Dataset(BaseModel):
    """What one build of data/processed/<name>.parquet has to match."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str | None = None  # None until a first build has been reviewed and committed
    rows: int | None = Field(default=None, ge=1)

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, v: str | None) -> str | None:
        if v is not None and not _SHA256.fullmatch(v):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return v

    @model_validator(mode="after")
    def _both_or_neither(self) -> Dataset:
        if (self.sha256 is None) != (self.rows is None):
            raise ValueError("set sha256 and rows together, from the same build")
        return self


class Datasets(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    datasets: dict[str, Dataset]

    @field_validator("datasets")
    @classmethod
    def _names(cls, v: dict[str, Dataset]) -> dict[str, Dataset]:
        for name in v:
            if not _NAME.fullmatch(name):
                raise ValueError(f"{name!r}: dataset names look like v0.1")
        return v


def load_datasets(path: Path) -> Datasets:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DatasetError(f"{_rel(path)} not found") from None
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        raise DatasetError(f"{_rel(path)}: can't read it ({e})") from None
    try:
        return Datasets.model_validate(data)
    except ValidationError as e:
        raise DatasetError(f"{_rel(path)}: {e}") from None


# --- rows in, Parquet out --------------------------------------------------------------------


def read_rows(files: list[Path]) -> list[Pair]:
    """Every row in the JSONL files. Raises DatasetError naming each bad line."""
    pairs: list[Pair] = []
    problems: list[str] = []
    for file in files:
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as e:
            raise DatasetError(f"{_rel(file)}: can't read it ({e})") from None
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                pairs.append(Pair.model_validate_json(line))
            except ValidationError as e:
                first = e.errors()[0]
                where = ".".join(str(p) for p in first["loc"]) or "row"
                problems.append(f"{_rel(file)}:{n}: {where}: {first['msg']}")
    if problems:
        raise DatasetError(_shown(problems, "row(s) don't match the schema"))
    return pairs


def to_table(pairs: list[Pair]) -> pa.Table:
    rows = [p.model_dump(mode="json") for p in sorted(pairs, key=lambda p: p.id)]
    return pa.Table.from_pylist(rows, schema=SCHEMA)


def write_parquet(table: pa.Table, path: Path) -> None:
    pq.write_table(table, path, **WRITE_OPTIONS)


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_pairs(name: str = DEFAULT_NAME, *, datasets_path: Path | None = None) -> list[Pair]:
    """The rows of data/processed/<name>.parquet, after checking it against datasets.yaml."""
    expected = _entry(load_datasets(datasets_path or paths.DATASETS_PATH), name)
    path = paths.PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise DatasetError(f"{_rel(path)} not found: run `make dataset`")
    if expected.sha256 is None:
        raise DatasetError(f"no checksum committed for {name} in datasets.yaml yet")
    if (actual := sha256(path)) != expected.sha256:
        raise DatasetError(
            f"{_rel(path)} has sha256 {actual}, not the committed {expected.sha256}:"
            " run `make dataset`"
        )
    return read_pairs(path)


def read_pairs(path: Path) -> list[Pair]:
    return [Pair.model_validate(row) for row in pq.read_table(path).to_pylist()]


# --- make dataset ----------------------------------------------------------------------------


def build(name: str, files: list[Path], datasets_path: Path, out_dir: Path) -> int:
    expected = _entry(load_datasets(datasets_path), name)
    if not files:
        print(
            f"make dataset: no row files in {_rel(paths.INTERIM_DIR)}/."
            " Extraction (#23) writes them there; it isn't part of `make dataset` yet."
        )
        return 1
    print(f"make dataset: {len(files)} row file(s) -> {_rel(out_dir)}/{name}.parquet")
    pairs = read_rows(files)
    if not pairs:
        print("make dataset: the row files are empty; nothing to build")
        return 1

    issues = validate(pairs)
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]
    if warnings:
        print(_by_check(warnings, "validation warning(s), not blocking"))
    if errors:
        print(_shown([_issue(i) for i in errors], "validation error(s); nothing written"))
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.parquet"
    fd, tmp_name = tempfile.mkstemp(dir=out_dir, prefix=f".{name}.", suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        os.chmod(tmp, 0o666 & ~_umask())  # mkstemp makes it 0600
        write_parquet(to_table(pairs), tmp)
        actual = sha256(tmp)
        rows = f"{len(pairs)} rows, sha256 {actual}"
        if expected.sha256 is None:
            os.replace(tmp, target)
            print(f"  built {_rel(target)}: {rows}")
            print(
                f"make dataset: no checksum committed for {name} yet, so this build isn't"
                f" checked. Once the rows have been reviewed, put these in"
                f" {_rel(datasets_path)}:\n    sha256: {actual}\n    rows: {len(pairs)}"
            )
            return 0
        if actual == expected.sha256:
            os.replace(tmp, target)
            print(f"  ok  {_rel(target)}: {rows}")
            print("make dataset: ok")
            return 0
        rejected = out_dir / f"{name}.mismatch.parquet"
        os.replace(tmp, rejected)
        committed = f"{expected.rows} rows, sha256 {expected.sha256}"
        print(f"  MISMATCH  {name}: built {rows}\n            committed {committed}")
        print(
            f"make dataset: kept the new build as {_rel(rejected)} and left {_rel(target)}"
            " as it was. If you changed the rows on purpose, put the new sha256 and rows in"
            f" {_rel(datasets_path)} in the same PR. If not, `make data OFFLINE=1` rules out"
            " the raw files; then compare the two files."
        )
        return 1
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sandhi_probe.data.dataset",
        description="Build data/processed/<name>.parquet from the extracted rows and check"
        " its SHA-256 against data/datasets.yaml.",
    )
    ap.add_argument(
        "--name", default=DEFAULT_NAME, help=f"dataset to build (default {DEFAULT_NAME})"
    )
    ap.add_argument(
        "--rows",
        type=Path,
        nargs="+",
        help=f"JSONL row files (default: {_rel(paths.INTERIM_DIR)}/*.jsonl)",
    )
    ap.add_argument("--datasets", type=Path, default=paths.DATASETS_PATH)
    ap.add_argument("--out-dir", type=Path, default=paths.PROCESSED_DIR)
    args = ap.parse_args(argv)
    files = args.rows if args.rows is not None else sorted(paths.INTERIM_DIR.glob("*.jsonl"))
    try:
        return build(args.name, files, args.datasets, args.out_dir)
    except DatasetError as e:
        print(f"make dataset: {e}", file=sys.stderr)
        return 2


def _entry(datasets: Datasets, name: str) -> Dataset:
    if name not in datasets.datasets:
        listed = ", ".join(datasets.datasets) or "none"
        raise DatasetError(f"{name} isn't in datasets.yaml (listed: {listed})")
    return datasets.datasets[name]


def _issue(issue: Issue) -> str:
    return f"[{issue.check}] {issue.message}"


# one line per check: how many, and the first one as an example
def _by_check(issues: list[Issue], what: str) -> str:
    groups: dict[int, list[Issue]] = {}
    for issue in issues:
        groups.setdefault(issue.check, []).append(issue)
    lines = [f"[{c}] x{len(g)}, e.g. {g[0].message}" for c, g in sorted(groups.items())]
    return f"{len(issues)} {what}:\n" + "\n".join(f"  {line}" for line in lines)


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _shown(lines: list[str], what: str) -> str:
    shown = "\n".join(f"  {line}" for line in lines[:MAX_SHOWN])
    more = f"\n  ... and {len(lines) - MAX_SHOWN} more" if len(lines) > MAX_SHOWN else ""
    return f"{len(lines)} {what}:\n{shown}{more}"


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(paths.REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


if __name__ == "__main__":
    sys.exit(main())
