"""`make dataset`: same rows give the same bytes, the checksum is enforced, bad rows stop it."""

import json
import os
import random
import stat
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from sandhi_probe import paths
from sandhi_probe.data import dataset
from sandhi_probe.data.dataset import (
    SCHEMA,
    DatasetError,
    load_datasets,
    main,
    read_pairs,
    read_rows,
    sha256,
    to_table,
    write_parquet,
)
from sandhi_probe.data.schema import Junction, Pair

# The rows below, written with WRITE_OPTIONS by the pyarrow in uv.lock. If this changes after a
# pyarrow upgrade, every dataset checksum changes too: update data/datasets.yaml in the same PR.
# If it fails on one OS only, the build isn't reproducible across machines.
KNOWN_SHA256 = "9925f726caf82f6e85e3670f1493319c1a3465edce679501eb487cc06f466282"

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


def make_pair(fused: str, split_parts: list[str], junction: Junction, n: int, **overrides) -> Pair:
    fields = dict(
        id=f"dcs-{n:06d}",
        fused=fused,
        split_parts=split_parts,
        junctions=[junction],
        source_corpus="DCS",
        source_ref=f"dcs-2022/text-1/sent-{n}/tok-1",
        sentence_id=f"dcs-2022/text-1/sent-{n}",
        split_provenance="gold",
        frequency=n,
        split_assignment="train",
    )
    return Pair(**(fields | overrides))


ROWS = [
    make_pair("tadapi", ["tat", "api"], TADAPI, 1),
    make_pair("rAmagfham", ["rAma", "gfham"], INTACT, 2, is_control=True),
    make_pair(
        "sItodakam",
        ["sItA", "udakam"],
        MERGED,
        3,
        alt_splits=[["sItA", "uda", "kam"]],
        frequency_bucket="low",
        split_assignment="test",
    ),
]


def write_rows(path: Path, pairs: list[Pair]) -> Path:
    path.write_text("".join(p.model_dump_json() + "\n" for p in pairs), encoding="utf-8")
    return path


def write_datasets(path: Path, sha: str | None = None, rows: int | None = None) -> Path:
    entry = {"sha256": sha, "rows": rows}
    path.write_text(json.dumps({"datasets": {"v0.1": entry}}), encoding="utf-8")
    return path


def run(tmp_path: Path, rows: list[Path], sha: str | None = None, n: int | None = None) -> int:
    datasets = write_datasets(tmp_path / "datasets.yaml", sha, n)
    out = tmp_path / "processed"
    return main(["--rows", *map(str, rows), "--datasets", str(datasets), "--out-dir", str(out)])


def built_sha(tmp_path: Path, pairs: list[Pair], name: str = "built") -> str:
    out = tmp_path / f"{name}.parquet"
    write_parquet(to_table(pairs), out)
    return sha256(out)


# --- the file layout -------------------------------------------------------------------------


def test_columns_follow_the_model() -> None:
    assert SCHEMA.names == list(Pair.model_fields)
    junction = SCHEMA.field("junctions").type.value_type
    assert [f.name for f in junction] == [*Junction.model_fields, "boundary_level"]


def test_rows_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "v.parquet"
    write_parquet(to_table(list(reversed(ROWS))), out)
    assert read_pairs(out) == ROWS  # sorted by id, every field back as it was
    table = pq.read_table(out)
    assert table.column("id").to_pylist() == ["dcs-000001", "dcs-000002", "dcs-000003"]
    levels = [j[0]["boundary_level"] for j in table.column("junctions").to_pylist()]
    assert levels == ["changed", "intact", "merged"]


def test_same_rows_give_the_same_bytes_in_any_order(tmp_path: Path) -> None:
    shuffled = list(ROWS)
    random.Random(0).shuffle(shuffled)
    assert shuffled != ROWS
    assert built_sha(tmp_path, ROWS, "a") == built_sha(tmp_path, shuffled, "b")


def test_known_rows_give_known_bytes(tmp_path: Path) -> None:
    assert built_sha(tmp_path, ROWS) == KNOWN_SHA256


def test_a_changed_row_changes_the_bytes(tmp_path: Path) -> None:
    changed = [ROWS[0].model_copy(update={"frequency": 99}), *ROWS[1:]]
    assert built_sha(tmp_path, ROWS, "a") != built_sha(tmp_path, changed, "b")


# --- make dataset ----------------------------------------------------------------------------


def test_first_build_prints_the_checksum_to_commit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = write_rows(tmp_path / "rows.jsonl", ROWS)
    assert run(tmp_path, [rows]) == 0
    out = capsys.readouterr().out
    built = tmp_path / "processed" / "v0.1.parquet"
    assert "no checksum committed for v0.1 yet" in out
    assert f"sha256: {sha256(built)}" in out
    assert "rows: 3" in out
    assert read_pairs(built) == ROWS


def test_matching_checksum_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = write_rows(tmp_path / "rows.jsonl", ROWS)
    sha = built_sha(tmp_path, ROWS)
    assert run(tmp_path, [rows], sha, 3) == 0
    assert "make dataset: ok" in capsys.readouterr().out
    assert sha256(tmp_path / "processed" / "v0.1.parquet") == sha


def test_rows_split_across_files_give_the_same_file(tmp_path: Path) -> None:
    a = write_rows(tmp_path / "a.jsonl", ROWS[2:])
    b = write_rows(tmp_path / "b.jsonl", ROWS[:2])
    assert run(tmp_path, [a, b], built_sha(tmp_path, ROWS), 3) == 0


def test_mismatch_keeps_the_old_file_and_the_new_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = write_rows(tmp_path / "rows.jsonl", ROWS)
    out = tmp_path / "processed"
    out.mkdir()
    (out / "v0.1.parquet").write_bytes(b"previous build")
    assert run(tmp_path, [rows], "0" * 64, 3) == 1
    printed = capsys.readouterr().out
    assert "MISMATCH" in printed
    assert (out / "v0.1.parquet").read_bytes() == b"previous build"
    assert read_pairs(out / "v0.1.mismatch.parquet") == ROWS
    assert not list(out.glob(".*.part"))


def test_warnings_are_summed_up_per_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    no_rule = ROWS[0].model_copy(update={"junctions": [TADAPI.model_copy(update={"rule_ids": []})]})
    rows = write_rows(tmp_path / "rows.jsonl", [no_rule, *ROWS[1:]])
    assert run(tmp_path, [rows]) == 0
    assert "  [10] x1, e.g. dcs-000001: changed join" in capsys.readouterr().out


def test_built_file_follows_the_umask(tmp_path: Path) -> None:
    rows = write_rows(tmp_path / "rows.jsonl", ROWS)
    old = os.umask(0o022)
    try:
        assert run(tmp_path, [rows]) == 0
    finally:
        os.umask(old)
    built = tmp_path / "processed" / "v0.1.parquet"
    assert stat.S_IMODE(built.stat().st_mode) == 0o644


def test_invalid_row_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = ROWS[0].model_copy(update={"split_parts": ["tat", "iti"]})  # doesn't rebuild
    rows = write_rows(tmp_path / "rows.jsonl", [broken, *ROWS[1:]])
    assert run(tmp_path, [rows]) == 1
    printed = capsys.readouterr().out
    assert "1 validation error(s); nothing written" in printed
    assert "[7] dcs-000001" in printed
    assert not (tmp_path / "processed").exists() or not any((tmp_path / "processed").iterdir())


def test_duplicate_ids_across_files_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = write_rows(tmp_path / "a.jsonl", ROWS)
    b = write_rows(tmp_path / "b.jsonl", ROWS[:1])
    assert run(tmp_path, [a, b]) == 1
    assert "validation error(s)" in capsys.readouterr().out


def test_bad_line_names_the_file_and_line(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text(ROWS[0].model_dump_json() + "\n\n" + '{"id": "dcs-9"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match=r"rows\.jsonl:3: fused: Field required"):
        read_rows([rows])


def test_schema_error_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text("not json\n", encoding="utf-8")
    assert run(tmp_path, [rows]) == 2
    assert "rows.jsonl:1" in capsys.readouterr().err


def test_no_row_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(paths, "INTERIM_DIR", tmp_path / "interim")
    datasets = write_datasets(tmp_path / "datasets.yaml")
    assert main(["--datasets", str(datasets), "--out-dir", str(tmp_path / "out")]) == 1
    assert "no row files in" in capsys.readouterr().out


def test_unknown_name_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = write_rows(tmp_path / "rows.jsonl", ROWS)
    datasets = write_datasets(tmp_path / "datasets.yaml")
    assert main(["--name", "v9.9", "--rows", str(rows), "--datasets", str(datasets)]) == 2
    assert "v9.9 isn't in datasets.yaml (listed: v0.1)" in capsys.readouterr().err


# --- datasets.yaml and load_pairs ------------------------------------------------------------


def test_committed_datasets_yaml_lists_v0_1() -> None:
    assert "v0.1" in load_datasets(paths.DATASETS_PATH).datasets


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"sha256": "abc", "rows": 3}, "64 lowercase hex"),
        ({"sha256": "a" * 64, "rows": None}, "together"),
        ({"sha256": None, "rows": 3}, "together"),
        ({"sha256": "a" * 64, "rows": 0}, "greater than or equal to 1"),
        ({"sha256": None, "rows": None, "path": "x"}, "Extra inputs"),
    ],
)
def test_datasets_yaml_rejects_bad_entries(tmp_path: Path, entry: dict, message: str) -> None:
    path = tmp_path / "datasets.yaml"
    path.write_text(json.dumps({"datasets": {"v0.1": entry}}), encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_datasets(path)


def test_datasets_yaml_rejects_bad_names(tmp_path: Path) -> None:
    path = tmp_path / "datasets.yaml"
    path.write_text(json.dumps({"datasets": {"latest": {}}}), encoding="utf-8")
    with pytest.raises(DatasetError, match="look like v0.1"):
        load_datasets(path)


def test_load_pairs_checks_the_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path)
    write_parquet(to_table(ROWS), tmp_path / "v0.1.parquet")
    good = write_datasets(tmp_path / "good.yaml", sha256(tmp_path / "v0.1.parquet"), 3)
    assert dataset.load_pairs("v0.1", datasets_path=good) == ROWS
    stale = write_datasets(tmp_path / "stale.yaml", "0" * 64, 3)
    with pytest.raises(DatasetError, match="not the committed"):
        dataset.load_pairs("v0.1", datasets_path=stale)
    unset = write_datasets(tmp_path / "unset.yaml")
    with pytest.raises(DatasetError, match="no checksum committed"):
        dataset.load_pairs("v0.1", datasets_path=unset)
