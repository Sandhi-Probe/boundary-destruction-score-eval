"""Checks that the package installs and the repo layout is where code expects it."""

import importlib

import pytest

from sandhi_probe import paths


@pytest.mark.parametrize("module", ["data", "encoding", "tokenizers", "metrics", "probes"])
def test_subpackages_import(module: str) -> None:
    importlib.import_module(f"sandhi_probe.{module}")


def test_data_dirs_exist() -> None:
    for d in (paths.RAW_DIR, paths.INTERIM_DIR, paths.PROCESSED_DIR):
        assert d.is_dir(), f"missing {d.relative_to(paths.REPO_ROOT)}"
