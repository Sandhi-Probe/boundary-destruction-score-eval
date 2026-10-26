"""Repo-relative paths, so no script hard-codes a location on someone's laptop."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # downloaded corpora, untouched
INTERIM_DIR = DATA_DIR / "interim"  # cleaned / re-encoded, not final
PROCESSED_DIR = DATA_DIR / "processed"  # versioned datasets, e.g. v0.1.parquet

CONFIGS_DIR = REPO_ROOT / "configs"
