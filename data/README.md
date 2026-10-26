# data/

Nothing in here is committed. `make data` rebuilds it from a manifest of source URLs and checksums (#17).

| Folder | What goes in it |
|---|---|
| `raw/` | Corpora exactly as downloaded (DCS, UoH). Never edited by hand. |
| `interim/` | Cleaned or re-encoded intermediates. Safe to delete. |
| `processed/` | Versioned datasets the experiments read, e.g. `v0.1.parquet`. |
