# sandhi-probe

How much do tokenizers destroy Sanskrit sandhi boundaries? We build a dataset of (fused, split) minimal pairs and score tokenizers with the **Sandhi Boundary Destruction Score (SBDS)**.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). It installs the right Python for you.

```bash
git clone https://github.com/Sandhi-Probe/boundary-destruction-score-eval.git
cd boundary-destruction-score-eval
make setup   # .venv with the exact versions in uv.lock
make check   # lint + tests; run before every PR
```

Run `make` on its own to list every target. Before your first PR, read [CONTRIBUTING.md](CONTRIBUTING.md).

## Layout

```
src/sandhi_probe/
  encoding/     Devanagari / IAST / SLP1 / HK / WX <-> canonical encoding
  data/         corpus loading, minimal-pair schema, extraction
  tokenizers/   one interface over GPT-2, Qwen, ByT5, ...
  metrics/      fertility, boundary alignment, SBDS
  probes/       probing classifiers on model activations
  paths.py      repo-relative paths; import these, don't hard-code
tests/          pytest; mirrors src/
data/           raw -> interim -> processed; gitignored, rebuilt by `make data`
notebooks/      exploration only; results move into src/ with a test
configs/        experiment settings (YAML)
docs/adr/       decision records: why we chose what we chose
docs/meetings/  sync notes
```

## Adding a dependency

`uv add <package>` (or `uv add --dev <package>` for tooling), then commit both `pyproject.toml` and `uv.lock`.
