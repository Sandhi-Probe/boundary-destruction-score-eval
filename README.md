# sandhi-probe

How much do tokenizers destroy Sanskrit sandhi boundaries? We build a dataset of (fused, split) minimal pairs and score tokenizers with the **Sandhi Boundary Destruction Score (SBDS)**.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). It installs the right Python for you.

```bash
git clone https://github.com/Sandhi-Probe/boundary-destruction-score-eval.git
cd boundary-destruction-score-eval
make setup   # .venv with the exact versions in uv.lock
make data    # download the raw corpora and verify their checksums
make check   # lint + tests; run before every PR
```

Run `make` on its own to list every target. Before your first PR, read [CONTRIBUTING.md](CONTRIBUTING.md).

## Running an experiment

Everyone runs the same code on the same data with the same tokenizers, so results from different laptops can be compared.

**Once per laptop:** install uv (above), then `make setup`.

**Before every experiment:**

```bash
git pull     # latest code, configs/tokenizers.yaml and data/manifest.yaml
make setup   # match .venv to uv.lock again; a pull may have changed the dependencies
make data    # fetch any new raw files and verify every checksum
```

Tokenizers are loaded at the Hugging Face commit pinned in [`configs/tokenizers.yaml`](configs/tokenizers.yaml), so everyone gets identical tokenizers. Load them through `sandhi_probe.tokenizers.inventory`, never by model id.

## Layout

```
src/sandhi_probe/
  encoding/     Devanagari / IAST / SLP1 / HK / WX <-> canonical encoding
  data/         corpus loading, minimal-pair schema, extraction
  tokenizers/   one interface over GPT-2, Gemma 4, mT5, ByT5 (list: configs/tokenizers.yaml)
  metrics/      fertility, boundary alignment, SBDS
  probes/       probing classifiers on model activations
  paths.py      repo-relative paths; import these, don't hard-code
tests/          pytest; mirrors src/
data/           manifest.yaml lists raw/; raw -> interim -> processed are gitignored
notebooks/      exploration only; results move into src/ with a test
configs/        experiment settings (YAML); tokenizers.yaml is the tokenizer list
docs/adr/       decision records: why we chose what we chose
docs/meetings/  sync notes
```

## Adding a dependency

`uv add <package>` (or `uv add --dev <package>` for tooling), then commit both `pyproject.toml` and `uv.lock`.
