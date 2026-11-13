# configs/

YAML files for experiment settings (which tokenizers, seeds, dataset version), so a run is reproducible from its config alone.

## tokenizers.yaml

The tokenizers we evaluate, one entry each: Hugging Face id pinned to a commit, family, vocab size, gated, licence, offset support, normalization, and notes for the harness. Code reads it through `sandhi_probe.tokenizers.inventory` (`load_inventory()`, `get("byt5")`, `load(spec)`); don't hard-code model ids anywhere else.

```bash
uv run python -m sandhi_probe.tokenizers.inventory           # validate and list
uv run python -m sandhi_probe.tokenizers.inventory --verify  # check every field on the Hub (network)
```

None of the current entries is gated. If you add one that is (Llama, for example), `--verify` skips it until you have access: request it on the model page, then `hf auth login`.
