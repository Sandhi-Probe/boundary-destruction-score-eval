# configs/

YAML files for experiment settings (which tokenizers, seeds, dataset version), so a run is reproducible from its config alone.

## tokenizers.yaml

The tokenizers we evaluate, one entry each: Hugging Face id pinned to a commit, family, vocab size, gated, licence, offset support, normalization, and notes for the harness. Code reads it through `sandhi_probe.tokenizers.inventory` (`load_inventory()`, `get("byt5")`, `load(spec)`); don't hard-code model ids anywhere else.

```bash
uv run python -m sandhi_probe.tokenizers.inventory           # validate and list
uv run python -m sandhi_probe.tokenizers.inventory --verify  # check every field on the Hub (network)
```

None of the current entries is gated. If you add one that is (Llama, for example), `--verify` skips it until you have access: request it on the model page, then `hf auth login`.

## Tokenizing text

Experiments get token spans through the harness, never from a tokenizer directly:

```python
from sandhi_probe.tokenizers.harness import tokenize

r = tokenize("gpt2", "tadapi")
r.spans  # ((0, 1), (1, 3), (3, 6)): (start, end) of each token in the text
r.inside_char_cuts  # token boundaries inside one character (byte-level cuts in Devanagari, IAST)
```

Spans always cover the whole string, in order, with no gaps or overlaps, and point into the text you passed in. The rules behind that (leading-space markers, mT5's normalization, cuts inside characters) are in the docstring of `src/sandhi_probe/tokenizers/harness.py`.
