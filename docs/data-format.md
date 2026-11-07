# Data format: minimal pairs

The schema for the minimal-pair dataset (#22). The typed model is `src/sandhi_probe/data/schema.py`.

One row = one fused string from the corpus. Each row has a list of junctions, one per join, since one fused string can have several joins.

All text and positions are in SLP1 (team decision).

## Row fields

| field | type | meaning | example |
|---|---|---|---|
| `id` | str | unique id for the row | `dcs-000123` |
| `fused` | str | the text after sandhi. Keeps the space for sandhi across words | `tadapi`, `rAma udakam` |
| `split_parts` | list[str] | the original words before sandhi | `["tat", "api"]` |
| `alt_splits` | list[list[str]] | other valid splits, if `fused` is ambiguous. Plain word lists, no junctions | `[["rAmA", "iti"]]` for `rAmeti` |
| `junctions` | list[Junction] | one entry per join (see below). Labels always describe `split_parts` | |
| `encoding` | str | script the text is in, always SLP1 | `slp1` |
| `source_corpus` | str | which corpus the row came from | `DCS` or `UoH` |
| `source_ref` | str | exact location in the source, with corpus version | `dcs-2022/text-123/sent-56/tok-3` |
| `sentence_id` | str | the sentence the row came from. Both halves of a DCS verse share one id. Used by the split rule | `dcs-2022/text-123/sent-56` |
| `split_provenance` | str | who made the split: a human (`gold`), DCS reconstructed it (`reconstructed`), or our code (`derived`) | `gold` |
| `frequency` | int | how many times `fused` shows up in the corpus | `4120` |
| `frequency_bucket` | str or null | high / mid / low from `frequency`, filled once we pick cutoffs | `high` |
| `is_control` | bool | true for control rows where nothing changes at the join | `false` |
| `split_assignment` | str | train / dev / test | `train` |

## Junction fields

| field | type | meaning | example (`tadapi`) |
|---|---|---|---|
| `span` | [int, int] | position of the letters sandhi changed in `fused`, as [start, end). Zero-width when nothing changed, e.g. `[4, 4]` | `[2, 3]` (the `d`) |
| `cut` | int or null | where the first word ends in `fused`. null if one letter belongs to both words | `3` |
| `underlying_left` | str | letters from the left word before sandhi | `t` |
| `underlying_right` | str | letters from the right word before sandhi | `""` |
| `surface` | str | what those letters became | `d` |
| `rule_class` | str | vowel / visarga / consonant, or none if nothing changed | `consonant` |
| `rule_ids` | list[str] | exact rules applied, keys from the rules table below. Empty for control rows | `["word_final_voicing"]` |
| `type` | str | `compound` (join inside a word) or `external` (join between two words) | `external` |
| `boundary_level` | computed | intact / changed / merged (see below) | `changed` |

`boundary_level` is computed from the other fields:

- **intact**: nothing changed. `rAma + gfham → rAmagfham`
- **changed**: letters changed, but you can still cut. `tat + api → tadapi`
- **merged**: one letter belongs to both words. `sItA + udakam → sItodakam`

A junction is "destroyed" when `boundary_level != "intact"`.

## Rules table

Each `rule_ids` key maps to its group and its Pāṇini sūtra. The list grows as extraction covers more rules.

| key | group | sūtra | example |
|---|---|---|---|
| `guna` | vowel | 6.1.87 | `a + i → e` |
| `savarna_dirgha` | vowel | 6.1.101 | `a + a → A` |
| `yan` | vowel | 6.1.77 | `i → y` before a vowel |
| `word_final_voicing` | consonant | 8.2.39 | `tat + api → tadapi` |

## Split rule

- **Rule:** all rows from the same sentence go in the same split.
- **Why:** so the test set never has a sentence the model already saw in training.
- **How:** hash `sentence_id` to pick train / dev / test (80/10/10). Same sentence always gets the same split, and adding data later doesn't reshuffle old rows.
- **Repeated words:** a common `fused` form like `tadapi` can still land in more than one split. That's fine for SBDS. The probe experiments drop any row whose `fused` form already appears in train.

## Still open

- Scoring criterion (#41) may add fields.
- UoH `source_ref` and `sentence_id` format: TBD once #21 has the data.
