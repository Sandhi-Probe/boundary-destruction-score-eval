# ADR-002: Raw data is rebuilt from a checksummed manifest, not DVC

- **Status:** Proposed
- **Date:** 2026-11-01
- **Deciders:** @jeojdi1, @TejasviniChawla, @Impure-King
- **Issue:** #17

## Context

Raw corpora don't go in git, but a fresh clone has to rebuild `data/raw/` byte for byte, every time, or every number we report is in doubt. We need to decide how.

What we're storing, from #21:

- **DCS** (Digital Corpus of Sanskrit) lives in the GitHub repo `OliverHellwig/sanskrit`, licensed CC BY 4.0. At commit `8aeed5a` its CoNLL-U folder has 23,279 files across 271 texts, 1.1 GB in total, the largest 3 MB. The sample Sparsh pulled for #21 is 6 MB.
- **UoH** (DTCS corpus, University of Hyderabad) is a set of HTML pages on `sanskrit.uohyd.ac.in`. There is no versioning and no site-wide licence, and redistribution is unclear (#21).

Two things can break a rebuild:

1. **Upstream changes.** DCS `master` moves, and UoH pages can be edited in place. A link to `.../tree/master/...` gives different bytes, or an HTML page instead of the file, as soon as upstream pushes.
2. **Upstream goes away**, or you're offline.

There's also a licence constraint: until #21 settles UoH redistribution, we shouldn't put a copy of UoH files anywhere we host, even privately.

## Options considered

| Option | For | Against | Evidence |
|---|---|---|---|
| **A. DVC with a remote** (S3, Google Drive, SSH) | Versioned data, a shared cache, pipeline stages, and a copy that survives upstream disappearing | Someone has to host and pay for the remote and share credentials with every member. The Google Drive remote needs OAuth setup per person. The remote is a re-hosted copy of UoH, which is the redistribution question #21 left open. Adds `dvc` and its dependencies to `uv.lock`, plus a second tool (`dvc pull`) everyone has to learn. | #21 licence notes |
| **B. DVC `import-url`, no remote** | Tracks the upstream URL without hosting anything | Still adds DVC. With no remote it has the same upstream dependency as C, and we'd get DVC's complexity without its main benefit. | |
| **C. Plain manifest checked by `make data`** | No new dependencies (`pyyaml` and `pydantic` are already in). Nothing is hosted, so nothing is re-distributed. The manifest is a readable file that reviewers can diff. Verifying 6 MB takes under a second. | We maintain the downloader (one module, about 600 lines, with tests for offline, dropped connections, retries and bad entries). It depends on upstream staying up. Fetching files one at a time doesn't scale to all 23k DCS files. | `tests/data/test_manifest.py`; all 10 #21 files downloaded with matching checksums, then verified with the network blocked |
| **D. Git LFS / commit the data** | Simplest for readers | Puts UoH files in a public repo, and the free-org LFS quota gets used up by every clone. | |

## Decision

We will list every file in `data/raw/` in `data/manifest.yaml`, with its source URL, SHA-256 and retrieval date. `make data` downloads anything missing and fails if any checksum doesn't match (option C).

The convention:

- **`path`** is relative to `data/raw/`, and its first folder is the corpus (`dcs/`, `uoh/`). Each name uses only lowercase ASCII letters, digits, `.`, `_` and `-`, with no leading `.` and at most 100 characters. macOS treats `DCS/` and `dcs/`, or the two Unicode forms of an accented name like `Kumārasaṃbhava`, as one name while Linux treats them as two, so anything looser can mean different files on different laptops.
- **`url`** is the exact file. GitHub links go through `raw.githubusercontent.com`, pinned to a full commit SHA. The loader rejects branches, tags and short SHAs in those links, `blob/`/`tree/` page links, and URLs with credentials in them.
- **`sha256`** is taken over the bytes as downloaded. **`retrieved`** is the day it was taken.
- Only `raw/` is in the manifest. `interim/` and `processed/` are derived from it, so they aren't listed or checksummed. [data/README.md](../../data/README.md) describes each folder.

`make data` behaviour:

- **Existing files:** a file that's already present is never downloaded again. If it doesn't match, it's reported, not replaced, because a person should look at it: either the file changed, or the manifest entry did (after a pull).
- **Downloads:** each file is written to a `.part` temp file and only moved into place once its checksum matches. A dropped connection, a captive wifi login page or a changed upstream never leaves a bad file behind. A run, `OFFLINE=1` included, holds a lock on `data/raw/`, so two runs can't collide, and it removes `.part` files left by a run that was killed.
- **Offline:** `make data` runs with `uv run --no-sync`, so a pull that changes the dependencies can't make it need the network. Files that are already there still get verified. A missing file fails fast, and the other files from the same host are skipped instead of each waiting for a timeout. `make data OFFLINE=1` skips downloads and never touches the network.
- **Retries:** only errors after the connection was made are retried (dropped connection, no response, 5xx, 429), 3 attempts, honouring a `Retry-After` in seconds up to 30 s. If a host still fails, its remaining files are skipped for that run. A 404 or a checksum mismatch fails immediately.

## Consequences

- **What gets easier:** a fresh clone is `make setup && make data`. If a teammate's numbers differ from yours, the raw data is ruled out by `make data OFFLINE=1`. Reviewers see data changes as a diff to `manifest.yaml`.
- **What gets harder:** if upstream changes a file, `make data` fails on new clones until someone checks the change and updates the entry (new `sha256` and `retrieved`) in a PR. That is on purpose: a silent change to the data is the thing this exists to catch. People with an existing copy won't notice until they re-download, so a scheduled CI job that rebuilds from scratch would catch drift early (follow-up, not needed yet).
- **What we give up:** if UoH takes a page down, new members can't download it. Anyone who already has a verified copy can share it out of band, and `make data` will verify it.
- **Follow-up:** #21 adds the real DCS and UoH entries. #23 adds the extraction step to `make data`.

## Revisit if

- We need more than a few hundred DCS files (the whole CoNLL-U set is 23,279 files). The fix would be one manifest entry for a pinned sparse git checkout or archive instead of one entry per file.
- A source changes or disappears more than once.
- We start keeping outputs that are expensive to rebuild, such as model activations for the probes. That data needs a remote, and DVC (option A) becomes worth its cost.
