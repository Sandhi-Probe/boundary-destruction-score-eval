# data/

Nothing in here is committed except this README and `manifest.yaml`. `make data` rebuilds the rest from the manifest's source URLs and checksums. Why it works this way: [ADR-002](../docs/adr/002-data-storage.md).

| Folder | What goes in it | Checked by |
|---|---|---|
| `raw/` | Corpora exactly as downloaded (DCS, UoH), one folder per corpus. Never edited by hand. | `manifest.yaml`: every listed file is verified; unlisted files get a warning |
| `interim/` | Cleaned or re-encoded intermediates. Safe to delete. | nothing; rebuild it |
| `processed/` | Versioned datasets the experiments read, e.g. `v0.1.parquet`, built by `make data` (#23). | rebuilt deterministically from `raw/` + code |

## Commands

```bash
make data            # download anything missing, then verify every checksum
make data OFFLINE=1  # verify only; never touches the network
```

Both work offline once `make setup` has run, even after a `git pull` that changed the dependencies. `make data` checks every file, reports any that don't match, and exits with an error. It never overwrites a file that's already in `raw/`.

## Adding a raw file

1. Find the exact file URL. For GitHub, pin it to a commit with `https://raw.githubusercontent.com/<owner>/<repo>/<commit-sha>/<path>`, not a `tree/` or `blob/` page or a branch name. Those change under you.
2. Download it and take the checksum: `shasum -a 256 <file>`.
3. Add an entry under `files:` in `manifest.yaml`, using the example at the top of that file. The `path` is relative to `raw/` and uses lowercase ASCII only, so `dcs/conllu/kumarasambhava-01.conllu`, not the original name with diacritics. Put the URL in double quotes.
4. Delete your local copy and run `make data`. It should download the file and pass.

## When `make data` fails

| Output | What it means |
|---|---|
| `make data: no .venv yet` | Run `make setup` once while you're online. |
| `can't resolve <host> ... you may be offline` | No network, or a typo in the host. Files already present were still verified. Run it again when you're online. |
| `TLS certificate ... isn't trusted` | Usually a hotel or campus wifi login page. Log in through a browser, or try another network. |
| `MISMATCH` on a file you already had | Your copy doesn't match the manifest. If you just pulled or switched branches, the entry probably changed: see `git log -p data/manifest.yaml`. Move the file out of `raw/` instead of deleting it (it may be your only copy), then run `make data` online. |
| `doesn't match the manifest` right after a download | A wifi login page was served instead of the file, or upstream changed it. Try another network. If it still fails, open an issue so someone can check the change and update the entry. For a pinned GitHub commit, the entry itself is wrong. |
| `HTTP 404` | The URL is wrong, or the file or commit is gone upstream. |
| `kept failing earlier in this run` (`HTTP 429`, `5xx`, no response) | The server is overloaded or rate-limiting you. That host's other files were skipped. Wait a while, then re-run: finished downloads are kept. |
| `another make data is running` | Wait for the other run to finish. |
| `warning ... not in the manifest` | A file in `raw/` that a fresh clone won't have. Add it to the manifest, or leave it if it belongs to another branch. |
