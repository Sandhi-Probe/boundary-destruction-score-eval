# data/

Only this README, `manifest.yaml` and the `.gitkeep` files are committed. `make data` rebuilds `raw/` from the manifest's URLs and checksums. Why it works this way: [ADR-002](../docs/adr/002-data-storage.md).

| Folder | What goes in it | Checked by |
|---|---|---|
| `raw/` | Corpora exactly as downloaded (DCS, UoH), one folder per corpus. Never edited by hand. | `manifest.yaml`: every listed file is verified; unlisted files get a warning |
| `interim/` | Cleaned or re-encoded intermediates. Safe to delete. | nothing; rebuild it |
| `processed/` | Versioned datasets the experiments read, e.g. `v0.1.parquet`. `make data` will build them (#23). | rebuilt deterministically from `raw/` + code |

## Commands

```bash
make data            # download anything missing, verify every checksum
make data OFFLINE=1  # verify only; never touches the network
```

Both run offline once `make setup` has run, even after a `git pull` that changed the dependencies, though a missing file then fails. Both report every file that doesn't match and exit with an error. Neither overwrites a file that's already in `raw/`. Both delete `.part` files left by a killed run and leave `raw/.make-data.lock` behind; ignore it.

## Adding a raw file

1. Find the exact file URL. For GitHub, use `https://raw.githubusercontent.com/<owner>/<repo>/<commit-sha>/<path>` with the full SHA. `make data` rejects `tree/` and `blob/` pages and branch names.
2. Download it and take the checksum: `shasum -a 256 <file>`.
3. Add an entry under `files:` in `manifest.yaml`, using the example at the top of that file. The `path` is lowercase ASCII only, so `dcs/conllu/kumarasambhava-01.conllu`, not the original name with diacritics.
4. Delete your local copy and run `make data`. It should download the file and pass.

## When `make data` fails

| Output | What it means |
|---|---|
| `make data: no .venv yet` | Run `make setup` once while you're online. |
| `can't resolve <host> ... you may be offline`, `was unreachable earlier in this run` | No network, or a typo in the host. That host's other files were skipped. Run it again when you're online. |
| `TLS certificate ... isn't trusted` | Usually a hotel or campus wifi login page. Log in through a browser, or try another network. |
| `MISMATCH` on a file you already had | If you just pulled or switched branches, the entry probably changed: see `git log -p data/manifest.yaml`. Move the file out of `raw/` instead of deleting it (it may be your only copy), then run `make data` online. |
| `doesn't match the manifest` right after a download | A wifi login page was served instead of the file, or upstream changed it. Try another network. If it still fails, open an issue so someone can check the change and update the entry. For a pinned GitHub commit, the entry itself is wrong. |
| `HTTP 404` | The URL is wrong, or the file or commit is gone upstream. |
| `kept failing earlier in this run` (`HTTP 429`, `5xx`, no response) | The server is overloaded or rate-limiting you. That host's other files were skipped. Wait a while, then re-run: finished downloads are kept. |
| `` another `make data` is running `` | Wait for the other run to finish. |
| `warning ... not in the manifest` | A file in `raw/` that a fresh clone won't have. Add it to the manifest, or leave it if it belongs to another branch. |
