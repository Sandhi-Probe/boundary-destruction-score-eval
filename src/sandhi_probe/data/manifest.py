"""Raw-data manifest: where every file in data/raw/ comes from, and its SHA-256.

`make data` runs this module. It downloads any manifest file that is missing from
data/raw/, then checks the checksum of every file, so a fresh clone rebuilds data/raw/
byte for byte or fails. Why a plain manifest and not DVC: docs/adr/002-data-storage.md.

    uv run python -m sandhi_probe.data.manifest            # download missing, verify all
    uv run python -m sandhi_probe.data.manifest --offline  # verify only, never download
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import io
import os
import re
import signal
import socket
import ssl
import stat
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path, PurePosixPath
from typing import IO

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from sandhi_probe import __version__, paths

USER_AGENT = (
    f"sandhi-probe/{__version__} (+https://github.com/Sandhi-Probe/boundary-destruction-score-eval)"
)
TIMEOUT_S = 30.0  # per socket operation, so a dead connection fails instead of hanging
ATTEMPTS = 3  # only for errors after the server answered: dropped connection, 5xx, 429
MAX_WAIT_S = 30.0  # cap on a server's Retry-After
CHUNK = 1 << 20
LOCK_NAME = ".make-data.lock"
MAX_WARNINGS = 20

# Lowercase ASCII only. macOS stores "ṃ" decomposed and Linux composed, and macOS treats
# DCS/ and dcs/ as one folder while Linux doesn't, so anything else can name different
# files on different laptops. A leading "." is reserved for temp and lock files.
_SEGMENT = re.compile(r"[a-z0-9_-][a-z0-9._-]{0,99}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_CONFLICT = re.compile(r"^(<{7}|={7}|>{7})( |$)", re.MULTILINE)
_GITHUB = {"github.com", "www.github.com"}
_JUNK = {".gitkeep", ".DS_Store", "Thumbs.db", "desktop.ini", LOCK_NAME}


# --- the manifest ----------------------------------------------------------------------------


class ManifestError(Exception):
    """The manifest file is missing, unreadable or invalid."""


class RawFile(BaseModel):
    """One file in data/raw/, exactly as downloaded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str  # relative to data/raw/, first segment is the corpus: dcs/..., uoh/...
    url: str
    sha256: str
    retrieved: date  # the day the checksum was taken

    @field_validator("path")
    @classmethod
    def _check_path(cls, v: str) -> str:
        parts = v.split("/")
        if v.endswith(".part") or not all(
            _SEGMENT.fullmatch(p) and not p.endswith(".") for p in parts
        ):
            raise ValueError(
                f"{v!r}: use a relative path under data/raw/ made of lowercase a-z, 0-9, '.', '_'"
                " and '-' (no leading '.', at most 100 characters per name),"
                " e.g. dcs/conllu/kumarasambhava-01.conllu"
            )
        return v

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, v: str) -> str:
        v = v.strip().lower()
        if not _SHA256.fullmatch(v):
            raise ValueError(f"{v!r} is not a SHA-256 hex digest (64 characters, 0-9 a-f)")
        return v

    @field_validator("url")
    @classmethod
    def _check_url(cls, v: str) -> str:
        if v != v.strip() or any(c < " " or c == "\x7f" for c in v):
            raise ValueError(f"{v!r}: keep the URL on one line, in double quotes, no spaces around")
        bad = ValueError(f"{v!r}: must be an http(s) URL with a valid host")
        try:
            u = urllib.parse.urlsplit(v)
            u.port  # noqa: B018 (raises ValueError on a bad port)
            host = (u.hostname or "").rstrip(".")
            host.encode("idna")  # raises UnicodeError on an empty or over-long label
        except (ValueError, UnicodeError):
            raise bad from None
        if u.scheme not in ("http", "https") or not host:
            raise bad
        if u.username or u.password:
            raise ValueError(f"{v!r}: no credentials in URLs, the manifest is committed to git")
        parts = [p for p in u.path.split("/") if p]
        if host in _GITHUB and len(parts) >= 3 and parts[2] in ("blob", "tree"):
            raise ValueError(
                f"{v!r} is a GitHub web page, not the file. Use "
                "https://raw.githubusercontent.com/<owner>/<repo>/<commit-sha>/<path>"
            )
        ref = _github_ref(host, parts)
        if ref is not None and not _COMMIT.fullmatch(ref.lower()):
            raise ValueError(
                f"{v!r} points at {ref!r}, which can move. Pin it to a full 40-character commit SHA"
            )
        return v

    @field_validator("retrieved")
    @classmethod
    def _check_retrieved(cls, v: date) -> date:
        if not date(2000, 1, 1) <= v <= date.today() + timedelta(days=1):
            raise ValueError(f"{v} can't be right; use the day you took the checksum")
        return v


def _github_ref(host: str, parts: list[str]) -> str | None:
    """The branch or commit a GitHub download URL points at, or None if it isn't one."""
    if host == "raw.githubusercontent.com" and len(parts) >= 4:
        return parts[2]
    if host == "codeload.github.com" and len(parts) >= 4:
        return parts[3]
    if host in _GITHUB and len(parts) >= 4 and parts[2] in ("raw", "archive"):
        return re.sub(r"\.(zip|tar\.gz)$", "", parts[3])
    return None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[RawFile]

    @field_validator("files", mode="before")
    @classmethod
    def _empty_is_none(cls, v: object) -> object:
        return [] if v in (None, "") else v

    @model_validator(mode="after")
    def _check_collisions(self) -> Manifest:
        seen = set()
        for f in self.files:
            if f.path in seen:
                raise ValueError(f"{f.path!r} is listed twice")
            seen.add(f.path)
        for p in seen:
            for parent in PurePosixPath(p).parents:
                if str(parent) in seen:
                    raise ValueError(f"{str(parent)!r} is a file but {p!r} uses it as a folder")
        return self


class _UniqueKeyLoader(yaml.BaseLoader):
    """Every value stays a string (no YAML 1.1 surprises like 012 -> 10), and duplicate keys
    are an error instead of PyYAML's default of silently keeping the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise yaml.constructor.ConstructorError(
                    None, None, "keys must be plain names", key_node.start_mark
                )
            if key in keys:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _UniqueKeyLoader.construct_mapping
)


def load_manifest(path: Path) -> Manifest:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ManifestError(f"{path}: manifest not found") from None
    except (OSError, UnicodeDecodeError) as e:
        raise ManifestError(f"{path}: can't read manifest, save it as UTF-8 ({e})") from None
    if conflict := _CONFLICT.search(text):
        line = text.count("\n", 0, conflict.start()) + 1
        raise ManifestError(f"{path}: unresolved git merge conflict at line {line}")
    try:
        doc = yaml.load(text, Loader=_UniqueKeyLoader)  # safe: BaseLoader builds no objects
    except (yaml.YAMLError, RecursionError) as e:
        hint = " (indent with spaces, not tabs)" if "\t" in text else ""
        msg = str(e).replace("<unicode string>", str(path))
        raise ManifestError(f"{path}: not valid YAML{hint}\n{msg}") from None
    if not isinstance(doc, dict):
        raise ManifestError(f"{path}: expected a mapping with a 'files:' list at the top level")
    try:
        return Manifest.model_validate(doc)
    except ValidationError as e:
        lines = [f"  {_describe_error(doc, err)}" for err in e.errors()]
        raise ManifestError(f"{path}: invalid manifest\n" + "\n".join(lines)) from None


def _describe_error(doc: dict, err: dict) -> str:
    loc = err["loc"]
    where = ".".join(str(x) for x in loc) or "files"
    if len(loc) >= 2 and loc[0] == "files" and isinstance(loc[1], int):
        files = doc.get("files")
        entry = files[loc[1]] if isinstance(files, list) and loc[1] < len(files) else None
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            where += f" ({entry['path']})"
    msg = err["msg"].removeprefix("Value error, ")
    if err["type"] == "extra_forbidden":
        msg += (
            " (the only top-level key is 'files')"
            if len(loc) == 1
            else " (allowed: path, url, sha256, retrieved)"
        )
    return f"{where}: {msg}"


# --- downloading -----------------------------------------------------------------------------


class DownloadError(Exception):
    def __init__(
        self,
        msg: str,
        *,
        unreachable: bool = False,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(msg)
        self.unreachable = unreachable  # never reached the server
        self.retryable = retryable  # the server answered, then something transient broke
        self.retry_after = retry_after  # seconds, if the server said (429/503)


def download(url: str, dest: Path, sha256: str, *, timeout: float = TIMEOUT_S) -> None:
    """Download url to dest. dest only appears, atomically, once its checksum matches."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, ATTEMPTS + 1):
        try:
            _download_once(url, dest, sha256, timeout)
            return
        except DownloadError as e:
            if not e.retryable or attempt == ATTEMPTS:
                raise
            time.sleep(min(e.retry_after or 2**attempt, MAX_WAIT_S))


def _download_once(url: str, dest: Path, sha256: str, timeout: float) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(_quote_url(url), headers={"User-Agent": USER_AGENT})
        h = hashlib.sha256()
        n = 0
        with os.fdopen(fd, "wb") as out, _open(req, timeout) as resp:
            os.fchmod(out.fileno(), 0o666 & ~_umask())  # mkstemp makes it 0600
            length = resp.headers.get("Content-Length")
            html = "text/html" in resp.headers.get("Content-Type", "")
            while True:
                try:
                    chunk = resp.read1(CHUNK)
                except (OSError, http.client.HTTPException) as e:
                    msg = f"connection dropped after {n:,} bytes ({e})"
                    raise DownloadError(msg, retryable=True) from None
                if not chunk:
                    break
                try:
                    out.write(chunk)
                    out.flush()
                except OSError as e:  # disk full, read-only: retrying won't help
                    raise DownloadError(
                        f"can't write to {dest.parent}: {e.strerror or e}"
                    ) from None
                h.update(chunk)
                n += len(chunk)
        if length is not None and length.isdigit() and n != int(length):
            raise DownloadError(
                f"connection dropped: got {n:,} of {int(length):,} bytes", retryable=True
            )
        got = h.hexdigest()
        if got != sha256:
            raise DownloadError(
                f"downloaded file doesn't match the manifest\n"
                f"expected {sha256}\ngot      {got} ({n:,} bytes)\n{_mismatch_hint(url, html)}"
            )
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _mismatch_hint(url: str, html: bool) -> str:
    u = urllib.parse.urlsplit(url)
    ref = _github_ref((u.hostname or "").rstrip("."), [p for p in u.path.split("/") if p])
    if ref is not None:
        return "A pinned commit can't change, so the sha256 or url in the manifest is wrong."
    page = " The server sent a web page." if html and not u.path.endswith((".html", ".htm")) else ""
    return (
        f"{page} On hotel or campus wifi this is usually a login page: log in through a browser"
        " or try another network. Otherwise upstream changed the file; open an issue."
    ).lstrip()


def _quote_url(url: str) -> str:
    """Percent-encode spaces and non-ASCII (DCS file names have both: "KumSaṃ, 1.conllu"),
    leaving existing %XX escapes alone. NFC first, so a name pasted from macOS Finder
    (decomposed) requests the same file as one typed on Linux."""
    u = urllib.parse.urlsplit(unicodedata.normalize("NFC", url))
    safe = "/%:@!$&'()*+,;=-._~"
    return urllib.parse.urlunsplit(
        u._replace(
            path=urllib.parse.quote(u.path, safe=safe),
            query=urllib.parse.quote(u.query, safe=safe + "?"),
        )
    )


def _open(req: urllib.request.Request, timeout: float) -> http.client.HTTPResponse:
    host = req.host
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        msg = f"HTTP {e.code} {str(e.reason).splitlines()[0]}"
        retry_after = _seconds_header(e.headers.get("Retry-After"))
        e.close()
        if e.code >= 500 or e.code == 429:
            raise DownloadError(msg, retryable=True, retry_after=retry_after) from None
        raise DownloadError(msg) from None
    except urllib.error.URLError as e:
        # urllib wraps failures to connect or send: DNS, refused, connect timeout, TLS.
        raise DownloadError(_cant_connect(e.reason, host, timeout), unreachable=True) from None
    except (OSError, http.client.HTTPException) as e:
        # Raised unwrapped from getresponse(), so the request got there: the host is up.
        raise DownloadError(_no_response(e, host, timeout), retryable=True) from None
    except ValueError as e:  # http.client.InvalidURL, a host that won't encode
        raise DownloadError(f"bad URL ({e})") from None


def _seconds_header(value: str | None) -> float | None:
    return float(value) if value and value.strip().isdigit() else None


def _cant_connect(reason: object, host: str, timeout: float) -> str:
    if isinstance(reason, socket.gaierror):
        return f"can't resolve {host}: check the URL, or you may be offline"
    if isinstance(reason, ssl.SSLCertVerificationError):
        return (
            f"the TLS certificate for {host} isn't trusted ({reason.verify_message}). On hotel or"
            " campus wifi this is usually a login page: log in through a browser or try another"
            " network"
        )
    if isinstance(reason, TimeoutError):
        return f"timed out connecting to {host} after {timeout:g}s, are you offline?"
    if isinstance(reason, ConnectionRefusedError):
        return f"{host} refused the connection"
    return f"can't connect to {host} ({reason})"


def _no_response(e: Exception, host: str, timeout: float) -> str:
    if isinstance(e, TimeoutError):
        return f"connected to {host} but got no response within {timeout:g}s"
    if isinstance(e, http.client.RemoteDisconnected):
        return f"{host} closed the connection without responding"
    return f"{host} sent a bad response ({e})"


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


# --- make data -------------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@dataclass
class Result:
    path: str
    status: str  # ok | downloaded | missing | mismatch | error | skipped
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status not in ("ok", "downloaded")


def sync(
    manifest: Manifest,
    raw_dir: Path,
    *,
    offline: bool = False,
    timeout: float = TIMEOUT_S,
    log: Callable[[str], object] = print,
) -> list[Result]:
    """Download missing files (unless offline), then verify every file. Never overwrites a
    file that is already there: a mismatch is reported, not repaired."""
    results = []
    # scheme://host:port -> why its remaining files are skipped. One dead or overloaded host
    # costs one timeout (or one round of retries), not one per file.
    given_up: dict[str, str] = {}
    for f in manifest.files:
        r = _sync_one(f, raw_dir / f.path, offline=offline, timeout=timeout, given_up=given_up)
        results.append(r)
        log(_format(r))
    return results


def _sync_one(
    f: RawFile, dest: Path, *, offline: bool, timeout: float, given_up: dict[str, str]
) -> Result:
    try:
        if dest.is_symlink() and not dest.exists():
            return Result(f.path, "error", f"broken symlink to {os.readlink(dest)}")
        if dest.exists():
            return _verify(f, dest)
        if offline:
            return Result(f.path, "missing", "not downloaded (offline)")
        u = urllib.parse.urlsplit(f.url)
        origin = f"{u.scheme}://{u.netloc.lower()}"
        if origin in given_up:
            return Result(f.path, "skipped", given_up[origin])
        try:
            download(f.url, dest, f.sha256, timeout=timeout)
        except DownloadError as e:
            first = str(e).splitlines()[0]
            if e.unreachable:
                given_up[origin] = f"{u.netloc} was unreachable earlier in this run ({first})"
            elif e.retryable:
                given_up[origin] = f"{u.netloc} kept failing earlier in this run ({first})"
            return Result(f.path, "error", f"{f.url}\n{e}")
        return Result(f.path, "downloaded")
    except OSError as e:
        return Result(f.path, "error", str(e))
    except Exception as e:  # one bad entry must not stop the rest being verified
        return Result(f.path, "error", f"unexpected {type(e).__name__}: {e}")


def _verify(f: RawFile, dest: Path) -> Result:
    if not stat.S_ISREG(dest.stat().st_mode):
        return Result(f.path, "error", "exists but is not a regular file")
    got = sha256_of(dest)
    if got == f.sha256:
        return Result(f.path, "ok")
    return Result(
        f.path,
        "mismatch",
        f"expected {f.sha256}\ngot      {got}\n"
        "This file changed, or its manifest entry did (after a pull or branch switch, see\n"
        "`git log -p data/manifest.yaml`). Move it out of data/raw/ rather than deleting it,\n"
        "since it may be your only copy, then re-run `make data` online.",
    )


def _format(r: Result) -> str:
    label = r.status.upper() if r.failed else r.status
    return f"  {label:<11} {r.path}" + "".join(f"\n      {d}" for d in r.detail.splitlines())


def untracked(manifest: Manifest, raw_dir: Path) -> list[str]:
    """A warning for each file in raw_dir that the manifest doesn't list."""
    listed = {f.path for f in manifest.files}
    folded = {f.path.casefold(): f.path for f in manifest.files}
    warnings = []
    for p in _walk(raw_dir):
        if p.name in _JUNK or p.name.startswith("._") or _is_part(p.name):
            continue
        rel = unicodedata.normalize("NFC", p.relative_to(raw_dir).as_posix())
        if rel in listed:
            continue
        if rel.casefold() in folded:
            warnings.append(f"{rel} is spelled {folded[rel.casefold()]} in the manifest; rename it")
        else:
            warnings.append(
                f"{rel} is not in the manifest: add it, or leave it if it's from another branch"
            )
    return sorted(warnings)


def _walk(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            yield Path(dirpath) / name


def _is_part(name: str) -> bool:
    return name.startswith(".") and name.endswith(".part")


class _Busy(Exception):
    pass


def _lock(raw_dir: Path) -> IO[str] | None:
    """Lock raw_dir for this run (released when the process exits), so cleaning up .part
    files can't delete another run's live download. None if the folder is read-only, in
    which case verifying still works."""
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        fh = (raw_dir / LOCK_NAME).open("a")
    except OSError:
        return None
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise _Busy from None
    return fh


def _remove_partials(raw_dir: Path) -> int:
    """Delete .part files left by a run that was killed (SIGKILL, power loss)."""
    n = 0
    for p in _walk(raw_dir):
        if _is_part(p.name):
            p.unlink(missing_ok=True)
            n += 1
    return n


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(paths.REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def _timeout(s: str) -> float:
    v = float(s)  # argparse turns a ValueError into a usage error
    if not 0 < v <= 600:
        raise argparse.ArgumentTypeError("must be between 0 and 600 seconds")
    return v


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        # Line-buffered so progress shows up live in CI logs and pipes.
        sys.stdout.reconfigure(line_buffering=True, errors="backslashreplace")
    ap = argparse.ArgumentParser(
        prog="python -m sandhi_probe.data.manifest",
        description="Download missing raw files and check every checksum in the manifest.",
    )
    ap.add_argument("--manifest", type=Path, default=paths.MANIFEST_PATH)
    ap.add_argument("--raw-dir", type=Path, default=paths.RAW_DIR)
    ap.add_argument("--offline", action="store_true", help="verify only, never download")
    ap.add_argument("--timeout", type=_timeout, default=TIMEOUT_S, help="seconds per socket read")
    args = ap.parse_args(argv)
    raw = args.raw_dir.expanduser()

    try:
        manifest = load_manifest(args.manifest.expanduser())
    except ManifestError as e:
        print(f"make data: {e}", file=sys.stderr)
        return 2
    if raw.is_symlink() and not raw.exists():
        target = os.readlink(raw)
        print(f"make data: {_rel(raw)} links to {target}, which doesn't exist (drive not mounted?)")
        return 1
    if raw.exists() and not raw.is_dir():
        print(f"make data: {_rel(raw)} exists but is not a folder")
        return 1
    try:
        lock = _lock(raw)  # held until the process exits
    except _Busy:
        print("make data: another `make data` is running on this folder; wait for it to finish")
        return 1

    n = len(manifest.files)
    mode = " (offline: verify only)" if args.offline else ""
    print(f"make data: {n} file(s) in {_rel(args.manifest)} -> {_rel(raw)}/{mode}")
    if lock is not None and (stale := _remove_partials(raw)):
        print(f"  removed {stale} leftover .part file(s) from an interrupted run")
    if not n:
        print("  manifest lists no files yet (corpora are added in #21)")
    try:
        results = sync(manifest, raw, offline=args.offline, timeout=args.timeout)
    except KeyboardInterrupt:
        print("\nmake data: interrupted. Partial downloads were removed; re-run to continue.")
        return 130

    warnings = untracked(manifest, raw)
    for w in warnings[:MAX_WARNINGS]:
        print(f"  warning     {w}")
    if len(warnings) > MAX_WARNINGS:
        print(f"  warning     ...and {len(warnings) - MAX_WARNINGS} more unlisted files")

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in counts.items()) or "nothing to check"
    failed = [r for r in results if r.failed]
    if not failed:
        print(f"make data: ok ({summary})")
        return 0
    print(f"make data: FAILED ({summary})")
    if any(r.status == "missing" for r in failed):
        print("  Run `make data` online (without OFFLINE=1) to download the missing files.")
    elif any(r.status in ("error", "skipped") for r in failed):
        print(
            "  Files already here were still verified and finished downloads are kept;"
            " re-run once the problem above is fixed."
        )
    return 1


if __name__ == "__main__":
    # A closed terminal or a cancelled CI job unwinds like Ctrl-C, so .part files get removed.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, _frame: sys.exit(128 + signum))
    sys.exit(main())
