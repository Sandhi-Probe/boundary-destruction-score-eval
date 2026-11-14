"""Manifest loading and `make data` download/verify behaviour. No real network: downloads
go to a local HTTP server, and an unreachable host is a port nothing listens on."""

from __future__ import annotations

import errno
import hashlib
import io
import os
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from http.client import BadStatusLine, RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple

import pytest

from sandhi_probe.data import manifest as m

Capsys = pytest.CaptureFixture[str]
Patch = pytest.MonkeyPatch

COMMIT = "8aeed5a1343d0e48b64eb32af8c00e8c6eb29359"
BODY = b"agnim ile purohitam\n"
SHA = hashlib.sha256(BODY).hexdigest()
LOCK = ".make-data.lock"


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def entry(path: str = "dcs/a.txt", url: str = "https://example.org/a.txt", sha: str = SHA) -> str:
    return f"  - path: {path}\n    url: {url}\n    sha256: {sha}\n    retrieved: 2026-10-31\n"


def load(tmp_path: Path, *entries: str) -> m.Manifest:
    return m.load_manifest(write(tmp_path, "files:\n" + "".join(entries)))


def rejection(tmp_path: Path, text: str) -> str:
    """The message load_manifest refuses text with."""
    with pytest.raises(m.ManifestError) as e:
        m.load_manifest(write(tmp_path, text))
    return str(e.value)


# --- loading -------------------------------------------------------------------------------


def test_loads_valid_manifest(tmp_path: Path) -> None:
    (f,) = load(tmp_path, entry()).files
    assert (f.path, f.sha256, f.retrieved.isoformat()) == ("dcs/a.txt", SHA, "2026-10-31")


@pytest.mark.parametrize("text", ["files: []\n", "files:\n"])
def test_empty_file_list_is_allowed(tmp_path: Path, text: str) -> None:
    assert m.load_manifest(write(tmp_path, text)).files == []


def test_repo_manifest_is_valid() -> None:
    m.load_manifest(m.paths.MANIFEST_PATH)


@pytest.mark.parametrize(
    ("written", "stored"),
    [
        ("1" * 64, "1" * 64),  # yaml.safe_load would turn this into an int
        (SHA.upper(), SHA),
        (f'" {SHA} "', SHA),
    ],
    ids=["all-digits", "uppercase", "spaces"],
)
def test_checksum_is_normalised(tmp_path: Path, written: str, stored: str) -> None:
    (f,) = load(tmp_path, entry(sha=written)).files
    assert f.sha256 == stored


NOT_A_MAPPING = "expected a mapping with a 'files:' list at the top level"
ENTRY = entry()
TOO_OLD = ENTRY.replace("2026-10-31", "1999-12-31")
TOO_NEW = ENTRY.replace("2026-10-31", str(date.today() + timedelta(days=2)))


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("", NOT_A_MAPPING),
        ("- just a list\n", NOT_A_MAPPING),
        ("files: [\n", "not valid YAML\n"),
        ("files:\n\t- path: a\n", "not valid YAML (indent with spaces, not tabs)\n"),
        ("files: " + "[" * 500 + "\n", "not valid YAML"),  # RecursionError inside PyYAML
        (
            "files:\n" + ENTRY.replace("    url:", "    sha256: 00\n    url:"),
            "duplicate key 'sha256'",
        ),
        ("files:\n  - {[a]: b}\n", "keys must be plain names"),
        ("files:\n" + ENTRY + "    licence: CC-BY\n", "(allowed: path, url, sha256, retrieved)"),
        ("fils: []\n", "(the only top-level key is 'files')"),
        ("files: nope\n", "files: Input should be a valid list"),
        ("files:\n  - nope\n", "files.0: Input should be"),
        ("files:\n  - path: [x]\n", "files.0.path: Input should be"),
        (
            "files:\n"
            + ENTRY.replace("    retrieved: 2026-10-31\n", "").replace(
                "    url: https://example.org/a.txt\n", ""
            ),
            "files.0.url (dcs/a.txt): Field required\n",
        ),
        (
            "files:\n" + entry(sha="nope"),
            "files.0.sha256 (dcs/a.txt): 'nope' is not a SHA-256 hex digest (64 characters,",
        ),
        ("{}\n", "files: Field required"),
        (
            "files:\n" + ENTRY.replace("  - path: dcs/a.txt\n    url:", "  - url:"),
            "files.0.path: Field",
        ),
        (
            "files:\n" + ENTRY.replace(f"    sha256: {SHA}\n", ""),
            "files.0.sha256 (dcs/a.txt): Field",
        ),
        ("files:\n" + entry(sha="abc"), "'abc' is not a SHA-256 hex digest"),  # too short
        ("files:\n" + entry(sha="a" * 63), "is not a SHA-256 hex digest"),  # one short
        ("files:\n" + entry(sha="a" * 65), "is not a SHA-256 hex digest"),  # too long
        ("files:\n" + entry(sha="z" * 64), "is not a SHA-256 hex digest"),  # right length, not hex
        ("files:\n" + ENTRY.replace("2026-10-31", "31/10/2026"), "files.0.retrieved (dcs/a.txt): "),
        (
            "files:\n" + ENTRY.replace("2026-10-31", "2099-01-01"),
            "2099-01-01 can't be right; use the day",
        ),
        ("files:\n" + TOO_OLD, "can't be right"),
        ("files:\n" + TOO_NEW, "can't be right"),
        ("files:\n" + ENTRY + ENTRY, "files: 'dcs/a.txt' is listed twice"),
        ("files:\n" + entry("dcs") + ENTRY, "'dcs' is a file but 'dcs/a.txt' uses it as a folder"),
        (
            "files:\n<<<<<<< HEAD\n" + ENTRY + "=======\n>>>>>>> main\n",
            "git merge conflict at line 2",
        ),
        ("files: []\n=======\n", "conflict at line 2"),
    ],
)
def test_rejects_bad_manifest(tmp_path: Path, text: str, needle: str) -> None:
    assert needle in rejection(tmp_path, text)


def test_unreadable_manifest(tmp_path: Path) -> None:
    with pytest.raises(m.ManifestError, match="manifest not found"):
        m.load_manifest(tmp_path / "nope.yaml")
    with pytest.raises(m.ManifestError, match="can't read manifest"):
        m.load_manifest(tmp_path)  # a folder
    latin1 = tmp_path / "latin1.yaml"
    latin1.write_bytes(b"files: \xff\n")
    with pytest.raises(m.ManifestError, match="save it as UTF-8"):
        m.load_manifest(latin1)


@pytest.mark.parametrize(
    "path",
    ["../etc/passwd", "/abs/a.txt", "dcs//a.txt", "dcs/./a.txt", "dcs\\a.txt", ".hidden/a",
     "dcs/Kumārasaṃbhava.conllu", "dcs/a b.txt", "dcs/a.txt.part", "dcs/", "DCS/a.txt",
     "dcs/a.", "dcs/" + "a" * 101],
)  # fmt: skip
def test_rejects_unsafe_paths(tmp_path: Path, path: str) -> None:
    msg = rejection(tmp_path, "files:\n" + entry(path=f"'{path}'"))  # single quotes: no \ escapes
    assert "files.0.path" in msg
    assert f"{path!r}: use a relative path under data/raw/ made of lowercase a-z, 0-9, " in msg


WEB_PAGE = (
    "is a GitHub web page, not the file. Use "
    "https://raw.githubusercontent.com/<owner>/<repo>/<commit-sha>/<path>"
)
PIN_IT = "which can move. Pin it to a full 40-character commit SHA"
CAN_MOVE = f"points at 'master', {PIN_IT}"
NOT_HTTP = "must be an http(s) URL with a valid host"
ONE_LINE = "keep the URL on one line, in double quotes, no spaces around"
CREDENTIALS = "no credentials in URLs, the manifest is committed to git"


@pytest.mark.parametrize(
    ("url", "needle"),
    [
        ("https://github.com/OliverHellwig/sanskrit/tree/master/dcs/data", WEB_PAGE),
        ("https://github.com/OliverHellwig/sanskrit/blob/master/dcs/readme.md", WEB_PAGE),
        ("https://www.github.com/o/r/blob/master/a.md", WEB_PAGE),
        ("https://github.com./o/r/blob/master/a.md", WEB_PAGE),
        ("https://github.com/o/r/blob", WEB_PAGE),
        ("https://raw.githubusercontent.com/OliverHellwig/sanskrit/master/dcs/a.md", CAN_MOVE),
        ("https://RAW.GITHUBUSERCONTENT.COM/OliverHellwig/sanskrit/master/dcs/a.md", CAN_MOVE),
        ("https://raw.githubusercontent.com/o/r/master/a", CAN_MOVE),
        ("https://raw.githubusercontent.com/o/r/8aeed5a/a.md", PIN_IT),  # short SHA
        (f"https://raw.githubusercontent.com/o/r/{COMMIT}0/a.md", PIN_IT),  # one too long
        ("https://github.com/OliverHellwig/sanskrit/raw/master/dcs/a.md", CAN_MOVE),
        ("https://github.com/OliverHellwig/sanskrit/archive/master.zip", CAN_MOVE),
        ("https://codeload.github.com/OliverHellwig/sanskrit/zip/master", CAN_MOVE),
        ("ftp://example.org/a.txt", NOT_HTTP),
        ("example.org/a.txt", NOT_HTTP),
        ('"http://[::1"', NOT_HTTP),
        ("http://example.org:99999/a", NOT_HTTP),
        ("http://a..b/x", NOT_HTTP),
        ("http:///a", NOT_HTTP),
        ("http://./a", NOT_HTTP),
        ("https://user:hunter2@example.org/a", CREDENTIALS),
        ("https://user@example.org/a", CREDENTIALS),
        ("https://:hunter2@example.org/a", CREDENTIALS),
        ('" https://example.org/a"', ONE_LINE),
        ('"https://example.org/a\\n"', ONE_LINE),
        ('"https://example.org/a\\nb"', ONE_LINE),
        ('"https://example.org/a\\x7fb"', ONE_LINE),
    ],
)
def test_rejects_bad_urls(tmp_path: Path, url: str, needle: str) -> None:
    assert needle in rejection(tmp_path, "files:\n" + entry(url=url))


@pytest.mark.parametrize(
    "url",
    [
        f"https://raw.githubusercontent.com/OliverHellwig/sanskrit/{COMMIT}/dcs/data/readme.md",
        f"https://raw.githubusercontent.com/o/r/{COMMIT.upper()}/a.md",
        f"https://github.com/o/r/raw/{COMMIT}/a.md",
        f"https://github.com/o/r/archive/{COMMIT}.zip",
        f"https://github.com/o/r/archive/{COMMIT}.tar.gz",
        f"https://codeload.github.com/o/r/zip/{COMMIT}",
        "https://github.com/o/r/releases/download/v1/a.zip",  # not checked; sha256 still is
        "https://github.com/o/r",  # too short to name a ref
        "https://raw.githubusercontent.com/o/r/a",
        "https://codeload.github.com/o/r/zip",
        "https://example.org/o/r/blob/master/a",  # only GitHub's URL shapes are special
        "https://example.org/o/r/archive/master.zip",
    ],
)
def test_accepts_good_urls(tmp_path: Path, url: str) -> None:
    load(tmp_path, entry(url=url))


@pytest.mark.parametrize("day", ["2000-01-01", str(date.today() + timedelta(days=1))])
def test_retrieved_date_range_is_inclusive(tmp_path: Path, day: str) -> None:
    load(tmp_path, ENTRY.replace("2026-10-31", day))


# --- local HTTP server -----------------------------------------------------------------------


class Reply(NamedTuple):
    status: int | None  # None sends body as is, with no status line or headers
    body: bytes = b""
    headers: dict[str, str] = {}  # an empty value leaves that header out
    delay_s: float = 0  # before answering
    stall_s: float = 0  # after the body, with the connection still open


OK = Reply(200, BODY)


def wait(seconds: float) -> None:
    threading.Event().wait(seconds)  # not time.sleep: the sleeps fixture patches that


class Handler(BaseHTTPRequestHandler):
    server: Server

    def do_GET(self) -> None:
        srv = self.server
        srv.hits[self.path] += 1
        srv.got_request.set()
        replies = srv.routes.get(self.path, [Reply(404, b"not found")])
        reply = replies[min(srv.hits[self.path], len(replies)) - 1]
        wait(reply.delay_s)
        if reply.status is not None:
            self.send_response(reply.status)
            for k, v in {"Content-Length": str(len(reply.body)), **reply.headers}.items():
                if v:
                    self.send_header(k, v)
            self.end_headers()
        self.wfile.write(reply.body)
        wait(reply.stall_s)

    def log_message(self, *args: object) -> None:
        pass


class Server(ThreadingHTTPServer):
    """routes: path -> replies, served in order (the last one repeats). Other paths are a 404."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.routes: dict[str, list[Reply]] = {}
        self.hits: Counter[str] = Counter()
        self.got_request = threading.Event()
        self.base = f"http://127.0.0.1:{self.server_address[1]}"

    def handle_error(self, request: object, client_address: object) -> None:
        # A slow reply can outlive a client that timed out. Anything else is a bug here.
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


@pytest.fixture
def server() -> Iterator[Server]:
    with Server() as s:
        # A short poll, or shutdown() waits out the default half second in every test.
        threading.Thread(target=s.serve_forever, args=(0.01,), daemon=True).start()
        yield s
        s.shutdown()


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: Patch) -> list[float]:
    """Backoff sleeps are recorded instead of slept."""
    calls: list[float] = []
    monkeypatch.setattr(m.time, "sleep", calls.append)
    return calls


def closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def project(tmp_path: Path, *entries: str) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    return write(tmp_path, "files:\n" + "".join(entries)), raw


def serve(tmp_path: Path, server: Server, **replies: list[Reply]) -> tuple[Path, Path]:
    """A project with one file per keyword: a=[...] is dcs/a.txt, answered in that order."""
    server.routes.update({f"/{name}.txt": r for name, r in replies.items()})
    return project(tmp_path, *(entry(f"dcs/{n}.txt", f"{server.base}/{n}.txt") for n in replies))


def put(raw: Path, rel: str = "dcs/a.txt", body: bytes = BODY) -> None:
    (raw / rel).parent.mkdir(parents=True, exist_ok=True)
    (raw / rel).write_bytes(body)


def run(man: Path, raw: Path, *extra: str) -> int:
    return m.main(["--manifest", str(man), "--raw-dir", str(raw), "--timeout", "2", *extra])


# --- the whole report --------------------------------------------------------------------------


def test_exact_report_of_a_mixed_run(tmp_path: Path, server: Server, capsys: Capsys) -> None:
    server.routes["/new.txt"] = [OK]
    man, raw = project(
        tmp_path,
        entry("dcs/have.txt"),
        entry("dcs/new.txt", server.base + "/new.txt"),
        entry("dcs/changed.txt"),
        entry("dcs/gone.txt", server.base + "/gone.txt"),
    )
    put(raw, "dcs/have.txt")
    put(raw, "dcs/changed.txt", b"edited\n")
    put(raw, "stray.txt")
    put(raw, ".old.txt.x1.part")
    assert run(man, raw) == 1
    got = hashlib.sha256(b"edited\n").hexdigest()
    expected = f"""\
make data: 4 file(s) in {man} -> {raw}/
  removed 1 leftover .part file(s) from an interrupted run
  ok          dcs/have.txt
  downloaded  dcs/new.txt
  MISMATCH    dcs/changed.txt
      expected {SHA}
      got      {got}
      This file changed, or its manifest entry did (after a pull or branch switch, see
      `git log -p data/manifest.yaml`). Online, `make data fix` moves it to
      data/raw/.replaced/ (never deletes it: it may be your only copy) and downloads a
      fresh one.
  ERROR       dcs/gone.txt
      {server.base}/gone.txt
      HTTP 404 Not Found
  warning     stray.txt is not in the manifest: add it, or leave it if it's from another branch
make data: FAILED (1 ok, 1 downloaded, 1 mismatch, 1 error)
  Files already here were still verified and finished downloads are kept; re-run once \
the problem above is fixed.
"""
    assert capsys.readouterr().out == expected


def test_exact_report_of_an_empty_manifest(tmp_path: Path, capsys: Capsys) -> None:
    man, raw = project(tmp_path)
    assert run(man, raw, "--timeout", "600") == 0  # the longest timeout allowed
    assert capsys.readouterr().out == (
        f"make data: 0 file(s) in {man} -> {raw}/\n"
        "  manifest lists no files yet (corpora are added in #21)\n"
        "make data: ok (nothing to check)\n"
    )


# --- sync ------------------------------------------------------------------------------------


@pytest.fixture(params=[0o002, 0o022])  # two values, so a hard-coded mode can't pass
def umask(request: pytest.FixtureRequest) -> Iterator[int]:
    old = os.umask(request.param)
    yield request.param
    assert os.umask(old) == request.param  # reading the umask has to put it back


def test_downloads_missing_file_then_verifies_without_network(
    tmp_path: Path, server: Server, capsys: Capsys, umask: int
) -> None:
    man, raw = serve(tmp_path, server, a=[OK])
    assert run(man, raw) == 0
    assert (raw / "dcs/a.txt").read_bytes() == BODY
    assert (raw / "dcs/a.txt").stat().st_mode & 0o777 == 0o666 & ~umask  # not mkstemp's 0600
    assert run(man, raw) == 0
    assert server.hits["/a.txt"] == 1  # second run only hashed the local file
    assert capsys.readouterr().out.endswith("make data: ok (1 ok)\n")


def test_existing_mismatch_fails_and_is_not_overwritten(
    tmp_path: Path, server: Server, capsys: Capsys
) -> None:
    man, raw = serve(tmp_path, server, a=[OK])
    put(raw, body=b"edited by hand\n")
    assert run(man, raw) == 1
    assert (raw / "dcs/a.txt").read_bytes() == b"edited by hand\n"
    assert not server.hits
    assert not (raw / ".replaced").exists()  # only FIX=1 moves anything
    assert "re-run once" not in capsys.readouterr().out  # it's not a download problem


def test_symlinks_to_real_things_are_followed(tmp_path: Path) -> None:
    man, real_raw = project(tmp_path, entry())
    put(tmp_path, "drive/a.txt")
    (real_raw / "dcs").mkdir()
    (real_raw / "dcs/a.txt").symlink_to(tmp_path / "drive/a.txt")
    (tmp_path / "link").symlink_to(real_raw)
    assert run(man, tmp_path / "link") == 0


def test_downloaded_mismatch_leaves_nothing_behind(
    tmp_path: Path, server: Server, capsys: Capsys
) -> None:
    page = Reply(200, b"<html>log in</html>", {"Content-Type": "text/html; charset=utf-8"})
    man, raw = serve(tmp_path, server, a=[page])
    assert run(man, raw) == 1
    assert sorted(p.name for p in raw.rglob("*")) == [LOCK, "dcs"]  # no file, no .part
    got = hashlib.sha256(page.body).hexdigest()
    assert (
        "      downloaded file doesn't match the manifest\n"
        f"      expected {SHA}\n"
        f"      got      {got} (19 bytes)\n"
        "      The server sent a web page. On hotel or campus wifi"
    ) in capsys.readouterr().out


WIFI_HINT = (
    "On hotel or campus wifi this is usually a login page: log in through a browser"
    " or try another network. Otherwise upstream changed the file; open an issue."
)
PINNED = "A pinned commit can't change, so the sha256 or url in the manifest is wrong."


@pytest.mark.parametrize(
    ("url", "html", "hint"),
    [
        ("http://h/a.txt", False, WIFI_HINT),
        ("http://h/a.txt", True, "The server sent a web page. " + WIFI_HINT),
        ("http://h/a.html", True, WIFI_HINT),
        ("http://h/a.htm", True, WIFI_HINT),
        (f"https://raw.githubusercontent.com/o/r/{COMMIT}/a.md", False, PINNED),
        (f"https://raw.githubusercontent.com./o/r/{COMMIT}/a.md", False, PINNED),
    ],
)
def test_mismatch_hint(url: str, html: bool, hint: str) -> None:
    assert m._mismatch_hint(url, html) == hint


@pytest.mark.parametrize(
    "reply", [Reply(404, b"gone"), Reply(200, b"wrong bytes\n")], ids=["404", "wrong-checksum"]
)
def test_permanent_failure_is_not_retried_and_does_not_skip_the_host(
    tmp_path: Path, server: Server, sleeps: list[float], reply: Reply
) -> None:
    man, raw = serve(tmp_path, server, a=[reply], b=[OK])
    assert run(man, raw) == 1
    assert server.hits == {"/a.txt": 1, "/b.txt": 1}
    assert sleeps == []


@pytest.mark.parametrize(
    ("replies", "waits"),
    [
        ([Reply(503, b"busy"), Reply(500, b"oops"), OK], [2, 4]),
        ([Reply(429, headers={"Retry-After": "7"}), Reply(429, headers={"Retry-After": "99"}), OK],
         [7, 30]),
        ([Reply(503, b"", {"Retry-After": "0"}), OK], [2]),
        ([Reply(503, b"", {"Retry-After": "1.5"}), OK], [2]),
        ([Reply(503, b"", {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), OK], [2]),
        ([Reply(200, BODY[:5], {"Content-Length": "999"}), OK], [2]),  # wifi drops mid-file
        ([Reply(None), OK], [2]),
    ],
    ids=["5xx", "retry-after", "retry-after-0", "retry-after-float", "retry-after-date",
         "truncated", "closed"],
)  # fmt: skip
def test_transient_failure_is_retried(
    tmp_path: Path, server: Server, sleeps: list[float], replies: list[Reply], waits: list[float]
) -> None:
    man, raw = serve(tmp_path, server, a=replies)
    assert run(man, raw) == 0
    assert (raw / "dcs/a.txt").read_bytes() == BODY
    assert server.hits["/a.txt"] == len(replies)
    assert sleeps == waits


@pytest.mark.parametrize(
    ("reply", "why"),
    [
        (Reply(503, b"busy"), "(HTTP 503 Service Unavailable)"),
        (Reply(None), " closed the connection without responding)"),
        (Reply(None, b"garbage\r\n"), " sent a bad response ("),
        (
            Reply(200, BODY[:5], {"Content-Length": "999"}),
            "(connection dropped: got 5 of 999 bytes)",
        ),
        (Reply(200, b"5\r\nagnim", {"Transfer-Encoding": "chunked"}), "dropped after 5 bytes ("),
    ],
    ids=["503", "closed", "garbage", "truncated", "chunked"],
)
def test_gives_up_after_three_attempts_and_skips_the_host(
    tmp_path: Path, server: Server, sleeps: list[float], capsys: Capsys, reply: Reply, why: str
) -> None:
    man, raw = serve(tmp_path, server, a=[reply], b=[OK])
    assert run(man, raw) == 1
    assert server.hits == {"/a.txt": 3}
    assert sleeps == [2, 4]  # no wait after the last attempt
    assert not list(raw.rglob("*.part"))
    out = capsys.readouterr().out
    assert "SKIPPED     dcs/b.txt" in out
    assert " kept failing earlier in this run (" in out
    assert why in out


@pytest.mark.parametrize("length", ["", "abc"], ids=["absent", "not-a-number"])
def test_download_without_a_usable_content_length(
    tmp_path: Path, server: Server, length: str
) -> None:
    man, raw = serve(tmp_path, server, a=[Reply(200, BODY, {"Content-Length": length})])
    assert run(man, raw) == 0


@pytest.mark.parametrize(
    ("reply", "why"),
    [
        (Reply(200, BODY, delay_s=1.0), "but got no response within 0.1s"),
        (
            Reply(200, BODY[:5], {"Content-Length": "999"}, stall_s=1.0),
            "connection dropped after 5 bytes (timed out)",
        ),
    ],
    ids=["before-headers", "mid-body"],
)
def test_stalled_server_is_retried_then_bounded(
    tmp_path: Path, server: Server, capsys: Capsys, reply: Reply, why: str
) -> None:
    man, raw = serve(tmp_path, server, a=[reply])
    start = time.monotonic()
    assert run(man, raw, "--timeout", "0.1") == 1
    assert time.monotonic() - start < 3
    assert server.hits["/a.txt"] == 3
    assert why in capsys.readouterr().out


class FullDisk(io.FileIO):
    def write(self, b: bytes) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")


@pytest.mark.parametrize(
    ("wrap", "report"),
    [
        # A chunk bigger than Python's write buffer fails in write()...
        (lambda f: f, "can't write to"),
        # ...a small one in flush() and again in close(), whose OSError wins.
        (io.BufferedWriter, "ERROR       dcs/a.txt\n      [Errno 28] No space left on device\n"),
    ],
    ids=["big-chunk", "small-chunk"],
)
def test_disk_full_is_reported_and_not_retried(
    tmp_path: Path,
    server: Server,
    monkeypatch: Patch,
    capsys: Capsys,
    wrap: Callable,
    report: str,
) -> None:
    man, raw = serve(tmp_path, server, a=[OK], b=[OK])
    monkeypatch.setattr(m.os, "fdopen", lambda fd, *_: wrap(FullDisk(fd, "wb")))
    assert run(man, raw) == 1
    assert server.hits == {"/a.txt": 1, "/b.txt": 1}  # the host is fine, so b is still tried
    out = capsys.readouterr().out
    assert report in out
    assert "No space left on device" in out
    assert not list(raw.rglob("*.part"))


def test_ctrl_c_removes_the_partial_download_and_exits_130(
    tmp_path: Path, monkeypatch: Patch, capsys: Capsys
) -> None:
    def ctrl_c(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    man, raw = project(tmp_path, entry())
    put(raw, "stray.txt")
    monkeypatch.setattr(m.urllib.request, "urlopen", ctrl_c)
    assert run(man, raw) == 130
    assert not list(raw.rglob("*.part"))
    assert capsys.readouterr().out == (
        f"make data: 1 file(s) in {man} -> {raw}/\n"
        "\nmake data: interrupted. Partial downloads were removed; re-run to continue.\n"
    )


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP], ids=["sigterm", "sighup"])
def test_killed_mid_download_removes_the_part_file(
    tmp_path: Path, server: Server, sig: signal.Signals
) -> None:
    # The only test of `python -m`, which is what `make data` actually runs.
    man, raw = serve(tmp_path, server, a=[Reply(200, BODY, delay_s=10)])
    args = ["-m", "sandhi_probe.data.manifest", "--manifest", str(man), "--raw-dir", str(raw)]
    proc = subprocess.Popen([sys.executable, *args], stdout=subprocess.DEVNULL)
    try:
        assert server.got_request.wait(10)
        assert list((raw / "dcs").glob(".a.txt.*.part"))
        proc.send_signal(sig)
        assert proc.wait(10) == 128 + sig
    finally:
        proc.kill()
    assert not list(raw.rglob("*.part"))
    assert (raw / LOCK).exists()  # left in place: deleting it would race the next run


def test_offline_verifies_but_never_downloads(
    tmp_path: Path, server: Server, capsys: Capsys
) -> None:
    man, raw = serve(tmp_path, server, a=[OK])
    assert run(man, raw, "--offline") == 1
    assert not server.hits
    assert not (raw / "dcs").exists()
    out = capsys.readouterr().out
    assert f"-> {raw}/ (offline: verify only)\n" in out
    assert "MISSING     dcs/a.txt\n      not downloaded (offline)\n" in out
    assert "without OFFLINE=1" in out
    put(raw)
    assert run(man, raw, "--offline") == 0


def test_unreachable_host_fails_fast_and_skips_only_its_own_files(
    tmp_path: Path, server: Server, sleeps: list[float], capsys: Capsys
) -> None:
    server.routes["/e.txt"] = [OK]
    port = closed_port()
    dead = f"127.0.0.1:{port}"
    man, raw = project(
        tmp_path,
        entry("dcs/a.txt", f"http://{dead}/a"),
        entry("dcs/b.txt", f"http://{dead}/b"),
        entry("dcs/c.txt", f"http://{dead}/c"),
        entry("dcs/d.txt", f"https://{dead}/d"),
        entry("dcs/e.txt", server.base + "/e.txt"),
        entry("dcs/f.txt", f"http://LOCALHOST:{port}/f"),
        entry("dcs/g.txt", f"http://localhost:{port}/g"),
    )
    put(raw, "dcs/c.txt")
    assert run(man, raw) == 1
    assert sleeps == []
    out = capsys.readouterr().out
    assert "ERROR       dcs/a.txt" in out
    assert (
        f"SKIPPED     dcs/b.txt\n      {dead} was unreachable earlier in this run"
        f" ({dead} refused the connection)\n"
    ) in out
    assert "ok          dcs/c.txt" in out  # already here: verified, not skipped
    assert "ERROR       dcs/d.txt" in out  # same port, other scheme: not skipped
    assert "downloaded  dcs/e.txt" in out  # same machine, other port: not skipped
    assert "SKIPPED     dcs/g.txt" in out  # host names ignore case
    assert "re-run once the problem above is fixed" in out


def tls_error() -> ssl.SSLCertVerificationError:
    e = ssl.SSLCertVerificationError(1, "certificate verify failed")
    e.verify_message = "self-signed certificate"
    return e


@pytest.mark.parametrize(
    ("describe", "error", "message"),
    [
        (
            m._cant_connect,
            socket.gaierror(8, "x"),
            "can't resolve h: check the URL, or you may be offline",
        ),
        (
            m._cant_connect,
            tls_error(),
            "the TLS certificate for h isn't trusted (self-signed certificate). On hotel or campus"
            " wifi this is usually a login page: log in through a browser or try another network",
        ),
        (m._cant_connect, TimeoutError(), "timed out connecting to h after 30s, are you offline?"),
        (m._cant_connect, ConnectionRefusedError(), "h refused the connection"),
        (m._cant_connect, OSError("no route"), "can't connect to h (no route)"),
        (m._no_response, TimeoutError(), "connected to h but got no response within 30s"),
        (m._no_response, RemoteDisconnected(), "h closed the connection without responding"),
        (m._no_response, BadStatusLine("x"), "h sent a bad response (x)"),
    ],
    ids=["dns", "tls", "connect-timeout", "refused", "other", "read-timeout", "closed", "garbage"],
)
def test_network_failure_messages(describe: Callable, error: Exception, message: str) -> None:
    assert describe(error, "h", 30) == message


def test_url_python_cannot_send_is_a_clear_error(tmp_path: Path, capsys: Capsys) -> None:
    # Passes validation, then fails before connecting: "。" won't encode for the Host header.
    man, raw = project(tmp_path, entry(url='"http://h。invalid/a"'))
    assert run(man, raw) == 1
    assert "      bad URL ('latin-1' codec can't encode" in capsys.readouterr().out


def test_non_ascii_url_is_percent_encoded(tmp_path: Path, server: Server) -> None:
    quoted = "/Kum%C4%81rasa%E1%B9%83bhava,%201.conllu"
    server.routes[quoted] = [OK]
    url = f'"{server.base}/Kumārasaṃbhava, 1.conllu"'
    man, raw = project(tmp_path, entry("dcs/sub/a.txt", url))
    assert run(man, raw) == 0
    assert (raw / "dcs/sub/a.txt").read_bytes() == BODY


@pytest.mark.parametrize(
    ("url", "quoted"),
    [
        ("http://h/a b", "http://h/a%20b"),
        ("http://h/a%20b", "http://h/a%20b"),  # already escaped: left alone
        ("http://h/Kumāra", "http://h/Kum%C4%81ra"),  # decomposed, as Finder pastes it
        ("http://h/a?q=b c&d=e?f", "http://h/a?q=b%20c&d=e?f"),
        ("http://h/a:@!$&'()*+,;=-._~/b", "http://h/a:@!$&'()*+,;=-._~/b"),
    ],
)
def test_quote_url(url: str, quoted: str) -> None:
    assert m._quote_url(url) == quoted


def test_one_crashing_entry_does_not_stop_the_rest(
    tmp_path: Path, monkeypatch: Patch, capsys: Capsys
) -> None:
    man, raw = project(tmp_path, entry("dcs/a.txt"), entry("dcs/b.txt"), entry("dcs/c.txt"))
    put(raw, "dcs/a.txt")
    put(raw, "dcs/b.txt")

    def flaky(p: Path) -> str:
        if p.name == "a.txt":
            raise RuntimeError("boom")
        return SHA

    monkeypatch.setattr(m, "sha256_of", flaky)
    assert run(man, raw, "--offline") == 1
    out = capsys.readouterr().out
    assert "ERROR       dcs/a.txt\n      unexpected RuntimeError: boom\n" in out
    assert "ok          dcs/b.txt" in out
    assert "without OFFLINE=1" in out  # the hint for missing files wins over the error hint
    assert "re-run once" not in out


@pytest.mark.parametrize("flags", [(), ("--offline",)], ids=["online", "offline"])
@pytest.mark.parametrize(
    ("create", "why"),
    [
        (Path.mkdir, "ERROR       dcs/a.txt\n      exists but is not a regular file\n"),
        (os.mkfifo, "ERROR       dcs/a.txt\n      exists but is not a regular file\n"),
        (
            lambda p: p.symlink_to("/nowhere"),
            "ERROR       dcs/a.txt\n      broken symlink to /nowhere\n",
        ),
    ],
    ids=["folder", "fifo", "broken-symlink"],
)
def test_non_regular_destination_is_an_error_not_a_hang(
    tmp_path: Path,
    capsys: Capsys,
    create: Callable,
    why: str,
    flags: tuple[str, ...],
) -> None:
    man, raw = project(tmp_path, entry(url=f"http://127.0.0.1:{closed_port()}/a"))
    (raw / "dcs").mkdir()
    create(raw / "dcs/a.txt")
    assert run(man, raw, *flags) == 1
    assert why in capsys.readouterr().out


@pytest.mark.parametrize(
    ("create", "why"),
    [
        (
            lambda p: p.symlink_to("unmounted-drive"),
            "links to unmounted-drive, which doesn't exist (drive not mounted?)",
        ),
        (Path.touch, "exists but is not a folder"),
    ],
    ids=["broken-symlink", "file"],
)
def test_unusable_raw_folder(tmp_path: Path, capsys: Capsys, create: Callable, why: str) -> None:
    man, raw = write(tmp_path, "files:\n"), tmp_path / "raw"
    create(raw)
    assert run(man, raw) == 1
    assert capsys.readouterr().out == f"make data: {raw} {why}\n"


def test_creates_a_missing_raw_folder(tmp_path: Path) -> None:
    assert run(write(tmp_path, "files:\n"), tmp_path / "data/raw") == 0
    assert (tmp_path / "data/raw" / LOCK).exists()


def test_second_run_on_the_same_folder_is_refused(tmp_path: Path, capsys: Capsys) -> None:
    man, raw = project(tmp_path)
    live = raw / ".a.txt.x1.part"  # the first run's download in progress
    live.touch()
    held = m._lock(raw)
    try:
        assert run(man, raw) == 1
        assert capsys.readouterr().out == (
            "make data: another `make data` is running on this folder; wait for it to finish\n"
        )
        assert live.exists()
    finally:
        held.close()
    assert run(man, raw) == 0
    assert not live.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores folder permissions")
def test_losing_access_to_the_folder_above_raw_is_not_ignored(tmp_path: Path) -> None:
    raw = tmp_path / "parent/raw"
    raw.mkdir(parents=True)
    raw.parent.chmod(0)
    try:
        with pytest.raises(PermissionError):  # os.walk alone would report an empty folder
            m.untracked(m.Manifest(files=[]), raw)
    finally:
        raw.parent.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores folder permissions")
def test_read_only_folder_is_verified_and_left_alone(tmp_path: Path, capsys: Capsys) -> None:
    man, raw = project(tmp_path, entry())
    put(raw)
    put(raw, ".a.txt.x1.part")
    raw.chmod(0o555)
    try:
        assert run(man, raw) == 0
    finally:
        raw.chmod(0o755)
    out = capsys.readouterr().out
    assert "ok          dcs/a.txt" in out
    assert "removed" not in out
    assert "warning" not in out
    assert (raw / ".a.txt.x1.part").exists()


def test_warns_about_unlisted_files_and_removes_stale_partials(
    tmp_path: Path, capsys: Capsys
) -> None:
    man, raw = project(tmp_path)
    for name in (".gitkeep", ".DS_Store", "Thumbs.db", "desktop.ini", "._stray.txt", "stray.txt",
                 "notes.part", ".a.txt.x1.part", "dcs/.b.txt.x2.part", "dcs/deep.txt"):  # fmt: skip
        put(raw, name)
    (raw / "dangling").symlink_to("/nowhere")
    assert run(man, raw) == 0
    out = capsys.readouterr().out
    assert "removed 2 leftover .part file(s)" in out
    assert not list(raw.rglob(".*.part"))
    tail = " is not in the manifest: add it, or leave it if it's from another branch"
    assert [line.removesuffix(tail) for line in out.splitlines() if "warning" in line] == [
        "  warning     dangling",
        "  warning     dcs/deep.txt",
        "  warning     notes.part",
        "  warning     stray.txt",
    ]


@pytest.mark.parametrize("n", [20, 23])
def test_warnings_are_capped_at_20(tmp_path: Path, capsys: Capsys, n: int) -> None:
    man, raw = project(tmp_path)
    for i in range(n):
        put(raw, f"f{i}.txt")
    assert run(man, raw) == 0
    out = capsys.readouterr().out
    assert out.count("is not in the manifest") == 20
    assert ("  warning     ...and 3 more unlisted files\n" in out) == (n == 23)
    assert out.count("unlisted files") == (n == 23)


def test_case_only_difference_says_rename_not_delete(tmp_path: Path) -> None:
    man, raw = project(tmp_path, entry())
    put(raw, "dcs/A.txt")
    (w,) = m.untracked(m.load_manifest(man), raw)
    assert w == "dcs/A.txt is spelled dcs/a.txt in the manifest; rename it"


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "soon", "601"])
def test_rejects_bad_timeout(tmp_path: Path, value: str) -> None:
    man, raw = project(tmp_path)
    with pytest.raises(SystemExit) as e:
        run(man, raw, "--timeout", value)
    assert e.value.code == 2


def test_invalid_manifest_exits_2(tmp_path: Path, capsys: Capsys) -> None:
    man, raw = project(tmp_path, entry(sha="nope"))
    assert run(man, raw) == 2
    captured = capsys.readouterr()
    assert f"make data: {man}: invalid manifest\n  files.0.sha256 (" in captured.err
    assert captured.out == ""


def test_default_arguments(tmp_path: Path, monkeypatch: Patch) -> None:
    """`make data` passes no flags, so these defaults are what people actually run."""
    man, raw = project(tmp_path, entry())
    monkeypatch.setattr(m.paths, "MANIFEST_PATH", man)
    monkeypatch.setattr(m.paths, "RAW_DIR", raw)
    requests = []

    def refuse(req: urllib.request.Request, timeout: float) -> None:
        requests.append((req.full_url, req.get_header("User-agent"), timeout))
        raise urllib.error.URLError(ConnectionRefusedError())

    monkeypatch.setattr(m.urllib.request, "urlopen", refuse)
    assert m.main([]) == 1
    assert requests == [("https://example.org/a.txt", m.USER_AGENT, 30)]


def test_tilde_in_paths_is_expanded(tmp_path: Path, monkeypatch: Patch) -> None:
    project(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert m.main(["--manifest", "~/manifest.yaml", "--raw-dir", "~/raw"]) == 0
    assert (tmp_path / "raw" / LOCK).exists()


def test_output_survives_a_non_unicode_terminal(tmp_path: Path, monkeypatch: Patch) -> None:
    # LANG=C on a lab machine: a non-ASCII folder name must not crash the run.
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stdout)
    assert run(write(tmp_path, "files:\n"), tmp_path / "कुमार") == 0
    assert stdout.line_buffering
    assert b"\\u0915" in stdout.buffer.getvalue()


# --- FIX=1: re-download files that fail their checksum (#46) ----------------------------------

CORRUPT = b"agnim ile purohitam\r\n"  # say, an editor changed the line endings
CORRUPT_SHA = hashlib.sha256(CORRUPT).hexdigest()
CHANGED = b"upstream changed\n"


def snapshot(p: Path) -> tuple[bytes, int]:
    return p.read_bytes(), p.stat().st_mtime_ns


def kept(raw: Path) -> dict[str, bytes]:
    """The old copies FIX=1 kept, by their path under raw/.replaced/<run>/."""
    root = raw / ".replaced"
    return {
        p.relative_to(root).as_posix().split("/", 1)[1]: p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_fix_repairs_a_corrupted_file_and_keeps_the_old_copy(
    tmp_path: Path, server: Server, capsys: Capsys
) -> None:
    man, raw = serve(tmp_path, server, a=[OK], b=[OK])
    put(raw, "dcs/a.txt", CORRUPT)
    put(raw, "dcs/b.txt")
    manifest_before = snapshot(man)
    assert run(man, raw, "--fix") == 0
    assert (raw / "dcs/a.txt").read_bytes() == BODY
    assert server.hits == {"/a.txt": 1}  # b was fine, so it wasn't downloaded
    assert kept(raw) == {"dcs/a.txt": CORRUPT}
    assert snapshot(man) == manifest_before  # the manifest is never changed
    (old,) = (raw / ".replaced").rglob("a.txt")
    assert capsys.readouterr().out == (
        f"make data: 2 file(s) in {man} -> {raw}/"
        " (fix: re-download files that fail their checksum)\n"
        f"  repaired    dcs/a.txt\n      old copy kept in {old}\n"
        "  ok          dcs/b.txt\n"
        "make data: ok (1 repaired, 1 ok)\n"
        f"  The old copies are in {raw / '.replaced'}/. Delete them once you're sure you don't"
        " need them.\n"
    )
    # Out of the way: a normal run neither warns about the old copy nor touches it.
    assert run(man, raw) == 0
    assert capsys.readouterr().out.endswith(
        f"-> {raw}/\n  ok          dcs/a.txt\n  ok          dcs/b.txt\nmake data: ok (2 ok)\n"
    )
    assert kept(raw) == {"dcs/a.txt": CORRUPT}


def test_fix_folder_follows_the_umask(tmp_path: Path, server: Server, umask: int) -> None:
    man, raw = serve(tmp_path, server, a=[OK])
    put(raw, "dcs/a.txt", CORRUPT)
    assert run(man, raw, "--fix") == 0
    (run_dir,) = (raw / ".replaced").iterdir()
    assert run_dir.stat().st_mode & 0o777 == 0o777 & ~umask  # not mkdtemp's 0700


def test_fix_twice_never_overwrites_an_old_copy(tmp_path: Path, server: Server) -> None:
    man, raw = serve(tmp_path, server, a=[OK])
    for body in (b"first\n", b"second\n"):
        put(raw, "dcs/a.txt", body)
        assert run(man, raw, "--fix") == 0
    assert sorted(p.read_bytes() for p in (raw / ".replaced").rglob("a.txt")) == [
        b"first\n",
        b"second\n",
    ]


@pytest.mark.parametrize(
    ("reply", "why"),
    [
        (
            Reply(200, CHANGED),
            "downloaded file doesn't match the manifest\n"
            f"      expected {SHA}\n"
            f"      got      {hashlib.sha256(CHANGED).hexdigest()} (17 bytes)\n",
        ),
        (Reply(404, b"gone"), "HTTP 404 Not Found\n"),
    ],
    ids=["fresh-copy-mismatches-too", "404"],
)
def test_fix_puts_the_original_back_when_the_fresh_download_fails(
    tmp_path: Path, server: Server, capsys: Capsys, reply: Reply, why: str
) -> None:
    man, raw = serve(tmp_path, server, a=[reply])
    put(raw, "dcs/a.txt", CORRUPT)
    manifest_before = snapshot(man)
    assert run(man, raw, "--fix") == 1
    assert (raw / "dcs/a.txt").read_bytes() == CORRUPT
    assert not (raw / ".replaced").exists()  # nothing kept, so no empty folders either
    assert not list(raw.rglob("*.part"))
    assert snapshot(man) == manifest_before
    out = capsys.readouterr().out
    assert (
        f"  MISMATCH    dcs/a.txt\n      expected {SHA}\n      got      {CORRUPT_SHA}\n"
        "      Not fixed: a fresh download failed, so the file is back as it was:\n"
        f"      {server.base}/a.txt\n      {why}"
    ) in out
    assert out.endswith("make data: FAILED (1 mismatch)\n")


def test_fix_interrupted_mid_download_puts_the_original_back(
    tmp_path: Path, monkeypatch: Patch
) -> None:
    def ctrl_c(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    man, raw = project(tmp_path, entry())
    put(raw, "dcs/a.txt", CORRUPT)
    monkeypatch.setattr(m.urllib.request, "urlopen", ctrl_c)
    assert run(man, raw, "--fix") == 130
    assert (raw / "dcs/a.txt").read_bytes() == CORRUPT
    assert not (raw / ".replaced").exists()
    assert not list(raw.rglob("*.part"))


def test_fix_does_nothing_offline(tmp_path: Path, server: Server, capsys: Capsys) -> None:
    man, raw = serve(tmp_path, server, a=[OK])
    put(raw, "dcs/a.txt", CORRUPT)
    assert run(man, raw, "--offline") == 1
    without_fix = capsys.readouterr().out
    assert run(man, raw, "--offline", "--fix") == 1
    assert capsys.readouterr().out == without_fix  # exactly as without FIX=1
    assert not server.hits
    assert (raw / "dcs/a.txt").read_bytes() == CORRUPT
    assert not (raw / ".replaced").exists()


def test_fix_and_an_unreachable_host(tmp_path: Path, capsys: Capsys) -> None:
    dead = f"127.0.0.1:{closed_port()}"
    man, raw = project(
        tmp_path,
        entry("dcs/a.txt", f"http://{dead}/a"),
        entry("dcs/b.txt", f"http://{dead}/b"),
        entry("dcs/c.txt", f"http://{dead}/c"),
    )
    put(raw, "dcs/a.txt", CORRUPT)
    put(raw, "dcs/c.txt", CORRUPT)
    assert run(man, raw, "--fix") == 1
    assert (raw / "dcs/a.txt").read_bytes() == CORRUPT
    assert (raw / "dcs/c.txt").read_bytes() == CORRUPT
    out = capsys.readouterr().out
    gone = f"{dead} was unreachable earlier in this run ({dead} refused the connection)"
    assert f"      {dead} refused the connection\n  SKIPPED     dcs/b.txt\n      {gone}\n" in out
    assert f"      Not fixed: {gone}\n" in out  # c: the host already failed


def test_fix_leaves_a_symlinked_file_alone(tmp_path: Path, server: Server, capsys: Capsys) -> None:
    man, raw = serve(tmp_path, server, a=[OK])
    put(tmp_path, "drive/a.txt", CORRUPT)
    (raw / "dcs").mkdir()
    (raw / "dcs/a.txt").symlink_to(tmp_path / "drive/a.txt")
    assert run(man, raw, "--fix") == 1
    assert (raw / "dcs/a.txt").is_symlink()
    assert (tmp_path / "drive/a.txt").read_bytes() == CORRUPT
    assert not server.hits
    assert (
        f"      Not fixed: it's a symlink to {tmp_path / 'drive/a.txt'}. Fix the file there.\n"
    ) in capsys.readouterr().out


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores folder permissions")
def test_fix_in_a_read_only_folder_leaves_the_file_alone(
    tmp_path: Path, server: Server, capsys: Capsys
) -> None:
    man, raw = serve(tmp_path, server, a=[OK])
    put(raw, "dcs/a.txt", CORRUPT)
    raw.chmod(0o555)
    try:
        assert run(man, raw, "--fix") == 1
    finally:
        raw.chmod(0o755)
    assert (raw / "dcs/a.txt").read_bytes() == CORRUPT
    assert not server.hits
    assert "      Not fixed: couldn't move it aside, so it's untouched ([Errno 13]" in (
        capsys.readouterr().out
    )


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
@pytest.mark.parametrize(
    ("args", "flags"),
    [
        (("data",), ""),
        (("data", "fix"), " --fix"),
        (("fix",), " --fix"),
        (("data", "FIX=1"), " --fix"),
        (("data", "OFFLINE=1"), " --offline"),
        (("data", "fix", "OFFLINE=1"), " --offline --fix"),  # offline wins: see test above
        (("data", "FIX=0"), ""),
    ],
)
def test_make_data_passes_the_flags(args: tuple[str, ...], flags: str) -> None:
    # make -n prints the commands without running them.
    out = subprocess.run(
        ["make", "-n", *args],
        cwd=m.paths.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert f"uv run --no-sync python -m sandhi_probe.data.manifest{flags}" in out.splitlines()


def test_failing_to_put_the_original_back_says_where_it_is(
    tmp_path: Path, server: Server, monkeypatch: Patch, capsys: Capsys
) -> None:
    man, raw = serve(tmp_path, server, a=[Reply(404, b"gone")])
    put(raw, "dcs/a.txt", CORRUPT)
    real_replace = os.replace

    def replace(src: object, dst: object) -> None:
        if ".replaced" in str(src):  # moving aside works, putting back doesn't
            raise OSError(errno.EROFS, "Read-only file system")
        real_replace(src, dst)

    monkeypatch.setattr(m.os, "replace", replace)
    assert run(man, raw, "--fix") == 1
    (old,) = (raw / ".replaced").rglob("a.txt")
    assert old.read_bytes() == CORRUPT
    out = capsys.readouterr().out
    assert (
        "  ERROR       dcs/a.txt\n      couldn't put the original back (Read-only file system);"
        f" it is in {old}\n"
    ) in out
    assert f"  Earlier `make data fix` runs kept old copies in {raw / '.replaced'}/.\n" in out


def test_a_failed_run_points_at_old_copies(tmp_path: Path, capsys: Capsys) -> None:
    # Say a fix run was killed mid-download: the file is only in .replaced/.
    man, raw = project(tmp_path, entry())
    put(raw, ".replaced/2026-11-14T120000-abc/dcs/a.txt", CORRUPT)
    assert run(man, raw, "--offline") == 1
    hint = f"  Earlier `make data fix` runs kept old copies in {raw / '.replaced'}/.\n"
    assert hint in capsys.readouterr().out
    put(raw)
    assert run(man, raw, "--offline") == 0
    assert hint not in capsys.readouterr().out  # only when something failed
