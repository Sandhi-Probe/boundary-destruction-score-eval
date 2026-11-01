"""Manifest loading and `make data` download/verify behaviour. No real network: downloads
go to a local HTTP server, and "offline" is a port nothing listens on."""

from __future__ import annotations

import errno
import hashlib
import os
import socket
import ssl
import textwrap
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sandhi_probe.data import manifest as m

COMMIT = "8aeed5a1343d0e48b64eb32af8c00e8c6eb29359"
BODY = b"agnim ile purohitam\n"
SHA = hashlib.sha256(BODY).hexdigest()


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def entry(path: str = "dcs/a.txt", url: str = "https://example.org/a.txt", sha: str = SHA) -> str:
    return f"  - path: {path}\n    url: {url}\n    sha256: {sha}\n    retrieved: 2026-10-31\n"


# --- loading -------------------------------------------------------------------------------


def test_loads_valid_manifest(tmp_path: Path) -> None:
    mf = m.load_manifest(write(tmp_path, "files:\n" + entry()))
    (f,) = mf.files
    assert (f.path, f.sha256, f.retrieved.isoformat()) == ("dcs/a.txt", SHA, "2026-10-31")


@pytest.mark.parametrize("text", ["files: []\n", "files:\n"])
def test_empty_file_list_is_allowed(tmp_path: Path, text: str) -> None:
    assert m.load_manifest(write(tmp_path, text)).files == []


def test_repo_manifest_is_valid() -> None:
    m.load_manifest(m.paths.MANIFEST_PATH)


def test_unquoted_all_digit_checksum_stays_a_string(tmp_path: Path) -> None:
    # yaml.safe_load would turn this into an int and the check would fail confusingly.
    sha = "1" * 64
    (f,) = m.load_manifest(write(tmp_path, "files:\n" + entry(sha=sha))).files
    assert f.sha256 == sha


def test_uppercase_checksum_is_normalised(tmp_path: Path) -> None:
    (f,) = m.load_manifest(write(tmp_path, "files:\n" + entry(sha=SHA.upper()))).files
    assert f.sha256 == SHA


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("", "mapping"),
        ("- just a list\n", "mapping"),
        ("files: [\n", "not valid YAML"),
        ("files:\n\t- path: a\n", "spaces, not tabs"),
        ("files:\n" + entry().replace("    url:", "    sha256: 00\n    url:"), "duplicate key"),
        ("files:\n  - {[a]: b}\n", "plain names"),
        ("files:\n" + entry() + "    licence: CC-BY\n", "allowed: path, url, sha256, retrieved"),
        ("fils: []\n", "only top-level key is 'files'"),
        ("files:\n" + entry().replace("    retrieved: 2026-10-31\n", ""), "retrieved"),
        ("files:\n" + entry(sha="abc"), "not a SHA-256"),
        ("files:\n" + entry().replace("2026-10-31", "31/10/2026"), "date"),
        ("files:\n" + entry().replace("2026-10-31", "2099-01-01"), "can't be right"),
        ("files:\n<<<<<<< HEAD\n" + entry() + "=======\n>>>>>>> main\n", "conflict at line 2"),
    ],
)
def test_rejects_bad_manifest(tmp_path: Path, text: str, needle: str) -> None:
    with pytest.raises(m.ManifestError, match=needle):
        m.load_manifest(write(tmp_path, text))


def test_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(m.ManifestError, match="not found"):
        m.load_manifest(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "path",
    ["../etc/passwd", "/abs/a.txt", "dcs//a.txt", "dcs/./a.txt", "dcs\\a.txt", ".hidden/a",
     "dcs/Kumārasaṃbhava.conllu", "dcs/a b.txt", "dcs/a.txt.part", "dcs/", "DCS/a.txt",
     "dcs/a.", "dcs/" + "a" * 101],
)  # fmt: skip
def test_rejects_unsafe_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(m.ManifestError, match=r"files\.0\.path.*lowercase"):
        m.load_manifest(write(tmp_path, "files:\n" + entry(path=f'"{path}"')))


@pytest.mark.parametrize(("a", "b"), [("dcs/a.txt", "dcs/a.txt"), ("dcs", "dcs/a.txt")])
def test_rejects_colliding_paths(tmp_path: Path, a: str, b: str) -> None:
    with pytest.raises(m.ManifestError, match="listed twice|uses it as a folder"):
        m.load_manifest(write(tmp_path, "files:\n" + entry(path=a) + entry(path=b)))


@pytest.mark.parametrize(
    ("url", "needle"),
    [
        ("https://github.com/OliverHellwig/sanskrit/tree/master/dcs/data", "web page"),
        ("https://github.com/OliverHellwig/sanskrit/blob/master/dcs/readme.md", "web page"),
        ("https://raw.githubusercontent.com/OliverHellwig/sanskrit/master/dcs/a.md", "can move"),
        ("https://RAW.GITHUBUSERCONTENT.COM/OliverHellwig/sanskrit/master/dcs/a.md", "can move"),
        ("https://github.com/OliverHellwig/sanskrit/raw/master/dcs/a.md", "can move"),
        ("https://github.com/OliverHellwig/sanskrit/archive/master.zip", "can move"),
        ("https://codeload.github.com/OliverHellwig/sanskrit/zip/master", "can move"),
        ("ftp://example.org/a.txt", "http"),
        ("example.org/a.txt", "http"),
        ('"http://[::1"', "http"),
        ("http://example.org:99999/a", "http"),
        ("http://a..b/x", "http"),
        ("https://user:hunter2@example.org/a", "credentials"),
        ('" https://example.org/a"', "one line"),
        ('"https://example.org/a\\n"', "one line"),
    ],
)
def test_rejects_bad_urls(tmp_path: Path, url: str, needle: str) -> None:
    with pytest.raises(m.ManifestError, match=needle):
        m.load_manifest(write(tmp_path, "files:\n" + entry(url=url)))


def test_accepts_commit_pinned_github_url(tmp_path: Path) -> None:
    url = f"https://raw.githubusercontent.com/OliverHellwig/sanskrit/{COMMIT}/dcs/data/readme.md"
    m.load_manifest(write(tmp_path, "files:\n" + entry(url=url)))


def test_error_names_the_entry(tmp_path: Path) -> None:
    with pytest.raises(m.ManifestError, match=r"files\.0\.sha256 \(dcs/a\.txt\)"):
        m.load_manifest(write(tmp_path, "files:\n" + entry(sha="nope")))


# --- local HTTP server -----------------------------------------------------------------------


class Server:
    """routes: path -> list of responses, served in order (the last one repeats).
    A response is (status, body[, headers[, delay_s]]); status None closes the connection
    without answering."""

    def __init__(self) -> None:
        self.routes: dict[str, list[tuple]] = {}
        self.hits: dict[str, int] = {}
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                srv.hits[self.path] = srv.hits.get(self.path, 0) + 1
                seq = srv.routes.get(self.path, [(404, b"not found")])
                status, body, *rest = seq[min(srv.hits[self.path], len(seq)) - 1]
                headers = rest[0] if rest else {}
                threading.Event().wait(rest[1] if len(rest) > 1 else 0)  # time.sleep is patched
                if status is None:
                    return
                self.send_response(status)
                for k, v in {"Content-Length": str(len(body)), **headers}.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, args=(0.01,), daemon=True).start()


@pytest.fixture
def server() -> Iterator[Server]:
    s = Server()
    yield s
    s.httpd.shutdown()
    s.httpd.server_close()


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Backoff sleeps are recorded instead of slept."""
    calls: list[float] = []
    monkeypatch.setattr(m.time, "sleep", calls.append)
    return calls


def closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make(tmp_path: Path, *entries: str) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    return write(tmp_path, "files:\n" + "".join(entries)), raw


def run(man: Path, raw: Path, *extra: str) -> int:
    return m.main(["--manifest", str(man), "--raw-dir", str(raw), "--timeout", "2", *extra])


# --- sync ------------------------------------------------------------------------------------


def test_downloads_missing_file_then_verifies_without_network(
    tmp_path: Path, server: Server
) -> None:
    server.routes["/a.txt"] = [(200, BODY)]
    man, raw = make(tmp_path, entry(url=server.base + "/a.txt"))
    assert run(man, raw) == 0
    assert (raw / "dcs/a.txt").read_bytes() == BODY
    assert run(man, raw) == 0
    assert server.hits["/a.txt"] == 1  # second run only hashed the local file


def test_downloaded_file_has_normal_permissions(tmp_path: Path, server: Server) -> None:
    server.routes["/a.txt"] = [(200, BODY)]
    man, raw = make(tmp_path, entry(url=server.base + "/a.txt"))
    assert run(man, raw) == 0
    assert (raw / "dcs/a.txt").stat().st_mode & 0o777 == 0o666 & ~m._umask()


def test_existing_mismatch_fails_and_is_not_overwritten(
    tmp_path: Path, server: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    server.routes["/a.txt"] = [(200, BODY)]
    man, raw = make(tmp_path, entry(url=server.base + "/a.txt"))
    (raw / "dcs").mkdir()
    (raw / "dcs/a.txt").write_bytes(b"edited by hand\n")
    assert run(man, raw) == 1
    assert (raw / "dcs/a.txt").read_bytes() == b"edited by hand\n"
    out = capsys.readouterr().out
    assert "MISMATCH" in out
    assert "Move it out of data/raw/ rather than deleting it" in out
    assert "/a.txt" not in server.hits


def test_downloaded_mismatch_leaves_nothing_behind(
    tmp_path: Path, server: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    server.routes["/a.txt"] = [(200, b"<html>Please log in to campus wifi</html>")]
    man, raw = make(tmp_path, entry(url=server.base + "/a.txt"))
    assert run(man, raw) == 1
    assert sorted(p.name for p in raw.rglob("*")) == [m.LOCK_NAME, "dcs"]  # no file, no .part
    assert "login page" in capsys.readouterr().out
    assert server.hits["/a.txt"] == 1  # a wrong checksum is not retried


def test_mismatch_hint_for_a_pinned_commit() -> None:
    url = f"https://raw.githubusercontent.com/OliverHellwig/sanskrit/{COMMIT}/dcs/a.md"
    assert "sha256 or url in the manifest is wrong" in m._mismatch_hint(url, html=False)


def test_404_fails_without_retrying_or_skipping_the_host(tmp_path: Path, server: Server) -> None:
    server.routes["/b.txt"] = [(200, BODY)]
    man, raw = make(
        tmp_path,
        entry("dcs/a.txt", server.base + "/gone.txt"),
        entry("dcs/b.txt", server.base + "/b.txt"),
    )
    assert run(man, raw) == 1
    assert server.hits == {"/gone.txt": 1, "/b.txt": 1}


def test_retries_server_errors(tmp_path: Path, server: Server, sleeps: list[float]) -> None:
    server.routes["/a.txt"] = [(503, b"busy"), (500, b"oops"), (200, BODY)]
    man, raw = make(tmp_path, entry(url=server.base + "/a.txt"))
    assert run(man, raw) == 0
    assert server.hits["/a.txt"] == 3
    assert sleeps == [2, 4]


def test_honours_retry_after_up_to_a_cap(
    tmp_path: Path, server: Server, sleeps: list[float]
) -> None:
    server.routes["/a.txt"] = [
        (429, b"", {"Retry-After": "7"}),
        (429, b"", {"Retry-After": "999"}),
        (200, BODY),
    ]
    man, raw = make(tmp_path, entry(url=server.base + "/a.txt"))
    assert run(man, raw) == 0
    assert sleeps == [7, m.MAX_WAIT_S]


def test_gives_up_after_three_attempts_and_skips_the_host(
    tmp_path: Path, server: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    server.routes["/a.txt"] = [(503, b"busy")]
    server.routes["/b.txt"] = [(200, BODY)]
    man, raw = make(
        tmp_path,
        entry("dcs/a.txt", server.base + "/a.txt"),
        entry("dcs/b.txt", server.base + "/b.txt"),
    )
    assert run(man, raw) == 1
    assert server.hits == {"/a.txt": m.ATTEMPTS}
    assert "kept failing earlier in this run (HTTP 503" in capsys.readouterr().out


def test_truncated_download_is_retried(tmp_path: Path, server: Server) -> None:
    # Wifi drops mid-file: the server promised more bytes than it sent.
    big = {"Content-Length": str(len(BODY) + 100)}
    server.routes["/a.txt"] = [(200, BODY[:5], big), (200, BODY)]
    man, raw = make(tmp_path, entry(url=server.base + "/a.txt"))
    assert run(man, raw) == 0
    assert (raw / "dcs/a.txt").read_bytes() == BODY


def test_closed_without_response_is_retried(tmp_path: Path, server: Server) -> None:
    server.routes["/a.txt"] = [(None, b""), (200, BODY)]
    man, raw = make(tmp_path, entry(url=server.base + "/a.txt"))
    assert run(man, raw) == 0
    assert server.hits["/a.txt"] == 2


def test_no_response_within_timeout_is_retried_then_bounded(
    tmp_path: Path, server: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    server.routes["/a.txt"] = [(200, BODY, {}, 1.0)]
    man, raw = make(tmp_path, entry(url=server.base + "/a.txt"))
    start = time.monotonic()
    assert run(man, raw, "--timeout", "0.2") == 1
    assert time.monotonic() - start < 3
    assert server.hits["/a.txt"] == m.ATTEMPTS
    assert "got no response within 0.2s" in capsys.readouterr().out


def test_disk_full_is_reported_and_not_retried(
    tmp_path: Path,
    server: Server,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.routes["/a.txt"] = [(200, BODY)]
    man, raw = make(tmp_path, entry(url=server.base + "/a.txt"))
    real_fdopen = os.fdopen

    class Full:
        def __init__(self, f: object) -> None:
            self.f = f

        def __enter__(self) -> Full:
            return self

        def __exit__(self, *exc: object) -> None:
            self.f.close()

        def fileno(self) -> int:
            return self.f.fileno()

        def write(self, b: bytes) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(m.os, "fdopen", lambda fd, *a, **k: Full(real_fdopen(fd, *a, **k)))
    assert run(man, raw) == 1
    assert server.hits["/a.txt"] == 1
    assert "No space left on device" in capsys.readouterr().out
    assert not list(raw.rglob("*.part"))


def test_offline_flag_never_downloads(
    tmp_path: Path, server: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    server.routes["/a.txt"] = [(200, BODY)]
    man, raw = make(tmp_path, entry(url=server.base + "/a.txt"))
    assert run(man, raw, "--offline") == 1
    assert not server.hits
    assert "without OFFLINE=1" in capsys.readouterr().out


def test_offline_still_verifies_present_files(tmp_path: Path) -> None:
    man, raw = make(tmp_path, entry())
    (raw / "dcs").mkdir()
    (raw / "dcs/a.txt").write_bytes(BODY)
    assert run(man, raw, "--offline") == 0


def test_unreachable_host_fails_fast_and_skips_only_its_own_files(
    tmp_path: Path, server: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    server.routes["/c.txt"] = [(200, BODY)]
    dead = f"http://127.0.0.1:{closed_port()}"
    man, raw = make(
        tmp_path,
        entry("dcs/a.txt", dead + "/a"),
        entry("dcs/b.txt", dead + "/b"),
        entry("dcs/c.txt", server.base + "/c.txt"),
    )
    assert run(man, raw) == 1
    out = capsys.readouterr().out
    assert "ERROR       dcs/a.txt" in out
    assert "refused the connection" in out
    assert "SKIPPED     dcs/b.txt" in out
    assert "downloaded  dcs/c.txt" in out  # same machine, different port: not skipped
    assert "re-run once the problem above is fixed" in out


def test_tls_failure_message_points_at_the_network() -> None:
    e = ssl.SSLCertVerificationError(1, "certificate verify failed")
    e.verify_message = "self-signed certificate"
    msg = m._cant_connect(e, "sanskrit.uohyd.ac.in", 30)
    assert "isn't trusted (self-signed certificate)" in msg
    assert "login page" in msg


def test_non_ascii_url_is_percent_encoded(tmp_path: Path, server: Server) -> None:
    quoted = "/Kum%C4%81rasa%E1%B9%83bhava,%201.conllu"
    server.routes[quoted] = [(200, BODY)]
    man, raw = make(tmp_path, entry(url=f'"{server.base}/Kumārasaṃbhava, 1.conllu"'))
    assert run(man, raw) == 0


def test_one_crashing_entry_does_not_stop_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    man, raw = make(tmp_path, entry("dcs/a.txt"), entry("dcs/b.txt"))
    (raw / "dcs").mkdir()
    (raw / "dcs/a.txt").write_bytes(BODY)
    (raw / "dcs/b.txt").write_bytes(BODY)
    real = m.sha256_of

    def flaky(p: Path) -> str:
        if p.name == "a.txt":
            raise RuntimeError("boom")
        return real(p)

    monkeypatch.setattr(m, "sha256_of", flaky)
    assert run(man, raw, "--offline") == 1
    out = capsys.readouterr().out
    assert "unexpected RuntimeError: boom" in out
    assert "ok          dcs/b.txt" in out


@pytest.mark.parametrize("kind", ["dir", "fifo"])
def test_non_regular_destination_is_an_error_not_a_hang(tmp_path: Path, kind: str) -> None:
    man, raw = make(tmp_path, entry())
    (raw / "dcs").mkdir()
    if kind == "dir":
        (raw / "dcs/a.txt").mkdir()
    else:
        os.mkfifo(raw / "dcs/a.txt")
    assert run(man, raw, "--offline") == 1


def test_broken_raw_dir_symlink(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    man = write(tmp_path, "files:\n")
    raw = tmp_path / "raw"
    raw.symlink_to(tmp_path / "unmounted-drive")
    assert run(man, raw) == 1
    assert "drive not mounted" in capsys.readouterr().out


def test_second_run_on_the_same_folder_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    man, raw = make(tmp_path)
    held = m._lock(raw)
    try:
        assert run(man, raw) == 1
        assert "another `make data` is running" in capsys.readouterr().out
    finally:
        held.close()
    assert run(man, raw) == 0


def test_warns_about_unlisted_files_and_removes_stale_partials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    man, raw = make(tmp_path)
    for name in (".gitkeep", ".DS_Store", "._stray.txt", "stray.txt", ".a.txt.x1.part"):
        (raw / name).touch()
    assert run(man, raw) == 0
    out = capsys.readouterr().out
    assert "stray.txt is not in the manifest" in out
    assert "removed 1 leftover .part file(s)" in out
    assert not (raw / ".a.txt.x1.part").exists()
    assert ".gitkeep" not in out and ".DS_Store" not in out and "._stray" not in out


def test_case_only_difference_says_rename_not_delete(tmp_path: Path) -> None:
    man, raw = make(tmp_path, entry())
    (raw / "dcs").mkdir()
    (raw / "dcs/A.txt").write_bytes(BODY)
    (w,) = m.untracked(m.load_manifest(man), raw)
    assert w == "dcs/A.txt is spelled dcs/a.txt in the manifest; rename it"


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "soon"])
def test_rejects_bad_timeout(tmp_path: Path, value: str) -> None:
    man, raw = make(tmp_path)
    with pytest.raises(SystemExit):
        run(man, raw, "--timeout", value)


def test_invalid_manifest_exits_2(tmp_path: Path) -> None:
    man, raw = make(tmp_path, entry(sha="nope"))
    assert run(man, raw) == 2
