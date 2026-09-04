"""The guard is tested against throwaway terms in a temporary directory only.
No real denied term appears in this file."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1] / "scripts" / "denylist_guard.py"
REAL_DENYLIST = Path(__file__).resolve().parents[1] / "scripts" / "denylist.sha256"


def load_guard():
    spec = importlib.util.spec_from_file_location("denylist_guard", GUARD)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def h(term: str) -> str:
    return hashlib.sha256(term.lower().encode()).hexdigest()


def write_denylist(path: Path, terms: list[str]) -> Path:
    path.write_text("# test\n" + "\n".join(h(t) for t in terms) + "\n", encoding="utf-8")
    return path


def test_candidates_cover_tokens_words_and_substrings() -> None:
    g = load_guard()
    got = set(g.candidates("Alpha-Beta zetaHost x:y"))
    assert "alpha-beta" in got
    assert "alpha" in got and "beta" in got
    assert "zeta" in got  # substring of zetahost
    assert "x:y" in got


def test_clean_tree_reports_zero_and_exits_zero(tmp_path: Path) -> None:
    g = load_guard()
    (tmp_path / "a.txt").write_text("nothing to see here\n")
    dl = write_denylist(tmp_path / "dl.sha256", ["zzqxterm", "other-secret-alias"])
    rc = g.main(["--denylist", str(dl), "--root", str(tmp_path), "--no-history", "--quiet"])
    assert rc == 0


def test_hit_in_file_content_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    g = load_guard()
    (tmp_path / "a.txt").write_text("this mentions ZZQXTERM inside\n")
    dl = write_denylist(tmp_path / "dl.sha256", ["zzqxterm"])
    rc = g.main(["--denylist", str(dl), "--root", str(tmp_path), "--no-history"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "zzqxterm" not in out  # the report never names the term
    assert "HIT" in out


def test_hit_embedded_in_identifier_fails(tmp_path: Path) -> None:
    g = load_guard()
    (tmp_path / "a.py").write_text("PATH = '/srv/zzqxtermhost/data'\n")
    dl = write_denylist(tmp_path / "dl.sha256", ["zzqxterm"])
    assert g.main(["--denylist", str(dl), "--root", str(tmp_path), "--no-history", "--quiet"]) == 1


def test_hit_in_file_name_fails(tmp_path: Path) -> None:
    g = load_guard()
    (tmp_path / "zzqxterm-notes.md").write_text("clean\n")
    dl = write_denylist(tmp_path / "dl.sha256", ["zzqxterm"])
    assert g.main(["--denylist", str(dl), "--root", str(tmp_path), "--no-history", "--quiet"]) == 1


def test_hit_in_git_history_only_fails(tmp_path: Path) -> None:
    g = load_guard()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    env_args = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    f = tmp_path / "a.txt"
    f.write_text("ZZQXTERM was here\n")
    subprocess.run(["git", *env_args, "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", *env_args, "commit", "-q", "-m", "one"], cwd=tmp_path, check=True)
    f.write_text("clean now\n")
    subprocess.run(["git", *env_args, "commit", "-q", "-am", "two"], cwd=tmp_path, check=True)
    dl = write_denylist(tmp_path / "dl.sha256", ["zzqxterm"])
    # Tree only: clean (the denylist file itself holds hashes, not the term).
    assert g.main(["--denylist", str(dl), "--root", str(tmp_path), "--no-history", "--quiet"]) == 0
    # With history: the first commit's content is found.
    assert g.main(["--denylist", str(dl), "--root", str(tmp_path), "--quiet"]) == 1


def test_hit_in_commit_message_fails(tmp_path: Path) -> None:
    g = load_guard()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    env_args = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    (tmp_path / "a.txt").write_text("clean\n")
    subprocess.run(["git", *env_args, "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", *env_args, "commit", "-q", "-m", "mention zzqxterm"], cwd=tmp_path, check=True
    )
    dl = write_denylist(tmp_path / "dl.sha256", ["zzqxterm"])
    assert g.main(["--denylist", str(dl), "--root", str(tmp_path), "--quiet"]) == 1


def test_malformed_denylist_is_rejected(tmp_path: Path) -> None:
    g = load_guard()
    bad = tmp_path / "dl.sha256"
    bad.write_text("not-a-hash\n")
    with pytest.raises(SystemExit):
        g.load_denylist(bad)


def test_committed_denylist_is_well_formed() -> None:
    g = load_guard()
    hashes = g.load_denylist(REAL_DENYLIST)
    assert len(hashes) >= 60  # the rule covers 62 context-limit keys alone
    assert len(set(hashes)) == len(hashes)
    assert hashes == sorted(hashes)


def test_guard_runs_as_a_script_on_this_repo() -> None:
    proc = subprocess.run(
        [sys.executable, str(GUARD), "--quiet"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "CLEAN" in proc.stdout
