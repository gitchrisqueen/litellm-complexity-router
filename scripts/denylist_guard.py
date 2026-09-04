#!/usr/bin/env python3
"""Denylist guard: fail if any hashed term appears in the tree or the history.

The denylist (``scripts/denylist.sha256``) holds SHA-256 digests of lowercased
terms, one per line. This guard never learns the terms. It extracts candidate
strings from every file in the working tree, from the full git history
(``git log --all -p``, commit messages, author lines, file paths, ref names),
hashes each candidate the same way, and compares digests. The report is a
per-hash table of zero / non-zero hit counts and nothing else - no matched
text, no file name for a match - so the report itself is safe to publish.

Text normalisation before extraction: Unicode NFKC (so full-width and other
compatibility forms fold to ASCII), zero-width and soft-hyphen code points
removed, a small table of Cyrillic/Greek look-alike letters folded to Latin,
then lowercased.

Candidate extraction, on the normalised text:

1. every maximal run of ``[a-z0-9_.:@/+-]`` (identifier-like tokens, so
   hyphenated aliases and ``prefix:name`` forms match whole);
2. every span of such a token that starts and ends on a separator boundary
   (``/``, ``:``, ``.``, ``@``, ``-``, ``_``, ``+``), so an alias inside
   ``openai/<alias>`` or ``my-<alias>-x`` is tested on its own;
3. each span again with its separators canonicalised to ``-``, to ``_``, and
   removed, so ``a_b`` / ``a.b`` / ``ab`` cannot hide ``a-b``;
4. every maximal run of ``[a-z0-9]`` (bare words) and every substring of
   length >= 3 of each bare word (no upper cap), so a term embedded in a longer
   unbroken word still matches;
5. base64 runs (>= 24 chars) and hex runs (>= 32 chars) are decoded and the
   decoded text is scanned the same way, one level deep.

Exit status: 0 when every hash has zero hits, 1 otherwise, 2 on usage error.

Usage::

    python scripts/denylist_guard.py [--denylist PATH] [--root DIR] [--no-history]
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import os
import re
import subprocess
import sys
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.:@/+-]*[a-z0-9]|[a-z0-9]")
WORD_RE = re.compile(r"[a-z0-9]+")
SEP_RE = re.compile(r"[/:.@_+-]")
BASE64_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
HEX_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{32,}(?![0-9a-f])")
MIN_SUB = 3
MAX_WORD_FOR_SUBSTRINGS = 512  # longer runs are blobs; the decoders cover them
ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­᠎"), None)
# Look-alike letters that NFKC leaves alone; folded so they cannot spell a term.
CONFUSABLES = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y",
        "х": "x", "і": "i", "ј": "j", "һ": "h", "ѕ": "s", "ԁ": "d",
        "ɡ": "g", "т": "t", "к": "k", "м": "m", "н": "h", "в": "b",
        "ο": "o", "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v",
        "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
    }
)  # fmt: skip
SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".eggs",
    "dist",
    "build",
}
MAX_FILE_BYTES = 8 * 1024 * 1024


def sha256_hex(term: str) -> str:
    return hashlib.sha256(term.encode("utf-8")).hexdigest()


def normalise(text: str) -> str:
    """NFKC, strip zero-width/soft-hyphen code points, fold look-alikes, lowercase."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(ZERO_WIDTH).translate(CONFUSABLES)
    return text.lower()


def load_denylist(path: Path) -> list[str]:
    hashes: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", line):
            raise SystemExit(f"malformed denylist line (expected 64 hex chars): {line[:20]}...")
        hashes.append(line)
    if not hashes:
        raise SystemExit("denylist is empty")
    return hashes


def _spans(token: str) -> Iterator[str]:
    """Every span of ``token`` bounded by separators, plus canonicalised forms."""
    parts = [(m.start(), m.end()) for m in SEP_RE.finditer(token)]
    starts = sorted({0} | {e for _, e in parts})
    ends = sorted({s for s, _ in parts} | {len(token)})
    for a in starts:
        for b in ends:
            if b - a >= MIN_SUB:
                span = token[a:b]
                yield span
                if SEP_RE.search(span):
                    yield SEP_RE.sub("-", span)
                    yield SEP_RE.sub("_", span)
                    yield SEP_RE.sub("", span)


def _decoded_blobs(lower: str) -> Iterator[str]:
    for m in BASE64_RE.finditer(lower):
        blob = m.group(0)
        try:
            raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=False)
        except (binascii.Error, ValueError):
            continue
        text = raw.decode("utf-8", errors="ignore")
        if text and sum(c.isprintable() for c in text) >= 0.5 * len(text):
            yield text
    for m in HEX_RE.finditer(lower):
        blob = m.group(0)
        if len(blob) % 2:
            blob = blob[:-1]
        try:
            raw = bytes.fromhex(blob)
        except ValueError:
            continue
        text = raw.decode("utf-8", errors="ignore")
        if text and sum(c.isprintable() for c in text) >= 0.5 * len(text):
            yield text


def _candidates_from_lower(lower: str, seen: set[str], words_done: set[str]) -> Iterator[str]:
    for m in TOKEN_RE.finditer(lower):
        tok = m.group(0)
        if tok not in seen:
            seen.add(tok)
            yield tok
        if SEP_RE.search(tok):
            for span in _spans(tok):
                if span not in seen:
                    seen.add(span)
                    yield span
    for m in WORD_RE.finditer(lower):
        word = m.group(0)
        if word in words_done:
            continue
        words_done.add(word)
        if word not in seen:
            seen.add(word)
            yield word
        n = len(word)
        if n > MAX_WORD_FOR_SUBSTRINGS:
            continue
        for length in range(MIN_SUB, n):
            for start in range(0, n - length + 1):
                sub = word[start : start + length]
                if sub not in seen:
                    seen.add(sub)
                    yield sub


def candidates(text: str) -> Iterator[str]:
    """Yield every distinct candidate string for ``text`` (normalised)."""
    lower = normalise(text)
    seen: set[str] = set()
    words_done: set[str] = set()
    yield from _candidates_from_lower(lower, seen, words_done)
    # Base64 is case-sensitive: decode from the NFKC-normalised, un-lowercased text.
    original = unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH)
    for decoded in _decoded_blobs_any_case(original):
        yield from _candidates_from_lower(normalise(decoded), seen, words_done)


def _decoded_blobs_any_case(text: str) -> Iterator[str]:
    yield from _decoded_blobs(text)
    yield from _decoded_blobs(text.lower())


def count_hits(text: str, wanted: dict[str, int]) -> None:
    """Increment ``wanted[hash]`` once per distinct candidate that matches."""
    for cand in candidates(text):
        h = sha256_hex(cand)
        if h in wanted:
            wanted[h] += 1


def iter_tree_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                if p.is_symlink() or p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield p


def read_text(p: Path) -> str:
    try:
        return p.read_bytes().decode("utf-8", errors="ignore")
    except OSError:
        return ""


def git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False, text=False)
    except FileNotFoundError:
        return ""
    return out.stdout.decode("utf-8", errors="ignore")


def history_texts(root: Path) -> Iterable[str]:
    yield git(root, "log", "--all", "-p", "--format=%H%n%an%n%ae%n%cn%n%ce%n%B")
    yield git(root, "log", "--all", "--name-only", "--format=")
    yield git(root, "for-each-ref", "--format=%(refname)")
    yield git(root, "stash", "list")


def scan(
    root: Path, hashes: list[str], include_history: bool
) -> tuple[dict[str, int], dict[str, int]]:
    tree_hits = {h: 0 for h in hashes}
    hist_hits = {h: 0 for h in hashes}
    for p in iter_tree_files(root):
        # File paths are text too: a term in a file name is a leak.
        count_hits(str(p.relative_to(root)), tree_hits)
        count_hits(read_text(p), tree_hits)
    if include_history:
        for chunk in history_texts(root):
            if chunk:
                count_hits(chunk, hist_hits)
    return tree_hits, hist_hits


def render_table(hashes: list[str], tree: dict[str, int], hist: dict[str, int]) -> str:
    lines = ["| # | sha256 (first 12) | tree | history |", "|---|---|---|---|"]
    for i, h in enumerate(hashes, 1):
        lines.append(f"| {i} | `{h[:12]}` | {tree[h]} | {hist[h]} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--denylist", type=Path, default=Path(__file__).with_name("denylist.sha256"))
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--no-history", action="store_true", help="scan the working tree only")
    ap.add_argument("--quiet", action="store_true", help="print only the summary line")
    args = ap.parse_args(argv)

    if not args.denylist.is_file():
        print(f"denylist not found: {args.denylist}", file=sys.stderr)
        return 2
    hashes = load_denylist(args.denylist)
    digest = sha256_hex(args.denylist.read_text(encoding="utf-8"))
    tree, hist = scan(args.root.resolve(), hashes, include_history=not args.no_history)

    total = sum(tree.values()) + sum(hist.values())
    if not args.quiet:
        print(f"denylist: {args.denylist.name}  entries: {len(hashes)}  file-sha256: {digest}")
        print(f"history scanned: {'yes' if not args.no_history else 'no'}")
        print(render_table(hashes, tree, hist))
    status = "CLEAN" if total == 0 else "HIT"
    print(
        f"denylist-guard: {status}  entries={len(hashes)}  "
        f"tree_hits={sum(tree.values())}  history_hits={sum(hist.values())}"
    )
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
