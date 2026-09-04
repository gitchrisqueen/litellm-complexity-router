#!/usr/bin/env python3
"""Generate ``scripts/denylist.sha256`` by rule from PRIVATE sources.

This script runs only on the maintainer's machine. It reads files that are not
in this repository and writes hashes only. The committed artifact carries no
plaintext term, and CI never runs this script.

Sources, each supplied by an environment variable and read only if set:

- ``DENYLIST_SOURCE_MODULE``: path to the private router module. Every key of
  its ``MODEL_CONTEXT_LIMITS`` dict and every *value* of its ``TIER_MODELS``
  dict is added. Both are found by parsing the file's AST; the module is never
  imported.
- ``DENYLIST_SOURCE_CONFIG``: path to the private proxy config. Every key
  under ``router_settings.model_group_alias`` is added.
- ``DENYLIST_EXTRA_TERMS``: path to a private newline-separated file of extra
  terms (system name, hostnames, tracker ids, and the like).

Rule exceptions, stated in plaintext because they are generic words the
repository must be allowed to use: the *keys* of ``TIER_MODELS`` are the
four tier names (SIMPLE, MEDIUM, COMPLEX, REASONING) that the README
publishes, and the word ``compaction`` is a key of the source dict but also
the neutral family name the extraction plan assigns to that model family.
Excluded terms are counted in the header comment so the exception is visible.

Terms are lowercased before hashing (the guard lowercases all text). Terms
shorter than 3 characters are skipped as unguardable and reported.
"""

from __future__ import annotations

import ast
import hashlib
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

RULE_EXCEPTIONS = {"compaction", "simple", "medium", "complex", "reasoning"}
MIN_TERM_LEN = 3


def _dict_from_ast(tree: ast.Module, name: str) -> ast.Dict | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id == name and isinstance(node.value, ast.Dict):
                    return node.value
    return None


def _string_keys(d: ast.Dict) -> list[str]:
    return [k.value for k in d.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]


def _string_values(d: ast.Dict) -> list[str]:
    return [v.value for v in d.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]


def terms_from_module(path: Path) -> tuple[list[str], dict[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    terms: list[str] = []
    limits = _dict_from_ast(tree, "MODEL_CONTEXT_LIMITS")
    if limits is None:
        raise SystemExit("MODEL_CONTEXT_LIMITS not found in source module")
    keys = _string_keys(limits)
    counts["MODEL_CONTEXT_LIMITS.keys"] = len(keys)
    terms.extend(keys)
    tier_models = _dict_from_ast(tree, "TIER_MODELS")
    if tier_models is None:
        raise SystemExit("TIER_MODELS not found in source module")
    values = _string_values(tier_models)
    counts["TIER_MODELS.values"] = len(values)
    terms.extend(values)
    counts["TIER_MODELS.keys(excluded)"] = len(_string_keys(tier_models))
    return terms, counts


def terms_from_config(path: Path) -> tuple[list[str], dict[str, int]]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    aliases = ((loaded.get("router_settings") or {}).get("model_group_alias")) or {}
    keys = [str(k) for k in aliases]
    return keys, {"config.model_group_alias.keys": len(keys)}


def terms_from_extra(path: Path) -> tuple[list[str], dict[str, int]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    terms = [ln for ln in lines if ln and not ln.startswith("#")]
    return terms, {"extra_terms": len(terms)}


def main() -> int:
    out = Path(__file__).with_name("denylist.sha256")
    all_terms: list[str] = []
    counts: dict[str, int] = {}
    for env, fn in (
        ("DENYLIST_SOURCE_MODULE", terms_from_module),
        ("DENYLIST_SOURCE_CONFIG", terms_from_config),
        ("DENYLIST_EXTRA_TERMS", terms_from_extra),
    ):
        p = os.environ.get(env)
        if not p:
            print(f"{env} not set - skipped", file=sys.stderr)
            continue
        terms, c = fn(Path(p).expanduser())
        all_terms.extend(terms)
        counts.update(c)

    if not all_terms:
        print("no sources provided; nothing written", file=sys.stderr)
        return 2

    normalised: set[str] = set()
    excluded = 0
    too_short = 0
    for t in all_terms:
        low = t.strip().lower()
        if low in RULE_EXCEPTIONS:
            excluded += 1
            continue
        if len(low) < MIN_TERM_LEN:
            too_short += 1
            continue
        normalised.add(low)

    hashes = sorted(hashlib.sha256(t.encode("utf-8")).hexdigest() for t in normalised)
    header = [
        "# denylist.sha256 - SHA-256 of lowercased denied terms, one per line.",
        "# Generated by scripts/generate_denylist.py from private sources; regenerate",
        "# locally whenever those sources change. The guard (scripts/denylist_guard.py)",
        "# hashes every candidate token in the tree and history and compares.",
        f"# generated: {datetime.now(UTC).date().isoformat()}",
        "# source counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        f"# rule exceptions applied: {excluded}; skipped (<{MIN_TERM_LEN} chars): {too_short}",
        f"# distinct terms hashed: {len(hashes)}",
    ]
    out.write_text("\n".join(header + hashes) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(hashes)} hashes")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"  excluded by rule exception: {excluded}; too short: {too_short}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
