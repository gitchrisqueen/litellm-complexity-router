#!/usr/bin/env python3
"""Sample outside-anchored E1 rows from public datasets, deterministically.

Maintainer-run; CI never executes this (it needs the network). It reads rows
from the Hugging Face datasets-server JSON API (or a cache directory holding
those responses), applies the filters below, samples with a fixed seed, maps
each dataset's own label to a tier under the committed mapping (v1), assigns
splits by a fixed pattern, and prints E1 rows as JSON lines.

The tier label is a pure function of the source dataset's own field
(``level``, ``category`` or the config name); the person running this script
does not read the prompt to choose the tier. That is what ``anchored: true``
means in ``evals/datasets/SOURCES.md``.

Filters (all sources): no ``@``, no ``http``, no run of seven or more digits,
word count within the per-source cap. Non-code sources also drop rows with
any capitalised word outside a short allowlist of sentence starters and
common nouns - a crude proxy for "no personal names" (sentence-initial names
included), applied before sampling so the choice is never made by hand.

Usage::

    python scripts/sample_public_rows.py --cache-dir /path/to/responses > rows.jsonl
    python scripts/sample_public_rows.py --fetch --cache-dir /path/to/responses > rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

SEED = 20260904
API = "https://datasets-server.huggingface.co/rows"
DOLLY_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/"
    "databricks-dolly-15k.jsonl"
)

# ── committed mapping v1: source label -> tier ─────────────────────────────────
MAPPING_VERSION = "anchor-mapping-v1"
ARC_MAP = {"ARC-Easy": "SIMPLE", "ARC-Challenge": "MEDIUM"}
DOLLY_MAP = {
    "open_qa": "SIMPLE",
    "classification": "SIMPLE",
    "brainstorming": "MEDIUM",
    "creative_writing": "MEDIUM",
    "general_qa": "MEDIUM",
}
MATH_MAP = {
    "Level 1": "COMPLEX",
    "Level 2": "COMPLEX",
    "Level 4": "REASONING",
    "Level 5": "REASONING",
}
GSM8K_TIER = "COMPLEX"
HUMANEVAL_TIER = "COMPLEX"

# ── how many rows per (source, label) group, in a fixed order ─────────────────
PLAN: list[tuple[str, str, int]] = [
    ("arc-easy", "ARC-Easy", 7),
    ("dolly-15k", "open_qa", 5),
    ("dolly-15k", "classification", 3),
    ("arc-challenge", "ARC-Challenge", 6),
    ("dolly-15k", "brainstorming", 3),
    ("dolly-15k", "creative_writing", 3),
    ("dolly-15k", "general_qa", 3),
    ("gsm8k", "main", 5),
    ("humaneval", "openai_humaneval", 5),
    ("math", "Level 1-2", 5),
    ("math", "Level 4-5", 15),
]
MATH_SUBJECTS = ["algebra", "prealgebra", "number_theory", "counting_and_probability", "geometry"]

# split pattern, period 12: 5 test / 5 train / 2 dev  ->  25 / 25 / 10 over 60 rows
SPLIT_PATTERN = [
    "test",
    "train",
    "test",
    "train",
    "dev",
    "test",
    "train",
    "test",
    "train",
    "dev",
    "test",
    "train",
]

WORD_CAP = {
    "arc-easy": 40,
    "arc-challenge": 40,
    "dolly-15k": 40,
    "gsm8k": 60,
    "math": 60,
    "humaneval": 90,
}
CAP_ALLOW = {
    "Earth",
    "Sun",
    "Moon",
    "Mars",
    "English",
    "Python",
    "Which",
    "What",
    "How",
    "Why",
    "When",
    "Where",
    "Who",
    "If",
    "The",
    "A",
    "An",
    "In",
    "On",
    "At",
    "Is",
    "Are",
    "Do",
    "Does",
    "Can",
    "Express",
    "Find",
    "Compute",
    "Simplify",
    "Evaluate",
    "Solve",
    "Determine",
    "Suppose",
    "Let",
    "For",
    "Two",
    "Three",
    "Four",
    "Five",
    "One",
    "There",
    "It",
    "This",
    "That",
    "They",
    "We",
    "You",
    "I",
    "US",
    "USA",
    "UK",
    "DNA",
    "TV",
    "Celsius",
    "Fahrenheit",
    "Kelvin",
    "Latin",
    "Roman",
    "Greek",
    "Arabic",
    "French",
    "Spanish",
    "German",
    "Chinese",
    "Japanese",
    "Italian",
    "Africa",
    "Europe",
    "Asia",
    "America",
    "Pacific",
    "Atlantic",
    "North",
    "South",
    "East",
    "West",
    "Internet",
    "Wi-Fi",
    "Olympic",
    "Olympics",
    "Christmas",
    "Halloween",
    "Thanksgiving",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
    "Give",
    "List",
    "Name",
    "Write",
    "Describe",
    "Explain",
    "Identify",
    "Classify",
    "Tell",
    "Provide",
    "Suggest",
    "Imagine",
    "Create",
    "Compose",
    "Using",
    "Compared",
    "Based",
    "According",
    "Some",
    "Many",
    "Most",
    "Each",
    "All",
    "During",
    "After",
    "Before",
    "Over",
    "Under",
    "Between",
    "Water",
    "Plants",
    "Animals",
    "Scientists",
    "Students",
    "Humans",
    "People",
    "New",
    "Rewrite",
    "Summarize",
    "Convert",
    "Generate",
    "Please",
    "Make",
    "Help",
    "Come",
    "Pick",
    "Choose",
    "Sort",
    "Categorize",
    "Divide",
    "Put",
    "Say",
    "Decide",
    "Answer",
    "Recommend",
    "Brainstorm",
    "Think",
    "Given",
    "Consider",
    "Assume",
    "Was",
    "Were",
    "Will",
    "Would",
    "Should",
    "Could",
    "Did",
    "Has",
    "Have",
    "Had",
    "Extract",
    "Label",
    "Group",
    "Rank",
    "Order",
    "Arrange",
    "Fill",
    "Complete",
    "Count",
    "Show",
    "Prove",
    "Round",
    "Calculate",
    "Estimate",
    "Predict",
    "Sum",
    "Add",
    "Subtract",
    "Multiply",
}

EMAIL_OR_URL = re.compile(r"@|https?://|www\.", re.I)
LONG_DIGITS = re.compile(r"\d{7,}")
CAP_WORD = re.compile(r"\b([A-Z][a-z]+)\b")


def fetch_json(url: str, cache: Path | None, name: str, allow_fetch: bool) -> Any:
    if cache is not None:
        p = cache / name
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    if not allow_fetch:
        raise SystemExit(f"not cached and --fetch not given: {name}")
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - public API
        data = resp.read()
    if cache is not None:
        (cache / name).write_bytes(data)
    return json.loads(data)


def rows_api(
    dataset: str, config: str, split: str, offset: int, cache: Path | None, fetch: bool
) -> list[dict[str, Any]]:
    url = f"{API}?dataset={dataset}&config={config}&split={split}&offset={offset}&length=100"
    name = f"{dataset.replace('/', '_')}__{config}__{split}__{offset}.json"
    data = fetch_json(url, cache, name, fetch)
    return [{"row_idx": r["row_idx"], **r["row"]} for r in data["rows"]]


def dolly_rows(cache: Path | None, fetch: bool) -> list[dict[str, Any]]:
    name = "databricks_databricks-dolly-15k.jsonl"
    p = (cache / name) if cache is not None else None
    if p is None or not p.is_file():
        if not fetch:
            raise SystemExit("dolly file not cached and --fetch not given")
        with urllib.request.urlopen(DOLLY_URL, timeout=120) as resp:  # noqa: S310
            data = resp.read()
        if p is not None:
            p.write_bytes(data)
        text = data.decode("utf-8")
    else:
        text = p.read_text(encoding="utf-8")
    out = []
    for i, line in enumerate(text.splitlines()):
        if line.strip():
            out.append({"row_idx": i, **json.loads(line)})
    return out


def wc(s: str) -> int:
    return len(s.split())


def clean_ok(text: str, source: str) -> bool:
    if EMAIL_OR_URL.search(text) or LONG_DIGITS.search(text):
        return False
    if wc(text) > WORD_CAP[source] or wc(text) < 3:
        return False
    if source == "math" and "[asy]" in text:
        return False
    if source != "humaneval":
        for m in CAP_WORD.finditer(text):
            if m.group(1) not in CAP_ALLOW:
                return False
    return True


def candidates(source: str, label: str, cache: Path | None, fetch: bool) -> list[dict[str, Any]]:
    """Return candidate rows for one (source, label) group after filtering."""
    out: list[dict[str, Any]] = []
    if source in ("arc-easy", "arc-challenge"):
        for r in rows_api("allenai/ai2_arc", label, "test", 0, cache, fetch):
            q = r["question"].strip()
            if r["choices"]["label"] and clean_ok(q, source):
                opts = " ".join(
                    f"({lab}) {txt}"
                    for lab, txt in zip(r["choices"]["label"], r["choices"]["text"], strict=True)
                )
                prompt = f"{q} {opts}"
                if clean_ok(prompt, source) or wc(prompt) <= WORD_CAP[source] + 40:
                    out.append(
                        {
                            "prompt": prompt,
                            "row_idx": r["row_idx"],
                            "field": "config",
                            "value": label,
                            "dataset": "allenai/ai2_arc",
                            "config": label,
                            "hf_split": "test",
                            "tier": ARC_MAP[label],
                        }
                    )
    elif source == "dolly-15k":
        for r in dolly_rows(cache, fetch):
            if r["category"] != label or r["context"].strip():
                continue
            q = " ".join(r["instruction"].split())
            if clean_ok(q, source):
                out.append(
                    {
                        "prompt": q,
                        "row_idx": r["row_idx"],
                        "field": "category",
                        "value": label,
                        "dataset": "databricks/databricks-dolly-15k",
                        "config": "default",
                        "hf_split": "train",
                        "tier": DOLLY_MAP[label],
                    }
                )
    elif source == "gsm8k":
        for off in (0, 100):
            for r in rows_api("openai/gsm8k", "main", "test", off, cache, fetch):
                q = " ".join(r["question"].split())
                if clean_ok(q, source):
                    out.append(
                        {
                            "prompt": q,
                            "row_idx": r["row_idx"],
                            "field": "config",
                            "value": "main",
                            "dataset": "openai/gsm8k",
                            "config": "main",
                            "hf_split": "test",
                            "tier": GSM8K_TIER,
                        }
                    )
    elif source == "humaneval":
        for off in (0, 100):
            for r in rows_api(
                "openai/openai_humaneval", "openai_humaneval", "test", off, cache, fetch
            ):
                p = r["prompt"].rstrip() + "\n"
                if clean_ok(p, source):
                    out.append(
                        {
                            "prompt": "Complete this Python function:\n\n" + p,
                            "row_idx": r["row_idx"],
                            "field": "task_id",
                            "value": r["task_id"],
                            "dataset": "openai/openai_humaneval",
                            "config": "openai_humaneval",
                            "hf_split": "test",
                            "tier": HUMANEVAL_TIER,
                        }
                    )
    elif source == "math":
        levels = {"Level 1-2": ("Level 1", "Level 2"), "Level 4-5": ("Level 4", "Level 5")}[label]
        for subj in MATH_SUBJECTS:
            for r in rows_api("EleutherAI/hendrycks_math", subj, "test", 0, cache, fetch):
                q = " ".join(r["problem"].split())
                if r["level"] in levels and clean_ok(q, source):
                    out.append(
                        {
                            "prompt": q,
                            "row_idx": r["row_idx"],
                            "field": "level",
                            "value": r["level"],
                            "dataset": "EleutherAI/hendrycks_math",
                            "config": subj,
                            "hf_split": "test",
                            "tier": MATH_MAP[r["level"]],
                            "subject": subj,
                        }
                    )
    return out


def pick(
    group: list[dict[str, Any]], n: int, rng: random.Random, source: str, label: str
) -> list[dict[str, Any]]:
    if source == "math":
        # one per subject for Level 1-2 (5), three per subject for Level 4-5 (15)
        per = n // len(MATH_SUBJECTS)
        chosen: list[dict[str, Any]] = []
        for subj in MATH_SUBJECTS:
            pool = [g for g in group if g["subject"] == subj]
            if len(pool) < per:
                raise SystemExit(f"math/{subj}/{label}: only {len(pool)} candidates, need {per}")
            chosen += rng.sample(pool, per)
        return chosen
    if len(group) < n:
        raise SystemExit(f"{source}/{label}: only {len(group)} candidates, need {n}")
    return rng.sample(group, n)


LICENSE = {
    "arc-easy": "CC BY-SA 4.0",
    "arc-challenge": "CC BY-SA 4.0",
    "dolly-15k": "CC BY-SA 3.0",
    "gsm8k": "MIT",
    "humaneval": "MIT",
    "math": "MIT",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--fetch", action="store_true", help="allow network fetches (fills the cache)")
    ap.add_argument("--first-id", type=int, default=37, help="numeric suffix of the first row id")
    ap.add_argument(
        "--revisions", type=Path, default=None, help="JSON {dataset: revision sha} to stamp"
    )
    ap.add_argument("--report", action="store_true", help="print candidate counts to stderr")
    args = ap.parse_args(argv)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    revisions = json.loads(args.revisions.read_text()) if args.revisions else {}

    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    for source, label, n in PLAN:
        group = candidates(source, label, args.cache_dir, args.fetch)
        if args.report:
            print(f"{source:14s} {label:18s} candidates={len(group):4d} need={n}", file=sys.stderr)
        for g in pick(group, n, rng, source, label):
            g["source"] = source
            selected.append(g)

    next_id = args.first_id
    for i, g in enumerate(selected):
        split = SPLIT_PATTERN[i % len(SPLIT_PATTERN)]
        rationale = f"{MAPPING_VERSION}: {g['dataset']} {g['field']}={g['value']} -> {g['tier']}"
        row = {
            "id": f"e1-{next_id:03d}",
            "prompt": g["prompt"],
            "has_tools": False,
            "expected_tier": g["tier"],
            "source": f"public:{g['source']}",
            "split": split,
            "anchored": True,
            "anchor": {
                "dataset": g["dataset"],
                "revision": revisions.get(g["dataset"], "unrecorded"),
                "config": g["config"],
                "split": g["hf_split"],
                "row_idx": g["row_idx"],
                "field": g["field"],
                "value": g["value"],
                "mapping": MAPPING_VERSION,
            },
            "license": LICENSE[g["source"]],
            "rater": MAPPING_VERSION,
            "rationale": rationale,
        }
        print(json.dumps(row, ensure_ascii=False))
        next_id += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
