"""The seven-dimension weighted complexity scorer.

``score_complexity`` turns the last user message of a chat request into a
score in ``[0.0, 1.0]``. Seven dimensions are each normalised to ``[0, 1]``
and combined with fixed weights; ``simpleIndicators`` is subtracted. The
positive weights sum to 0.95, so the achievable maximum is 0.95, not 1.0.

Only the last user message is scored. System prompts are static and dense
with technical vocabulary, so scoring the whole conversation would push every
request into the top tier regardless of what the user asked. Configurable
``strip_patterns`` remove injected context blocks before scoring, for the same
reason; if stripping empties the message, the unstripped text is scored
instead (see :func:`extract_text`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any, Final

# ── Weights ───────────────────────────────────────────────────────────────────

DIMENSION_WEIGHTS: Final[dict[str, float]] = {
    "tokenCount": 0.10,
    "codePresence": 0.30,
    "reasoningMarkers": 0.25,
    "technicalTerms": 0.25,
    "simpleIndicators": 0.05,  # subtracted - lowers the score
    "multiStepPatterns": 0.03,
    "questionComplexity": 0.02,
}

#: Score returned when no scorable text can be extracted from the request.
#: A deliberate mid-scale default; ``tests/test_scoring.py`` pins the intent.
EMPTY_TEXT_SCORE: Final[float] = 0.3

#: Default injected-context blocks stripped before scoring. Deployments
#: replace this with the tag names their own tooling injects.
DEFAULT_STRIP_PATTERNS: Final[tuple[str, ...]] = (r"<injected-context[^>]*>.*?</injected-context>",)

# Normalisation denominators: N hits = full weight for that dimension.
_TOKEN_FULL_WEIGHT_WORDS: Final[int] = 150
_CODE_HITS_FULL: Final[int] = 4
_REASONING_HITS_FULL: Final[int] = 3
_TECH_HITS_FULL: Final[int] = 3
_SIMPLE_HITS_FULL: Final[int] = 2
_MULTISTEP_HITS_FULL: Final[int] = 3
_SHORT_PROMPT_WORDS: Final[int] = 12

# ── Pattern lists ─────────────────────────────────────────────────────────────

CODE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"```"),
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bclass\s+\w+"),
    re.compile(r"\bfunction\s+\w+\s*\("),
    re.compile(r"\b(const|let|var)\s+\w+\s*="),
    re.compile(r"\breturn\b"),
    re.compile(r"\bimport\s+\w+"),
    re.compile(r"\bfrom\s+\w+\s+import\b"),
    re.compile(r"\b[\w./-]+\.(py|ts|tsx|js|jsx|go|rs|java|rb|php|sh|json|ya?ml|toml)\b", re.I),
    re.compile(r"\bfor\s+\w+\s+in\s+"),
    re.compile(r"\bif\b.{1,40}\belse\b"),
    re.compile(r"[{};\[\]]{3,}"),  # bracket runs read as code
    re.compile(
        r"\b(python|javascript|typescript|java|rust|golang|bash|sql|html|css|c\+\+|ruby|php)\b",
        re.I,
    ),
    re.compile(
        r"\b(algorithm|recursion|pointer|heap|stack|queue|linked.?list|binary.?tree|hash.?map)\b",
        re.I,
    ),
    re.compile(r"->|=>|:=|\|\||&&"),
)

REASONING_WORDS: Final[frozenset[str]] = frozenset(
    [
        # analytical / evaluative verbs
        "analyze", "analyse", "compare", "contrast", "evaluate", "justify",
        "prove", "proof", "demonstrate", "derive", "argue", "critique", "assess",
        "debate", "infer", "deduce", "synthesize", "recommend", "design",
        "architect", "predict", "diagnose", "investigate", "validate",
        # multi-part / strategic framing
        "strategy", "tradeoff", "trade-off", "pros and cons",
        "advantages", "disadvantages", "nuanced", "consider",
        "what if", "should we", "would you", "how should",
        "relationship between", "impact of", "role of", "effect of",
        "implications", "consequences", "hypothesis", "theory", "evidence",
        "reasoning", "logic",
        # theoretical / proof terms
        "paradox", "contradiction", "counterexample", "undecidable", "decidable",
        "axiom", "postulate", "reduction", "diagonalization",
        # deep analytical connectors
        "causal", "causality", "correlation", "mechanism", "underlying",
        "limitation", "assumption", "implication", "constraint",
    ]
)  # fmt: skip

TECHNICAL_PHRASES: Final[tuple[re.Pattern[str], ...]] = (
    # CS / engineering
    re.compile(
        r"\b(algorithm|complexity|big.?o|recursion|dynamic\s+programming|np.complete|np.hard)\b",
        re.I,
    ),
    re.compile(
        r"\b(machine\s+learning|neural\s+network|deep\s+learning|llm|transformer|embedding)\b",
        re.I,
    ),
    re.compile(
        r"\b(api|rest|graphql|websocket|microservice|kubernetes|docker|container|protocol)\b",
        re.I,
    ),
    re.compile(
        r"\b(async|concurrency|parallelism|thread|mutex|deadlock|race\s+condition|semaphore)\b",
        re.I,
    ),
    re.compile(r"\b(database|sql|nosql|index|query|schema|transaction|acid|replication)\b", re.I),
    re.compile(
        r"\b(optimization|scalability|latency|throughput|bottleneck|profiling|performance)\b",
        re.I,
    ),
    re.compile(
        r"\b(tcp|http|tls|dns|ip\s+address|handshake|packet|socket|network|bandwidth)\b", re.I
    ),
    re.compile(
        r"\b(computability|decidability|turing|halting|formal\s+language|automata|grammar)\b",
        re.I,
    ),
    # math / science
    re.compile(
        r"\b(derivative|integral|matrix|vector|eigenvalue|probability|statistics|regression)\b",
        re.I,
    ),
    re.compile(
        r"\b(quantum|entropy|relativity|thermodynamics|differential\s+equation|topology)\b",
        re.I,
    ),
    re.compile(
        r"\b(polynomial|logarithm|fourier|gradient\s+descent|convergence|divergence)\b", re.I
    ),
    # economics / finance / other domains
    re.compile(
        r"\b(interest\s+rate|inflation|fiscal|monetary|gdp|elasticity|supply|demand|equilibrium)\b",
        re.I,
    ),
    re.compile(
        r"\b(pharmacokinetics|litigation|amortization|valuation|taxonomy|epistemology)\b", re.I
    ),
    re.compile(
        r"\b(distributed\s+system|blockchain|cryptography|zero.?knowledge|consensus)\b", re.I
    ),
    re.compile(r"\b(architecture|infrastructure|methodology|paradigm|framework|tradeoff)\b", re.I),
)

SIMPLE_PHRASES: Final[tuple[str, ...]] = (
    "what is ", "who is ", "who was ", "when did ", "where is ",
    "define ", "what are ", "name the ", "list the ", "how many ",
    "what year ", "true or false", "yes or no", "spell ", "translate ",
    "capital of", "what color", "what time",
)  # fmt: skip

MULTISTEP_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bstep[- ]by[- ]step\b", re.I),
    re.compile(r"\bfirst\b.{1,60}\bthen\b", re.I),
    re.compile(r"\bfinally\b", re.I),
    re.compile(r"^\s*\d+[\.\)]\s", re.MULTILINE),
    re.compile(r"\b(phase|part|stage|step)\s+\d\b", re.I),
    re.compile(r"\b(workflow|pipeline|process|sequence|procedure)\b", re.I),
)

COMPLEX_WH: Final[re.Pattern[str]] = re.compile(
    r"\b(why|how|what if|should|could|would|might)\b", re.I
)


# ── Text extraction ───────────────────────────────────────────────────────────


def compile_strip_patterns(patterns: Iterable[str] | None) -> list[re.Pattern[str]]:
    """Compile ``strip_patterns`` (regex strings) with DOTALL so blocks may span lines."""
    if patterns is None:
        patterns = DEFAULT_STRIP_PATTERNS
    return [re.compile(p, re.DOTALL) for p in patterns]


def extract_text(
    messages: Sequence[Any],
    strip_patterns: Sequence[re.Pattern[str]] | None = None,
) -> str:
    """Return the last user message's text, with injected blocks stripped.

    Rules, in order:

    1. Only ``role == "user"`` messages count; the last one wins.
    2. String content is used as-is. List content contributes only its
       ``{"type": "text"}`` blocks, joined by a space - image-only or
       tool-result-only turns therefore extract as empty.
    3. Each compiled pattern in ``strip_patterns`` is removed.
    4. If stripping removed everything, the unstripped text is returned, so a
       message consisting solely of injected blocks is still scored on its
       own content rather than falling into the empty branch.
    """
    compiled = compile_strip_patterns(None) if strip_patterns is None else strip_patterns
    last_user_content = ""
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content") or ""
        if isinstance(content, str):
            last_user_content = content
        elif isinstance(content, list):
            parts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            last_user_content = " ".join(parts)
    stripped = last_user_content
    for pattern in compiled:
        stripped = pattern.sub("", stripped)
    stripped = stripped.strip()
    return stripped if stripped else last_user_content.strip()


# ── Scoring ───────────────────────────────────────────────────────────────────


def score_breakdown(
    messages: Sequence[Any],
    *,
    strip_patterns: Sequence[re.Pattern[str]] | None = None,
    weights: dict[str, float] | None = None,
    empty_text_score: float = EMPTY_TEXT_SCORE,
) -> dict[str, float]:
    """Return every dimension's normalised value plus the combined ``score``.

    Keys are the seven ``DIMENSION_WEIGHTS`` names (each in ``[0, 1]``),
    ``word_count``, ``empty`` (1.0 when the empty branch fired) and ``score``.
    """
    w = DIMENSION_WEIGHTS if weights is None else weights
    text = extract_text(messages, strip_patterns)
    if not text:
        return {
            "tokenCount": 0.0,
            "codePresence": 0.0,
            "reasoningMarkers": 0.0,
            "technicalTerms": 0.0,
            "simpleIndicators": 0.0,
            "multiStepPatterns": 0.0,
            "questionComplexity": 0.0,
            "word_count": 0.0,
            "empty": 1.0,
            "score": empty_text_score,
        }

    lower = text.lower()
    words = lower.split()
    word_count = len(words)

    # 1. tokenCount - longer prompts correlate with complexity; 150 words = full weight.
    token_score = min(1.0, word_count / _TOKEN_FULL_WEIGHT_WORDS)

    # 2. codePresence - four pattern hits = full weight.
    code_hits = sum(1 for p in CODE_PATTERNS if p.search(text))
    code_score = min(1.0, code_hits / _CODE_HITS_FULL)

    # 3. reasoningMarkers - three words = full weight (sensitive for short prompts).
    reasoning_hits = sum(1 for word in REASONING_WORDS if word in lower)
    reasoning_score = min(1.0, reasoning_hits / _REASONING_HITS_FULL)

    # 4. technicalTerms - three domain-pattern hits = full weight.
    tech_hits = sum(1 for p in TECHNICAL_PHRASES if p.search(text))
    tech_score = min(1.0, tech_hits / _TECH_HITS_FULL)

    # 5. simpleIndicators (negative) - short single-answer questions get the full penalty.
    simple_hits = sum(1 for phrase in SIMPLE_PHRASES if phrase in lower)
    simple_penalty = min(1.0, simple_hits / _SIMPLE_HITS_FULL)
    if word_count <= _SHORT_PROMPT_WORDS and simple_hits > 0:
        simple_penalty = 1.0

    # 6. multiStepPatterns - three hits = full weight.
    multistep_hits = sum(1 for p in MULTISTEP_PATTERNS if p.search(text))
    multistep_score = min(1.0, multistep_hits / _MULTISTEP_HITS_FULL)

    # 7. questionComplexity - question marks and open-ended wh-words.
    question_marks = lower.count("?")
    complex_wh_count = len(COMPLEX_WH.findall(text))
    question_score = min(1.0, (question_marks * 0.15) + (complex_wh_count * 0.1))

    score = (
        w["tokenCount"] * token_score
        + w["codePresence"] * code_score
        + w["reasoningMarkers"] * reasoning_score
        + w["technicalTerms"] * tech_score
        - w["simpleIndicators"] * simple_penalty
        + w["multiStepPatterns"] * multistep_score
        + w["questionComplexity"] * question_score
    )
    return {
        "tokenCount": token_score,
        "codePresence": code_score,
        "reasoningMarkers": reasoning_score,
        "technicalTerms": tech_score,
        "simpleIndicators": simple_penalty,
        "multiStepPatterns": multistep_score,
        "questionComplexity": question_score,
        "word_count": float(word_count),
        "empty": 0.0,
        "score": max(0.0, min(1.0, score)),
    }


def score_complexity(
    messages: Sequence[Any],
    *,
    strip_patterns: Sequence[re.Pattern[str]] | None = None,
    weights: dict[str, float] | None = None,
    empty_text_score: float = EMPTY_TEXT_SCORE,
) -> float:
    """Return a complexity score in ``[0.0, 1.0]`` for a chat request's messages."""
    return score_breakdown(
        messages,
        strip_patterns=strip_patterns,
        weights=weights,
        empty_text_score=empty_text_score,
    )["score"]
