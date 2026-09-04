"""Scoring-stage tests: written from the scorer's stated behaviour, not from any
prior test suite. Every expected number is derived in a comment."""

from __future__ import annotations

import re

import pytest

from complexity_router.scoring import (
    DEFAULT_STRIP_PATTERNS,
    DIMENSION_WEIGHTS,
    EMPTY_TEXT_SCORE,
    compile_strip_patterns,
    extract_text,
    score_breakdown,
    score_complexity,
)


def user(text: str | list) -> list[dict]:
    return [{"role": "user", "content": text}]


# ── weights ───────────────────────────────────────────────────────────────────


def test_weights_are_the_published_seven() -> None:
    assert DIMENSION_WEIGHTS == {
        "tokenCount": 0.10,
        "codePresence": 0.30,
        "reasoningMarkers": 0.25,
        "technicalTerms": 0.25,
        "simpleIndicators": 0.05,
        "multiStepPatterns": 0.03,
        "questionComplexity": 0.02,
    }


def test_positive_weights_sum_to_achievable_maximum() -> None:
    positive = sum(v for k, v in DIMENSION_WEIGHTS.items() if k != "simpleIndicators")
    assert positive == pytest.approx(0.95)


# ── extract_text ──────────────────────────────────────────────────────────────


def test_extract_uses_last_user_message_only() -> None:
    msgs = [
        {"role": "system", "content": "You are a distributed systems architect."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    assert extract_text(msgs) == "second"


def test_extract_joins_only_text_blocks_from_list_content() -> None:
    content = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "describe"},
        {"type": "text", "text": "this"},
    ]
    assert extract_text(user(content)) == "describe this"


def test_extract_image_only_turn_is_empty() -> None:
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    assert extract_text(user(content)) == ""


def test_extract_strips_default_injected_block() -> None:
    text = "<injected-context source='x'>\nkubernetes latency database\n</injected-context>\nhi"
    assert extract_text(user(text)) == "hi"


def test_extract_falls_back_to_unstripped_when_stripping_empties_message() -> None:
    text = "<injected-context>only injected content here</injected-context>"
    assert extract_text(user(text)) == text


def test_extract_honours_custom_strip_patterns() -> None:
    patterns = compile_strip_patterns([r"<memo>.*?</memo>"])
    text = "<memo>algorithm database</memo> hello"
    assert extract_text(user(text), patterns) == "hello"
    # the default pattern is not applied when a custom list is passed
    default_block = "<injected-context>x</injected-context> hello"
    assert extract_text(user(default_block), patterns) == default_block


def test_compile_strip_patterns_defaults_and_dotall() -> None:
    compiled = compile_strip_patterns(None)
    assert [p.pattern for p in compiled] == list(DEFAULT_STRIP_PATTERNS)
    assert all(p.flags & re.DOTALL for p in compiled)


def test_extract_ignores_non_dict_messages_and_none_content() -> None:
    msgs = ["garbage", {"role": "user", "content": None}, {"role": "user"}]
    assert extract_text(msgs) == ""


# ── empty branch ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "system", "content": "no user turn"}],
        user(""),
        user("   \n\t "),
        user([{"type": "image_url", "image_url": {"url": "data:,"}}]),
    ],
)
def test_empty_text_returns_the_pinned_default(messages: list) -> None:
    # The intent pinned here: no scorable text -> the mid-scale default, not 0.
    assert score_complexity(messages) == EMPTY_TEXT_SCORE == 0.3
    assert score_breakdown(messages)["empty"] == 1.0


def test_empty_text_score_is_configurable() -> None:
    assert score_complexity(user(""), empty_text_score=0.0) == 0.0


# ── worked examples, every term computed ──────────────────────────────────────


def test_single_reasoning_word_scores_0_0840() -> None:
    # "analyze": 1 word -> tokenCount = 1/150; reasoningMarkers = 1/3; nothing else.
    # 0.10 * (1/150) + 0.25 * (1/3) = 0.000667 + 0.083333 = 0.0840
    score = score_complexity(user("analyze"))
    assert score == pytest.approx(0.10 / 150 + 0.25 / 3, abs=1e-6)
    assert score == pytest.approx(0.0840, abs=5e-5)


def test_four_code_pattern_hits_reach_at_least_0_30() -> None:
    # A fenced block, a def, a return and an import: four distinct patterns.
    text = "```\ndef f(x):\n    return x\n```\nimport os"
    b = score_breakdown(user(text))
    assert b["codePresence"] == 1.0
    assert b["score"] >= 0.30


def test_short_simple_question_gets_full_penalty() -> None:
    # "what is the capital of France?" -> 6 words <= 12 and a simple phrase hit:
    # simple_penalty = 1.0 regardless of hit count. Question mark adds 0.02*0.15.
    b = score_breakdown(user("what is the capital of France?"))
    assert b["simpleIndicators"] == 1.0
    expected = 0.10 * (6 / 150) - 0.05 * 1.0 + 0.02 * 0.15
    assert b["score"] == pytest.approx(max(0.0, expected), abs=1e-6)


def test_simple_penalty_scales_with_hits_on_longer_prompts() -> None:
    # Over 12 words, so the short-prompt override does not fire; one hit -> 1/2.
    text = "define the term that people use when they talk about a very long sentence here"
    assert len(text.split()) > 12
    b = score_breakdown(user(text))
    assert b["simpleIndicators"] == 0.5


def test_token_count_saturates_at_150_words() -> None:
    text = " ".join(["word"] * 300)
    assert score_breakdown(user(text))["tokenCount"] == 1.0
    assert score_breakdown(user(" ".join(["word"] * 75)))["tokenCount"] == 0.5


def test_technical_terms_saturate_at_three_pattern_hits() -> None:
    # kubernetes (api group), latency (perf group), database (db group): 3 groups.
    b = score_breakdown(user("kubernetes latency database"))
    assert b["technicalTerms"] == 1.0
    # A second word from the same group does not add a hit.
    b2 = score_breakdown(user("kubernetes docker"))
    assert b2["technicalTerms"] == pytest.approx(1 / 3)


def test_multistep_and_question_dimensions() -> None:
    text = "Explain step by step why this pipeline fails, and finally how to fix it?"
    b = score_breakdown(user(text))
    # step-by-step, "finally", "pipeline" -> 3 hits -> 1.0
    assert b["multiStepPatterns"] == 1.0
    # one '?' (0.15) + why, how (2 * 0.1) = 0.35
    assert b["questionComplexity"] == pytest.approx(0.35)


def test_question_complexity_caps_at_one() -> None:
    text = "why? how? should? could? would? might? what if?"
    assert score_breakdown(user(text))["questionComplexity"] == 1.0


def test_score_is_clamped_to_unit_interval() -> None:
    assert 0.0 <= score_complexity(user("what is ")) <= 1.0
    dense = (
        "```\ndef f():\n    return 1\n```\nimport os\n"
        "analyze compare evaluate kubernetes latency database "
        "step by step first then finally why? how?"
    )
    assert 0.0 <= score_complexity(user(dense)) <= 1.0


def test_custom_weights_are_applied() -> None:
    weights = dict(DIMENSION_WEIGHTS)
    weights["reasoningMarkers"] = 0.0
    assert score_complexity(user("analyze"), weights=weights) == pytest.approx(0.10 / 150)


def test_breakdown_keys_are_complete() -> None:
    b = score_breakdown(user("hello"))
    assert set(b) == set(DIMENSION_WEIGHTS) | {"word_count", "empty", "score"}
    assert b["word_count"] == 1.0
    assert b["empty"] == 0.0


def test_scoring_is_deterministic() -> None:
    text = "Compare the tradeoffs of TCP versus UDP for a real-time pipeline."
    assert score_complexity(user(text)) == score_complexity(user(text))
