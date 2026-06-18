"""Tests for the Prometheus text parser."""

from __future__ import annotations

from llmtop.adapters.prom import parse_prometheus, prom_value
from llmtop.adapters.base import derive_rate

SAMPLE = """\
# HELP vllm:generation_tokens_total Number of generation tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="m"} 100.0
vllm:generation_tokens_total{engine="1",model_name="m"} 50.0
vllm:num_requests_running{model_name="m"} 2.0
# a comment
vllm:kv_cache_usage_perc{model_name="m"} 0.5
"""


def test_strips_labels_and_sums_across_label_sets():
    parsed = parse_prometheus(SAMPLE)
    # Two label sets summed.
    assert parsed["vllm:generation_tokens_total"] == 150.0
    assert parsed["vllm:num_requests_running"] == 2.0
    assert parsed["vllm:kv_cache_usage_perc"] == 0.5


def test_prom_value_default_for_missing():
    parsed = parse_prometheus(SAMPLE)
    assert prom_value(parsed, "does:not:exist") is None
    assert prom_value(parsed, "does:not:exist", default=0.0) == 0.0


def test_ignores_comments_and_blank_lines():
    parsed = parse_prometheus("# only a comment\n\n   \n")
    assert parsed == {}


def test_parses_real_vllm_fixture():
    from tests.conftest import load_text

    parsed = parse_prometheus(load_text("vllm_8088_metrics.txt"))
    # These series exist in the recorded fixture.
    assert "vllm:generation_tokens_total" in parsed
    assert "vllm:kv_cache_usage_perc" in parsed
    assert "vllm:num_requests_running" in parsed


def test_derive_rate_basic_and_guards():
    assert derive_rate(200, 100, 2.0) == 50.0
    # Missing inputs -> None
    assert derive_rate(None, 100, 2.0) is None
    assert derive_rate(200, None, 2.0) is None
    # Non-positive dt -> None
    assert derive_rate(200, 100, 0) is None
    # Counter reset (went backwards) -> None, not a negative spike
    assert derive_rate(50, 100, 2.0) is None
