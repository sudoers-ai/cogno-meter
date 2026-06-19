"""Unit tests for the overage math + the two confirmed worked examples."""

import pytest

from cogno_meter import Modality, Plan, PriceBook, UsageRecord, meter

# Example plan (values are illustrative; real ones are host-owned).
BASIC = Plan(name="Básico", monthly_token_limit=500_000, overage_price=0.05, base_price=20.0)


def test_under_allowance_no_overage():
    book = PriceBook.default()
    records = [UsageRecord(Modality.LLM, "ollama:mistral", tokens_in=300_000, tokens_out=100_000)]
    bill = meter(records, plan=BASIC, book=book)
    assert bill.total_tokens == 400_000
    assert bill.overage_tokens == 0
    assert bill.overage_cost == 0.0
    assert bill.total == 20.0


def test_compute_bill_debug_log(caplog):
    """Pure function emits only DEBUG (no INFO/WARNING per call)."""
    import logging
    book = PriceBook.default()
    records = [UsageRecord(Modality.LLM, "ollama:mistral", tokens_in=600_000, tokens_out=0)]
    with caplog.at_level(logging.DEBUG, logger="cogno_meter.billing"):
        meter(records, plan=BASIC, book=book)
    debug = [r for r in caplog.records if "event=compute_bill" in r.message]
    assert debug and debug[0].levelno == logging.DEBUG
    assert "overage_tokens=100000" in debug[0].message
    # never INFO/WARNING from a pure billing function
    assert not [r for r in caplog.records if r.levelno >= logging.INFO]


def test_example_1_local_model_exceeded():
    """Local LLM (1.2M tok) + Kokoro TTS (50k chars ×2 = 100k) = 1.3M billable.
    Overage 800k → R$0.04; provider cost 0 (local); total R$20.04."""
    book = PriceBook.default()
    records = [
        UsageRecord(Modality.LLM, "ollama:mistral:latest", tokens_in=1_000_000, tokens_out=200_000),
        UsageRecord(Modality.TTS, "local:kokoro", chars=50_000),
    ]
    bill = meter(records, plan=BASIC, book=book)
    assert bill.total_tokens == 1_300_000
    assert bill.overage_tokens == 800_000
    assert bill.overage_cost == pytest.approx(0.04)
    assert bill.provider_cost_usd == 0.0
    assert bill.provider_cost_brl == 0.0
    assert bill.total == pytest.approx(20.04)


def test_example_2_external_paid_model_exceeded():
    """gpt-4.1-mini 1.0M in + 0.5M out = 1.5M billable. Overage 1.0M → R$0.05.
    Provider cost = $0.40 + $0.80 = $1.20 → R$6.84 (transparency only)."""
    book = PriceBook.default()
    records = [
        UsageRecord(Modality.LLM, "openai:gpt-4.1-mini", tokens_in=1_000_000, tokens_out=500_000),
    ]
    bill = meter(records, plan=BASIC, book=book)
    assert bill.total_tokens == 1_500_000
    assert bill.overage_tokens == 1_000_000
    assert bill.overage_cost == pytest.approx(0.05)
    assert bill.total == pytest.approx(20.05)
    # transparency: real upstream cost (client pays the provider via BYOK)
    assert bill.provider_cost_usd == pytest.approx(1.20)
    assert bill.provider_cost_brl == pytest.approx(6.84)


def test_overage_rate_is_per_plan_not_per_model():
    """The overage charge depends on the PLAN rate, not the model that ran.
    Same overage tokens on an expensive external model vs a free local one →
    identical platform charge (only provider_cost differs)."""
    book = PriceBook.default()
    local = meter([UsageRecord(Modality.LLM, "ollama:mistral", tokens_in=1_500_000, tokens_out=0)],
                  plan=BASIC, book=book)
    cloud = meter([UsageRecord(Modality.LLM, "openai:gpt-4o", tokens_in=1_500_000, tokens_out=0)],
                  plan=BASIC, book=book)
    assert local.overage_cost == cloud.overage_cost          # same plan-rate charge
    assert local.total == cloud.total
    assert local.provider_cost_usd == 0.0                    # local free
    assert cloud.provider_cost_usd > 0.0                     # cloud tracked (transparency)


def test_audio_multiplier_makes_audio_cost_more_allowance():
    """1M chars of TTS draws down 2M billable tokens (×2 multiplier)."""
    book = PriceBook.default()
    plan = Plan(name="t", monthly_token_limit=1_000_000, overage_price=0.10)
    bill = meter([UsageRecord(Modality.TTS, "local:kokoro", chars=1_000_000)], plan=plan, book=book)
    assert bill.total_tokens == 2_000_000
    assert bill.overage_tokens == 1_000_000
    assert bill.overage_cost == pytest.approx(0.10)
