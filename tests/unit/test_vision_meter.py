"""Unit tests for vision metering parity and cost calculation (Tasks C, D)."""

import pytest
from cogno_meter.pricing import PriceBook
from cogno_meter.types import Modality, UsageRecord


# ── Test 4: Meter Parity (cost 0.0 iff billable_tokens 0 for unknown modalities) ─────────
def test_4_meter_parity_zero_cost_implies_zero_billable_tokens_for_unknown_modalities():
    pb = PriceBook.default()

    # An unknown/unsupported modality string (e.g. "unknown", "custom_modality")
    rec = UsageRecord(
        modality="unknown",  # type: ignore
        model="some-model",
        tokens_in=100,
        tokens_out=50,
        chars=500,
    )

    cost = pb.usage_cost_usd(rec)
    tokens = pb.billable_tokens(rec)

    # Parity assertion: if usage_cost_usd returns zero for unknown modality, billable_tokens must also be zero
    assert cost == 0.0
    assert tokens == 0, f"Expected 0 billable tokens for unknown modality, got {tokens}"


# ── Test 5: Vision Call Cost > 0 (modality=LLM, stage=vision) ───────────────────────────
def test_5_vision_call_with_llm_modality_costs_more_than_zero():
    pb = PriceBook.default()

    # Vision calls are metered with modality = LLM, stage = vision, model catalogued (e.g. openai:gpt-4o)
    rec = UsageRecord(
        modality=Modality.LLM,
        model="openai:gpt-4o",
        tokens_in=500,
        tokens_out=100,
        chars=200,
    )

    cost = pb.usage_cost_usd(rec)
    tokens = pb.billable_tokens(rec)

    # Vision call must cost > $0 and produce > 0 billable tokens
    assert cost > 0.0, f"Expected vision call cost > 0, got {cost}"
    assert tokens == 600, f"Expected 600 billable tokens, got {tokens}"
