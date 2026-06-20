"""
Edge-case coverage for the billing math — boundaries the happy-path tests skip:
exact-allowance boundaries, a zero/full allowance, empty periods, audio rounding,
partial/empty price books, the embedding token fallback, and the fuzzy-prefix
resolution ordering (a guard against reordering DEFAULT_RATES).
"""

import pytest

from cogno_meter import (
    Modality,
    PriceBook,
    Plan,
    UsageRecord,
    compute_bill,
    meter,
)

PLAN = Plan(name="pro", monthly_token_limit=1_000_000, overage_price=0.05, base_price=20.0)
BOOK = PriceBook.default()


def _llm(tin: int, tout: int = 0, model: str = "ollama:mistral") -> UsageRecord:
    return UsageRecord(modality=Modality.LLM, model=model, tokens_in=tin, tokens_out=tout)


# ── allowance boundaries ─────────────────────────────────────────────────
def test_exactly_at_allowance_has_no_overage():
    bill = compute_bill(plan=PLAN, billable_tokens=1_000_000, provider_cost_usd=0.0, book=BOOK)
    assert bill.overage_tokens == 0
    assert bill.overage_cost == 0.0
    assert bill.total == 20.0


def test_one_token_over_allowance():
    bill = compute_bill(plan=PLAN, billable_tokens=1_000_001, provider_cost_usd=0.0, book=BOOK)
    assert bill.overage_tokens == 1
    # 1 token / 1e6 * 0.05 rounds to 0.0 at 6dp, but the token count is exact
    assert bill.overage_cost == round((1 / 1_000_000) * 0.05, 6)


def test_zero_allowance_charges_all_tokens():
    plan = Plan(name="payg", monthly_token_limit=0, overage_price=0.05, base_price=0.0)
    bill = compute_bill(plan=plan, billable_tokens=2_000_000, provider_cost_usd=0.0, book=BOOK)
    assert bill.overage_tokens == 2_000_000
    assert bill.overage_cost == pytest.approx(0.10)
    assert bill.total == pytest.approx(0.10)


# ── empty / trivial periods ──────────────────────────────────────────────
def test_meter_empty_period_is_base_price_only():
    bill = meter([], plan=PLAN, book=BOOK)
    assert bill.total_tokens == 0
    assert bill.overage_tokens == 0
    assert bill.total == 20.0
    assert bill.provider_cost_usd == 0.0 and bill.provider_cost_brl == 0.0


def test_meter_aggregates_multiple_records():
    bill = meter([_llm(400_000, 200_000), _llm(300_000, 300_000)], plan=PLAN, book=BOOK)
    assert bill.total_tokens == 1_200_000
    assert bill.overage_tokens == 200_000


# ── audio rounding (chars × multiplier) ──────────────────────────────────
def test_audio_billable_rounds_to_nearest_int():
    book = PriceBook(audio_multiplier=1.5)
    # 3 chars × 1.5 = 4.5 → banker's rounding → 4
    rec = UsageRecord(modality=Modality.TTS, model="local:kokoro", chars=3)
    assert book.billable_tokens(rec) == 4
    # 5 chars × 1.5 = 7.5 → 8
    assert book.billable_tokens(UsageRecord(modality=Modality.STT, model="x", chars=5)) == 8


def test_audio_provider_cost_uses_minutes_not_chars():
    # STT provider cost is per-minute; billable tokens are per-char. Different fields.
    rec = UsageRecord(modality=Modality.STT, model="openai:whisper-1", chars=600, minutes=10)
    assert BOOK.usage_cost_usd(rec) == pytest.approx(0.06)        # 10 min × 0.006
    assert BOOK.billable_tokens(rec) == int(round(600 * BOOK.audio_multiplier))


# ── embedding token fallback ─────────────────────────────────────────────
def test_embedding_cost_falls_back_to_tokens_out_when_in_is_zero():
    rec = UsageRecord(modality=Modality.EMBEDDING, model="openai:text-embedding-3-small",
                      tokens_in=0, tokens_out=2_000_000)
    assert BOOK.usage_cost_usd(rec) == pytest.approx(0.04)        # uses tokens_out


# ── partial / empty price books ──────────────────────────────────────────
def test_empty_rates_book_costs_zero_but_still_meters_tokens():
    book = PriceBook(rates={})
    assert book.llm_cost_usd("openai:gpt-4o", 1_000_000, 1_000_000) == 0.0
    assert book.embedding_cost_usd("openai:text-embedding-3-small", 1_000_000) == 0.0
    # tokens still count toward the allowance even with no rate table
    assert book.billable_tokens(_llm(500, 500)) == 1000


def test_from_mapping_uses_defaults_for_missing_keys():
    book = PriceBook.from_mapping({"llm": {"_default": {"input": 0.0, "output": 0.0}}})
    assert book.usd_brl_rate == pytest.approx(5.70)               # default
    assert book.audio_multiplier == pytest.approx(2.0)            # default
    # unknown top-level keys are ignored (only llm/embedding/stt/tts kept)
    book2 = PriceBook.from_mapping({"llm": {}, "garbage": {"x": 1}})
    assert "garbage" not in book2.rates


# ── fuzzy-prefix resolution ordering (guards DEFAULT_RATES order) ─────────
def test_versioned_model_resolves_to_most_specific_seed():
    # A dated nano build must resolve to the nano rate, not the broader gpt-4.1.
    nano = BOOK.llm_cost_usd("openai:gpt-4.1-nano-2025-01-01", 1_000_000, 0)
    assert nano == pytest.approx(0.10)                            # nano input rate
    mini = BOOK.llm_cost_usd("openai:gpt-4.1-mini-2025-01-01", 1_000_000, 0)
    assert mini == pytest.approx(0.40)                            # mini, not nano/4.1


def test_provider_cost_brl_uses_rate():
    book = PriceBook(usd_brl_rate=6.0)
    bill = compute_bill(plan=PLAN, billable_tokens=0, provider_cost_usd=2.0, book=book)
    assert bill.provider_cost_brl == pytest.approx(12.0)
