"""Unit tests for the price book resolution + per-modality cost."""

import pytest

from cogno_meter import Modality, PriceBook, UsageRecord


def test_exact_match():
    book = PriceBook.default()
    # gpt-4.1-mini: input 0.40, output 1.60 (USD/1M)
    cost = book.llm_cost_usd("openai:gpt-4.1-mini", 1_000_000, 500_000)
    assert cost == pytest.approx(0.40 + 0.80)


def test_fuzzy_prefix_match_handles_version_suffix():
    book = PriceBook.default()
    # 'openai:gpt-4o-mini-2024-07-18' resolves to 'openai:gpt-4o-mini'
    cost = book.llm_cost_usd("openai:gpt-4o-mini-2024-07-18", 1_000_000, 0)
    assert cost == pytest.approx(0.15)


def test_provider_default_for_self_hosted():
    book = PriceBook.default()
    # ollama:_default → 0 (self-hosted)
    assert book.llm_cost_usd("ollama:mistral:latest", 5_000_000, 2_000_000) == 0.0


def test_unknown_model_falls_to_global_default_zero():
    book = PriceBook.default()
    assert book.llm_cost_usd("madeup:model", 1_000_000, 1_000_000) == 0.0


def test_tts_cost_per_million_chars():
    book = PriceBook.default()
    # openai:tts-1 = 15.00 USD / 1M chars
    assert book.tts_cost_usd("openai:tts-1", 100_000) == pytest.approx(1.5)


def test_stt_cost_per_minute():
    book = PriceBook.default()
    assert book.stt_cost_usd("openai:whisper-1", 10) == pytest.approx(0.06)


def test_local_audio_is_free():
    book = PriceBook.default()
    assert book.tts_cost_usd("local:kokoro", 1_000_000) == 0.0
    assert book.stt_cost_usd("local:faster-whisper", 100) == 0.0


def test_billable_tokens_llm_direct():
    book = PriceBook.default()
    rec = UsageRecord(modality=Modality.LLM, model="ollama:mistral",
                      tokens_in=1000, tokens_out=200)
    assert book.billable_tokens(rec) == 1200


def test_billable_tokens_audio_uses_char_multiplier():
    book = PriceBook.default()  # audio_multiplier=2.0
    rec = UsageRecord(modality=Modality.TTS, model="local:kokoro", chars=50_000)
    assert book.billable_tokens(rec) == 100_000


def test_embedding_cost_per_million_tokens():
    book = PriceBook.default()
    assert book.embedding_cost_usd("openai:text-embedding-3-small", 2_000_000) == pytest.approx(0.04)


def test_usage_cost_dispatch_by_modality():
    book = PriceBook.default()
    llm = UsageRecord(Modality.LLM, "openai:gpt-4o-mini", tokens_in=1_000_000, tokens_out=0)
    emb = UsageRecord(Modality.EMBEDDING, "openai:text-embedding-3-small", tokens_in=1_000_000)
    stt = UsageRecord(Modality.STT, "openai:whisper-1", minutes=10)
    tts = UsageRecord(Modality.TTS, "openai:tts-1", chars=100_000)
    assert book.usage_cost_usd(llm) == pytest.approx(0.15)
    assert book.usage_cost_usd(emb) == pytest.approx(0.02)
    assert book.usage_cost_usd(stt) == pytest.approx(0.06)
    assert book.usage_cost_usd(tts) == pytest.approx(1.5)


def test_from_mapping_overrides_rates_and_multiplier():
    book = PriceBook.from_mapping({
        "llm": {"x:y": {"input": 1.0, "output": 2.0}},
        "tts": {"_default": 0.0},
        "usd_brl_rate": 6.0,
        "audio_multiplier": 3.0,
    })
    assert book.usd_brl_rate == 6.0
    assert book.audio_multiplier == 3.0
    assert book.llm_cost_usd("x:y", 1_000_000, 1_000_000) == pytest.approx(3.0)
    rec = UsageRecord(modality=Modality.STT, model="local:w", chars=10)
    assert book.billable_tokens(rec) == 30
