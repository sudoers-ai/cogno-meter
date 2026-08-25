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


def test_retired_grok_slugs_are_priced_as_what_xai_actually_serves():
    """A retired xAI slug is REDIRECTED to grok-4.3 and billed at grok-4.3 rates, so the meter
    must charge 4.3 for it — and the key must stay in the book. Dropping it does not fail loudly:
    'grok:grok-4' prefix-matches nothing ('grok-4' is not a prefix of 'grok-4.3'), so it would
    resolve to _default = 0 and report paid traffic as free."""
    book = PriceBook.default()
    live = book.llm_cost_usd("grok:grok-4.3", 1_000_000, 1_000_000)
    for retired in ("grok:grok-3", "grok:grok-3-mini", "grok:grok-4", "grok:grok-4.1-fast"):
        assert book.llm_cost_usd(retired, 1_000_000, 1_000_000) == pytest.approx(live)
    assert live == pytest.approx(1.25 + 2.50)
    # the flagship is NOT collapsed into that rate
    assert book.llm_cost_usd("grok:grok-4.5", 1_000_000, 1_000_000) == pytest.approx(2.00 + 6.00)


def test_grok_420_snapshot_variants_resolve_by_prefix():
    book = PriceBook.default()
    for variant in ("grok:grok-4.20-0309-reasoning", "grok:grok-4.20-0309-non-reasoning",
                    "grok:grok-4.20-multi-agent-0309"):
        assert book.llm_cost_usd(variant, 1_000_000, 0) == pytest.approx(1.25)


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


def test_bare_model_name_resolves_to_prefixed_rate():
    # the ledger stores the backend's bare name ('gpt-4o'), but rates are keyed 'openai:gpt-4o'
    book = PriceBook.default()
    assert book.llm_cost_usd("gpt-4o", 1_000_000, 1_000_000) == 2.50 + 10.00
    assert book.llm_cost_usd("gpt-4o-mini", 1_000_000, 0) == 0.15
    # bare + versioned → longest bare prefix wins (not the shorter 'gpt-4o')
    assert book.llm_cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == 0.15
    # a bare local model has no rate → self-hosted 0 (correct, not a mismatch)
    assert book.llm_cost_usd("qwen3:8b", 1_000_000, 1_000_000) == 0.0
    # prefixed names still resolve (back-compat)
    assert book.llm_cost_usd("openai:gpt-4o", 1_000_000, 0) == 2.50


def test_prefix_resolution_is_longest_first_not_insertion_order():
    # a short key that string-prefixes a longer one must NOT steal the longer model's rate,
    # regardless of dict-insertion order (the step-2 prefix loop was insertion-ordered).
    book = PriceBook.from_mapping({"llm": {
        "openai:gpt-5": {"input": 1.0, "output": 1.0},          # inserted FIRST, shorter
        "openai:gpt-5.5-pro": {"input": 30.0, "output": 180.0},  # inserted after, longer
    }})
    # a dated build of the flagship resolves to gpt-5.5-pro, not gpt-5
    cost = book.llm_cost_usd("openai:gpt-5.5-pro-2025-01-01", 1_000_000, 1_000_000)
    assert cost == pytest.approx(210.0)   # not 2.0 (the gpt-5 rate)


def test_per_provider_default_beats_global_default():
    book = PriceBook.from_mapping({"llm": {
        "openai:_default": {"input": 10.0, "output": 30.0},
        "_default": {"input": 0.0, "output": 0.0},
    }})
    # an un-catalogued openai model hits openai:_default (40.0), not the global 0
    assert book.llm_cost_usd("openai:gpt-brand-new", 1_000_000, 1_000_000) == pytest.approx(40.0)
    # a provider with no per-provider default still falls to the global 0
    assert book.llm_cost_usd("groq:whatever", 1_000_000, 1_000_000) == 0.0


def test_none_token_fields_do_not_crash_billing():
    book = PriceBook.default()
    rec = UsageRecord(modality=Modality.LLM, model="openai:gpt-4o", tokens_in=None, tokens_out=500)
    assert book.billable_tokens(rec) == 500          # None coerced to 0, no TypeError
    assert book.usage_cost_usd(rec) == pytest.approx(500 / 1_000_000 * 10.0)


def test_embedding_billable_matches_cost_path():
    # billable tokens and the priced amount must interpret the same embedding record identically
    book = PriceBook.default()
    rec = UsageRecord(modality=Modality.EMBEDDING, model="openai:text-embedding-3-small",
                      tokens_in=1_000_000, tokens_out=1_000_000)
    assert book.billable_tokens(rec) == 1_000_000    # tokens_in or tokens_out (mirrors cost)


# ── an uncatalogued cloud model must never meter as free ──────────────────────────────────

@pytest.mark.parametrize("model", [
    "openai:gpt-9-turbo", "anthropic:claude-opus-9", "gemini:gemini-9-pro",
    "grok:grok-9", "groq:llama-9-70b", "deepseek:deepseek-v9",
])  # groq/deepseek have NO rates at all — cogno-synapse ships backends for both
def test_an_uncatalogued_cloud_model_is_not_free(model):
    """Measured 2026-08-05: every cloud provider priced an unknown model at 0.00.

    The grok block in the price book already names this failure — "a request to a retired
    slug ... would fall through to _default = 0 and report paid traffic as FREE" — and keeps
    dead keys around to avoid it. It is not specific to xAI: any provider shipping a model
    nobody has catalogued yet bills nothing until someone remembers to add it. A meter that
    reports zero for paid traffic is worse than one that reports approximately.
    """
    book = PriceBook.default()
    rate = book._resolve(book.rates["llm"], model)
    assert rate is not None and rate["output"] > 0.0, f"{model} metered free"


def test_the_floor_is_the_providers_own_cheapest_rate():
    """Derived from the table, never guessed, so it stays right as the table changes. Cheapest
    and not flagship on purpose: this feeds BudgetGuard as well as billing, and the smallest
    non-zero number cannot wrongly block a tenant."""
    book = PriceBook.default()
    llm = book.rates["llm"]
    openai_rates = [r for k, r in llm.items()
                    if k.startswith("openai:") and not k.endswith(":_default")]
    cheapest = min(openai_rates, key=lambda r: (r["output"], r["input"]))
    assert book._resolve(llm, "openai:gpt-9-turbo") == cheapest


def test_a_local_model_with_a_TAG_is_still_free():
    """An Ollama model is natively named model:tag — qwen3:8b, mistral:latest. "Has a
    colon" therefore cannot mean "has a provider", and a first version of the floor read it
    that way and started CHARGING for models on the user's own hardware. The existing
    test_bare_model_name_resolves_to_prefixed_rate caught it; this pins it directly."""
    book = PriceBook.default()
    for local in ("qwen3:8b", "mistral:latest", "nomic-embed-text:latest"):
        assert book.llm_cost_usd(local, 1_000_000, 1_000_000) == 0.0, local


def test_an_unknown_scheme_is_not_charged():
    """Only a KNOWN cloud provider gets the floor; anything else resolves as before."""
    assert PriceBook.default().llm_cost_usd("madeup:model", 1_000_000, 1_000_000) == 0.0


def test_self_hosted_stays_free():
    """Contraprova: ollama carries an explicit ``ollama:_default = 0`` and keeps it — the floor
    must never start charging for a model running on the user's own hardware."""
    book = PriceBook.default()
    assert book._resolve(book.rates["llm"], "ollama:anything-at-all")["output"] == 0.0


def test_a_catalogued_model_is_unaffected():
    book = PriceBook.default()
    assert book._resolve(book.rates["llm"], "openai:gpt-4o-mini")["output"] == 0.6


def test_every_tts_model_a_host_can_offer_has_a_stated_NON_ZERO_rate():
    """An uncatalogued model does not fail — it lands on the cheapest known floor and logs
    `rate_uncatalogued` on EVERY call. Same number, unstated, plus a warning per message.

    **Presence alone is not the assertion**, and a review measured why: with only a key check,
    `15.00 → 0.0` passed 57/57 — metering paid traffic as free, which is precisely the
    regression PR #13 exists to prevent. The rate has to be positive.

    It still does not check the number is RIGHT (`gpt-4o-mini-tts` is billed by audio-output
    token while this book is per character, and its entry says so). It checks the number is
    DECLARED and not zero, which is what makes it reviewable at all."""
    from cogno_meter.pricing import DEFAULT_RATES

    offered = {"openai:tts-1", "openai:tts-1-hd", "openai:gpt-4o-mini-tts"}
    rates = DEFAULT_RATES["tts"]
    missing = offered - set(rates)
    assert not missing, f"tts models with no stated rate: {sorted(missing)}"
    free = {m for m in offered if not rates[m] > 0}
    assert not free, f"paid cloud tts metered as FREE: {sorted(free)}"
