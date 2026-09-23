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


# ── the provider's prompt cache (``UsageRecord.cached_tokens``) ────────────────────────────
#
# Measured live 2026-09-03: a second call with the same prefix reported 2432 cached of 2625
# prompt tokens (92.6%). The field existed in the provider payload and was read NOWHERE, so
# every EGO correction retry — 40.2% of the month's tokens — was priced as if the prompt were
# fresh. These pin the three outcomes and the one thing that must NOT move.

def test_a_cached_prompt_is_priced_at_the_models_cached_rate_not_the_input_rate():
    """The discount is the TABLE's, to the token — not a percentage this code invents."""
    book = PriceBook.default()
    # gpt-4o-mini: input 0.15, cached_input 0.075, output 0.60 (USD/1M)
    cost = book.llm_cost_usd("openai:gpt-4o-mini", 1_000_000, 100_000, cached_tokens=800_000)
    expected = (200_000 / 1e6) * 0.15 + (800_000 / 1e6) * 0.075 + (100_000 / 1e6) * 0.60
    assert cost == pytest.approx(expected)
    # …and it is strictly cheaper than pricing the same call as all-fresh.
    assert cost < book.llm_cost_usd("openai:gpt-4o-mini", 1_000_000, 100_000)


def test_the_live_measurement_prices_lower_by_exactly_the_table():
    """The turn that was actually measured: 2432 cached of 2625 prompt tokens."""
    book = PriceBook.default()
    rec = UsageRecord(Modality.LLM, "openai:gpt-4o-mini",
                      tokens_in=2625, tokens_out=100, cached_tokens=2432)
    expected = (193 / 1e6) * 0.15 + (2432 / 1e6) * 0.075 + (100 / 1e6) * 0.60
    assert book.usage_cost_usd(rec) == pytest.approx(expected)


@pytest.mark.parametrize("model", sorted(
    k for k in PriceBook.default().rates["llm"] if not k.endswith("_default")))
def test_a_record_with_no_cached_tokens_prices_exactly_as_before(model):
    """The BYTE-IDENTICAL twin, over every catalogued model: with the field absent (0), the
    result is the pre-change formula — ``tokens_in × input + tokens_out × output`` — and the
    ``cached_input`` column changes nothing at all."""
    book = PriceBook.default()
    rates = book.rates["llm"][model]
    legacy = (1_000_000 / 1e6) * rates["input"] + (250_000 / 1e6) * rates["output"]
    assert book.llm_cost_usd(model, 1_000_000, 250_000) == pytest.approx(legacy)
    assert book.usage_cost_usd(
        UsageRecord(Modality.LLM, model, tokens_in=1_000_000, tokens_out=250_000)
    ) == pytest.approx(legacy)


def test_a_model_with_no_cache_rate_is_charged_the_full_input_rate(caplog):
    """NEVER guess a discount: full price, and say so.

    The model here used to be ``gpt-5.6-luna``, on the belief that its cached rate was not
    published. It is — the page prints it — and the belief cost 8.8x on ~92% of one
    deployment's spend. The specimen is now ``gpt-5.5-pro``, whose Cached input cell really
    is a ``-``, and it is named in ``CACHE_RATE_NOT_PUBLISHED`` so the two cases cannot be
    confused again by anyone reading only this test."""
    from cogno_meter.pricing import CACHE_RATE_NOT_PUBLISHED

    book = PriceBook.default()
    assert "openai:gpt-5.5-pro" in CACHE_RATE_NOT_PUBLISHED
    assert "cached_input" not in book.rates["llm"]["openai:gpt-5.5-pro"]
    full = book.llm_cost_usd("openai:gpt-5.5-pro", 1_000_000, 0)
    with caplog.at_level("WARNING", logger="cogno_meter.pricing"):
        cached = book.llm_cost_usd("openai:gpt-5.5-pro", 1_000_000, 0, cached_tokens=900_000)
    assert cached == pytest.approx(full)
    assert "cache_rate_uncatalogued" in caplog.text
    assert "openai:gpt-5.5-pro" in caplog.text


def test_the_missing_rate_is_declared_once_per_model_not_once_per_call(caplog):
    """A per-call line on a model that runs every turn is a line nobody reads."""
    book = PriceBook.default()
    with caplog.at_level("WARNING", logger="cogno_meter.pricing"):
        for _ in range(5):
            book.llm_cost_usd("openai:gpt-5.5-pro", 1_000, 0, cached_tokens=900)
        book.llm_cost_usd("anthropic:claude-opus-4-6", 1_000, 0, cached_tokens=900)
    lines = [r for r in caplog.records if "cache_rate_uncatalogued" in r.getMessage()]
    assert len(lines) == 2
    # …and a FRESH book counts again (the dedup is per instance, so tests do not infect
    # each other and a long-lived process is not silenced by a short-lived one).
    with caplog.at_level("WARNING", logger="cogno_meter.pricing"):
        caplog.clear()
        PriceBook.default().llm_cost_usd("openai:gpt-5.5-pro", 1_000, 0, cached_tokens=900)
    assert "cache_rate_uncatalogued" in caplog.text


def test_the_cache_discount_does_not_touch_the_allowance():
    """The plan is a contract in TOKENS. What the provider cached is our saving, not the
    customer's: subtracting it here would silently rewrite every published plan."""
    book = PriceBook.default()
    plain = UsageRecord(Modality.LLM, "openai:gpt-4o-mini", tokens_in=2625, tokens_out=100)
    cached = UsageRecord(Modality.LLM, "openai:gpt-4o-mini", tokens_in=2625, tokens_out=100,
                         cached_tokens=2432)
    assert book.billable_tokens(cached) == book.billable_tokens(plain) == 2725
    # …while the cost DID move, so the test is not passing by the field being inert.
    assert book.usage_cost_usd(cached) < book.usage_cost_usd(plain)


def test_more_cached_than_prompt_tokens_never_produces_a_credit():
    """A provider payload that over-reports cached tokens must not make the fresh count
    negative — that would be a refund on our own bill."""
    book = PriceBook.default()
    cost = book.llm_cost_usd("openai:gpt-4o-mini", 1_000, 0, cached_tokens=9_999)
    assert cost == pytest.approx((1_000 / 1e6) * 0.075)
    assert book.llm_cost_usd("openai:gpt-4o-mini", 1_000, 0, cached_tokens=-5) == \
        pytest.approx((1_000 / 1e6) * 0.15)


def test_a_host_supplied_book_can_carry_its_own_cache_rates():
    """``from_mapping`` is how a deployment overrides the seed — the new column must survive
    it, or the feature only ever works for the shipped table."""
    book = PriceBook.from_mapping({
        "llm": {"acme:model": {"input": 10.0, "output": 20.0, "cached_input": 1.0}},
        "embedding": {"_default": 0.0}, "stt": {"_default": 0.0}, "tts": {"_default": 0.0},
    })
    assert book.llm_cost_usd("acme:model", 1_000_000, 0, cached_tokens=900_000) == \
        pytest.approx((100_000 / 1e6) * 10.0 + (900_000 / 1e6) * 1.0)


def test_every_seeded_cache_rate_is_cheaper_than_the_models_own_input_rate():
    """A ``cached_input`` at or above ``input`` is a typo that would OVER-charge; a negative
    one is a credit. Cheap, and it guards the whole column as the table grows."""
    for model, rates in PriceBook.default().rates["llm"].items():
        if not isinstance(rates, dict) or "cached_input" not in rates:
            continue
        assert 0.0 < rates["cached_input"] < rates["input"], model


# ── the entry that was wrong, and the gate that keeps the next one from being silent ───────
#
# One table row, on the model that is ~92% of one deployment's provider spend: gpt-5.6-luna
# shipped at input 1.00 / output 6.00 and no cache rate, against a published Standard,
# short-context 0.20 / 1.20 / 0.02. Nobody was over-BILLED — the invoice and the monthly
# allowance are both in TOKENS — but the daily BRL ceiling is in money, so it degraded and
# refused service at 22% of the real budget. The defect did not take money; it took service.

_EGO_CALL = dict(tokens_in=17151, tokens_out=97, cached_tokens=8519)  # trace 1960, turn 97


def test_the_ruler_reproduces_the_number_the_ledger_actually_stored():
    """The twin, run in BOTH worlds: the same real call under the entry that shipped.

    ``$0.017733`` is what ``token_ledger`` holds for it, to the sixth decimal — this is the
    line that proves the test measures the same thing the system measured, and it cannot rot,
    because the broken table is built here rather than read from the book."""
    was_shipped = PriceBook.from_mapping({"llm": {
        "openai:gpt-5.6-luna": {"input": 1.00, "output": 6.00},   # no cached_input → full price
        "_default": {"input": 0.0, "output": 0.0},
    }})
    before = was_shipped.llm_cost_usd("gpt-5.6-luna", **_EGO_CALL)
    assert before == pytest.approx(0.017733, abs=5e-7)
    after = PriceBook.default().llm_cost_usd("gpt-5.6-luna", **_EGO_CALL)
    assert before / after == pytest.approx(8.81, abs=0.01)


def test_the_measured_call_is_priced_from_the_published_standard_rates():
    """The bare name is the one the ledger stores — the factory strips the prefix before it
    writes. Both the rates and the resulting cost are pinned: reverting the row fails the
    second assertion, and fixing input/output while forgetting ``cached_input`` (the fix a
    hurry produces, worth $0.003547 on this call) fails both."""
    book = PriceBook.default()
    assert book.rates["llm"]["openai:gpt-5.6-luna"] == {
        "input": 0.20, "output": 1.20, "cached_input": 0.02}
    assert book.llm_cost_usd("gpt-5.6-luna", **_EGO_CALL) == pytest.approx(0.00201318, abs=1e-8)
    # half a fix is not a fix, and it is neither the red nor the green number
    half = (17151 / 1e6) * 0.20 + (97 / 1e6) * 1.20
    assert half == pytest.approx(0.0035466, abs=1e-8)
    assert book.llm_cost_usd("gpt-5.6-luna", **_EGO_CALL) < half


def test_the_two_calls_in_the_same_turn_that_were_already_right_do_not_move():
    """The control. The same turn priced two other models correctly; this PR must not touch a
    digit of either, and both rates are confirmed against the same page as the three it does
    change."""
    book = PriceBook.default()
    assert book.llm_cost_usd("gpt-4o-mini", 3182, 326, cached_tokens=3072) == \
        pytest.approx(0.00044250, abs=1e-8)
    assert book.llm_cost_usd("gpt-4.1-nano", 1485, 41, cached_tokens=1280) == \
        pytest.approx(0.00006890, abs=1e-8)


@pytest.mark.parametrize("model,rates", [
    ("openai:gpt-5.6-luna", {"input": 0.20, "output": 1.20, "cached_input": 0.02}),
    ("openai:gpt-5.6-terra", {"input": 2.00, "output": 12.00, "cached_input": 0.20}),
    ("openai:gpt-5.6-sol", {"input": 4.00, "output": 20.00, "cached_input": 0.40}),
])
def test_the_5_6_family_carries_the_standard_short_context_rates(model, rates):
    """All three were wrong on every column, and all three disagreements between readers were
    the same mistake: a different CELL of a 4-tier x 2-context grid, not a different price.
    ``gpt-5.6-sol``'s 4.00/20.00 is promotional (the page says at least through 2026-11-21),
    which is the one row here with an expiry rather than a correction."""
    assert PriceBook.default().rates["llm"][model] == rates


# ── the mechanism: silence becomes a written decision ──────────────────────────────────────

def test_every_openai_model_declares_its_cache_rate_or_declares_why_not():
    """The generalisation of the defect, and the only part of this that outlives the row.

    Six OpenAI models carried no ``cached_input`` and nothing said whether that meant "the
    provider publishes none" or "nobody looked". One of them was 92% of the bill. There is no
    third state now: carry the rate, or be named with the reason."""
    from cogno_meter.pricing import CACHE_RATE_NOT_PUBLISHED, DEFAULT_RATES

    undeclared = sorted(
        k for k, r in DEFAULT_RATES["llm"].items()
        if k.startswith("openai:") and not k.endswith("_default")
        and isinstance(r, dict) and "cached_input" not in r
        and k not in CACHE_RATE_NOT_PUBLISHED)
    assert not undeclared, (
        "OpenAI models with neither a cached_input rate nor a written reason: "
        f"{undeclared} — add the published rate, or name it in CACHE_RATE_NOT_PUBLISHED")


def test_the_exemption_list_cannot_become_a_parking_lot():
    """Every list like this one rots the same way: a name stays on it after the reason is gone,
    and the exemption becomes permanent. Each entry must name a model the table actually has,
    must still lack the rate, and must say why."""
    from cogno_meter.pricing import CACHE_RATE_NOT_PUBLISHED, DEFAULT_RATES

    llm = DEFAULT_RATES["llm"]
    for model, reason in CACHE_RATE_NOT_PUBLISHED.items():
        assert model in llm, f"{model} is exempted but is not in the price book"
        assert "cached_input" not in llm[model], \
            f"{model} now HAS a cached rate — drop it from CACHE_RATE_NOT_PUBLISHED"
        assert reason and reason.strip(), f"{model} is exempted with no reason"


def test_the_providers_this_table_cannot_model_are_named_with_their_reason():
    """Anthropic and Gemini are out of the gate because their caching has a dimension this book
    does not have — a write/read split and a per-token-HOUR storage line. That is a reason, not
    an oversight, so it is data a test can read. If one of their models ever acquires a
    ``cached_input`` the exemption is the thing that has to move, not the assertion."""
    from cogno_meter.pricing import CACHE_RATE_NOT_MODELLED, DEFAULT_RATES

    llm = DEFAULT_RATES["llm"]
    for provider, reason in CACHE_RATE_NOT_MODELLED.items():
        models = [k for k in llm if k.startswith(f"{provider}:") and not k.endswith("_default")]
        assert models, f"{provider} is exempted but has no models in the book"
        assert reason and reason.strip(), f"{provider} is exempted with no reason"
        priced = [m for m in models if "cached_input" in llm[m]]
        assert not priced, (
            f"{priced} carry a cached_input while {provider} is declared un-modellable — "
            "either the dimension is now modelled (drop the exemption) or the rate is a guess")


# ── the floor could not fire on a single LLM row: the ledger stores the BARE name ──────────
#
# ``create_backend("openai:gpt-4o-mini")`` splits the spec and hands the backend the bare model,
# so the ledger holds ``gpt-4o-mini``. The floor added in #13 tested for a ``provider:`` prefix
# and therefore never fired: every model outside the table metered at ZERO, in silence. Measured
# on the base tree over the four names below (all 0.0, no log line) and over all six Bedrock ids
# in one deployment's catalogue.
#
# The rule these pin is a DISJUNCTION, because the two halves have different costs when wrong:
# the floor may only fire where a provider can honestly be attributed (a false hit invents cost
# on somebody's own hardware and inflates their ceiling — trading this defect for a worse one),
# and everything else must at least be NAMED. What may never happen is neither.

_BARE_CLOUD_NAMES = [
    "gpt-6-mini",                    # an OpenAI model this book has not catalogued yet
    "claude-opus-5",                 # ditto, Anthropic
    "deepseek-chat",                 # a provider with no rates at all in the book
    "anthropic.claude-opus-4-6-v1",  # Bedrock's vendor-qualified form
]

# every Bedrock id in one deployment's model catalogue — the population, not the example
_BEDROCK_CATALOGUE = [
    "meta.llama4-scout-17b-instruct-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "meta.llama4-maverick-17b-instruct-v1:0",
    "anthropic.claude-opus-4-6-v1",
]


@pytest.mark.parametrize("model", _BARE_CLOUD_NAMES)
def test_a_bare_cloud_model_name_is_floored_and_named(model, caplog):
    """All four are attributable, so BOTH halves fire: a non-zero floor and a line naming it."""
    book = PriceBook.default()
    with caplog.at_level("WARNING", logger="cogno_meter.pricing"):
        cost = book.llm_cost_usd(model, 1_000_000, 1_000_000)
    assert cost > 0.0, f"{model} metered FREE"
    assert model in caplog.text, f"{model} was floored without being named"


@pytest.mark.parametrize("model", _BEDROCK_CATALOGUE)
def test_no_model_in_the_bedrock_catalogue_meters_free_in_silence(model, caplog):
    """Four of the six carry a vendor this book knows and take the floor; the two ``meta.`` ones
    do not, and cost 0 — but they are NAMED, which is the half of the rule that has to hold for
    every id, including the ones nothing can be floored against."""
    book = PriceBook.default()
    with caplog.at_level("WARNING", logger="cogno_meter.pricing"):
        cost = book.llm_cost_usd(model, 1_000_000, 1_000_000)
    assert cost > 0.0 or model in caplog.text, f"{model} metered free, in silence"


@pytest.mark.parametrize("model", ["qwen3:8b", "ollama:nomic-embed-text:latest",
                                   "mistral:latest", "nomic-embed-text:latest",
                                   "ollama:anything-at-all"])
def test_a_self_hosted_model_stays_exactly_free_and_stays_quiet(model, caplog):
    """THE control, and the most important assertion in this file.

    3.7M tokens in one ledger are these models. Their zero is deliberate and defended: a fix
    that makes the floor bite here invents cost where there is none and inflates the tenant's
    daily ceiling — the same harm as the defect, pointing the other way. Quiet as well as free,
    because a warning on the model that runs every turn is how a warning stops being read."""
    book = PriceBook.default()
    with caplog.at_level("WARNING", logger="cogno_meter.pricing"):
        assert book.llm_cost_usd(model, 1_000_000, 1_000_000) == 0.0, model
    assert not caplog.text, f"{model} is declared free, not unknown: {caplog.text}"


def test_the_colon_cannot_be_the_discriminator_and_gpt_oss_is_why():
    """``gpt-oss:20b`` runs on Ollama and its leading token is the very family that identifies
    OpenAI. Five of the six Bedrock ids also carry a colon, so neither "has a colon" nor "has
    the family" can decide alone. The dot does: a vendor-qualified id has one before the tag."""
    book = PriceBook.default()
    assert book.llm_cost_usd("gpt-oss:20b", 1_000_000, 1_000_000) == 0.0
    assert book.llm_cost_usd("gpt-6-mini", 1_000_000, 1_000_000) > 0.0        # same family
    assert book.llm_cost_usd("anthropic.claude-opus-4-6-v1", 1_000, 1_000) > 0.0  # same colon-less


def test_an_ambiguous_family_is_not_attributed(caplog):
    """``whisper`` is shipped by openai AND groq in the stt table, so a bare ``whisper-*`` cannot
    be charged to either. Ambiguous means NOT attributed — named, never guessed."""
    book = PriceBook.default()
    from cogno_meter.pricing import DEFAULT_RATES
    assert PriceBook._families(DEFAULT_RATES["stt"])["whisper"] is None
    with caplog.at_level("WARNING", logger="cogno_meter.pricing"):
        assert book.stt_cost_usd("whisper-9-turbo", 10) == 0.0
    assert "whisper-9-turbo" in caplog.text


def test_a_declared_provider_default_is_honoured_and_a_bare_name_reaches_it(caplog):
    """A host that declares ``<provider>:_default`` gets it — for the prefixed id, for a bare
    name the table lets it attribute, and for a scheme this book does not know, which is how
    ``ollama:_default = 0`` keeps meaning what it says."""
    book = PriceBook.from_mapping({"llm": {
        "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},   # gives the book a 'gpt' family
        "openai:_default": {"input": 10.0, "output": 30.0},
        "ollama:_default": {"input": 0.0, "output": 0.0},
        "_default": {"input": 0.0, "output": 0.0},
    }})
    assert book.llm_cost_usd("openai:gpt-brand-new", 1_000_000, 1_000_000) == pytest.approx(40.0)
    assert book.llm_cost_usd("gpt-brand-new", 1_000_000, 1_000_000) == pytest.approx(40.0)
    with caplog.at_level("WARNING", logger="cogno_meter.pricing"):
        assert book.llm_cost_usd("ollama:whatever", 1_000_000, 1_000_000) == 0.0
    assert not caplog.text          # a DECLARED zero is a decision, not an unknown


def test_attribution_is_one_sided_and_a_book_with_no_catalogue_cannot_attribute(caplog):
    """The inference is derived from the table, so a book that catalogues no model has nothing
    to derive from and attributes nothing. The miss costs today's behaviour (zero) plus a line
    naming the model; the opposite error would invent a charge. That asymmetry is the design."""
    book = PriceBook.from_mapping({"llm": {
        "openai:_default": {"input": 10.0, "output": 30.0},
        "_default": {"input": 0.0, "output": 0.0},
    }})
    with caplog.at_level("WARNING", logger="cogno_meter.pricing"):
        assert book.llm_cost_usd("gpt-brand-new", 1_000_000, 1_000_000) == 0.0
    assert "gpt-brand-new" in caplog.text


def test_a_catalogued_model_resolves_exactly_as_before(caplog):
    """The byte-identical twin: nothing on the matching paths moved, and a model the book
    prices must never reach the floor or emit a line."""
    book = PriceBook.default()
    with caplog.at_level("WARNING", logger="cogno_meter.pricing"):
        assert book.llm_cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
        assert book.llm_cost_usd("openai:gpt-4o", 1_000_000, 0) == pytest.approx(2.50)
        assert book.llm_cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(0.15)
    assert not caplog.text
