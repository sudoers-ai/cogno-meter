

def test_every_tts_model_a_host_can_offer_has_a_stated_rate():
    """An uncatalogued model does not fail — it lands on the cheapest known floor and logs
    `rate_uncatalogued` on EVERY call. Same number, unstated, plus a warning per message.

    This does not check the number is right (`gpt-4o-mini-tts` is token-billed while this book
    is per character, and its entry says so). It checks the number is DECLARED, which is what
    makes it reviewable at all."""
    from cogno_meter.pricing import DEFAULT_RATES

    offered = {"openai:tts-1", "openai:tts-1-hd", "openai:gpt-4o-mini-tts"}
    missing = offered - set(DEFAULT_RATES["tts"])
    assert not missing, f"tts models with no stated rate: {sorted(missing)}"
