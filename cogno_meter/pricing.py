"""
cogno_meter.pricing — the price book + per-modality cost resolution.

Adapted clean-room from the parent ``cogno/core/pricing.py``: model rates with
exact → fuzzy-prefix (version suffixes) → ``_default`` resolution, USD→currency
conversion, and per-modality cost. Two distinct numbers come out of here:

  * **billable_tokens** — what counts toward the plan allowance/overage (LLM &
    embedding tokens directly; audio = ``chars × audio_multiplier``).
  * **cost_usd** — the real upstream provider cost (transparency only). Local /
    self-hosted models resolve to ``_default = 0``. A prompt the provider served
    from its own cache (``UsageRecord.cached_tokens``) is priced at the model's
    ``cached_input`` rate where the book has one — and at the FULL input rate,
    with a warning, where it does not. The two numbers move independently: the
    cache discount is ours, the allowance is the customer's.

The lib ships a default seed (illustrative rates); a host injects its own via
``PriceBook.from_mapping(...)``. No YAML dependency — the host loads its config
(YAML/JSON/DB) and hands over a plain mapping.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field

from cogno_meter.types import Modality, UsageRecord

# Pure functions: no WARNING/INFO (the budget-block decision is the host's, and
# the caller logs the returned Bill). DEBUG only — inspect the calculation.
logger = logging.getLogger("cogno_meter.pricing")

# Illustrative seed — values are examples (verify against providers; host overrides).
# llm: USD per 1M tokens (input/output). embedding: USD per 1M tokens.
# Providers whose traffic is BILLED by someone. The floor below only applies to these, and the
# reason is a colon: an Ollama model is natively named ``model:tag`` (``qwen3:8b``,
# ``mistral:latest``), so "has a colon" cannot mean "has a provider". Treating it that way
# started charging for models running on the user's own hardware — caught by the existing
# `test_bare_model_name_resolves_to_prefixed_rate`, which pins qwen3:8b at 0.
#
# Kept in step with the backends cogno-synapse ships (its factory's named providers plus the
# _OPENAI_COMPATIBLE registry). A name not on this list resolves as before: unknown scheme,
# global default, zero.
#
# **The ledger does not store the prefix.** ``create_backend("openai:gpt-4o-mini")`` splits the
# spec and hands the backend the BARE model, so what reaches this book is ``gpt-4o-mini``, not
# ``openai:gpt-4o-mini``. The floor below was written against the prefixed form and therefore
# never fired on a single LLM row: every model outside the table metered at ZERO, in silence —
# measured over the four names below and over all six Bedrock ids in one deployment's catalogue.
# So the provider is now also inferred from a bare name (``_provider_of``), and anything left
# unattributed is NAMED at WARNING instead of costing nothing quietly.
_CLOUD_PROVIDERS = frozenset({
    "openai", "anthropic", "gemini", "grok", "groq", "bedrock",
    "deepseek", "moonshot", "xai", "openrouter", "together", "fireworks",
})

# ``cached_input`` (OPTIONAL, USD per 1M tokens) — what the provider charges for the part of the
# prompt it served from its own cache. **Absent means FULL PRICE**, which is exactly what this
# book did before the key existed, and the direction that is safe: a missing rate over-states our
# cost, so the daily BRL ceiling fires EARLY. A GUESSED discount would under-state it and the
# ceiling would fire LATE — the failure this file's own doctrine ("refuse to call paid traffic
# free") exists to prevent, one level down.
#
# **Which cell of the page a number comes from is part of the number.** OpenAI's pricing page is a
# 4-tier × 2-context grid — Standard / Batch / Flex / Fast, each split into *Short context*
# (≤272K input tokens) and *Long context* (>272K) with the four headers Input | Cached input |
# Cache writes | Output repeated on both sides. Every rate below is **Standard, short context**,
# because that is what an ordinary API call is billed at and no traffic this meter has seen comes
# near 272K. That sentence is not decoration: three readings of this same page disagreed and every
# disagreement was a different cell, not a different price (see the PR that added this line). Two dimensions of that grid are NOT modelled here and both are named, with their
# direction, rather than left to be rediscovered:
#   * **Long context** (>272K input) is 2x input / 2x cached / 1.5x output for the 5.6 family. A
#     call that big is priced here at the short-context rate — an UNDER-report. No record carries
#     a context class, so fixing it means a second column keyed on ``tokens_in``, not a number.
#   * **Cache writes** (1.25x input) is a THIRD partition of the prompt, not an additive fee
#     ("Input tokens are either Input, Cached Input, or Cache Write"). ``UsageRecord`` has no
#     count for it, so those tokens are charged at ``input`` — an UNDER-report bounded at 25% of
#     whatever share was written. It needs a field before it can need a rate.
#
# The key is seeded wherever the provider publishes a flat per-token cached-input rate, and only
# for OpenAI. That is not a preference, it is the shape of the products:
#   * OpenAI bills cached prompt tokens at a fixed lower per-token rate, automatically, with no
#     write surcharge and no storage line — one number per model, which is what this column is.
#   * Anthropic splits it in two (cache WRITE at 1.25x base input, cache READ at 0.10x) and the
#     usage payload reports them separately; one column cannot carry both, and pricing a read at
#     0.10x while silently ignoring the write would under-state the bill.
#   * Gemini's context caching is billed per token-HOUR of storage plus a discounted read — a
#     dimension (time) this book does not have.
# Those two get NO ``cached_input`` and are charged full price, with the warning below naming
# them. Adding them properly means adding their dimensions, not a number — and that sentence now
# lives in ``CACHE_RATE_NOT_MODELLED`` where a test can read it, instead of only in this comment.
#
# **Silence is not a third state.** An ``openai:`` model either carries ``cached_input`` or is
# named in ``CACHE_RATE_NOT_PUBLISHED`` with the reason; ``test_every_openai_model_declares_its_
# cache_rate_or_declares_why_not`` fails on the one that does neither. Before that gate existed
# six OpenAI models had no rate and no declaration, and nothing distinguished "the provider does
# not publish one" from "nobody has looked yet" — which is how ``gpt-5.6-luna`` ran for weeks at
# 5x its input rate and 5x its output rate with the cache discount thrown away on top. It is the
# dominant line of one deployment's reported provider spend: measured over its ``token_ledger``
# twice on 2026-09-06, hours apart, at 78.2% and 77.5% (the SHARE is the durable figure; the
# totals were already stale between the two readings), and reported at ~92% when this row was
# corrected — a share that has NOT been re-measured here, and is quoted as the later report it is. Measured on one real call (trace 1960, turn 97, the EGO
# stage: 17151 in / 8519 of them cached / 97 out) the book reported **$0.017733** — the figure in
# ``token_ledger`` to the sixth decimal — against **$0.002013** at the published rates: an 8.8x
# over-report. Nobody was over-BILLED (the invoice and the monthly allowance are both denominated
# in tokens, and ``billable_tokens`` never saw this), but the daily BRL ceiling is denominated in
# money: the guard saw R$ 6.18 of spend in 24h where the truth was ≈R$ 1.38 and started degrading
# and refusing service at 22% of the real budget. The defect took nobody's money; it took service.
# stt: USD per minute of audio. tts: USD per 1M characters. _default: self-hosted = 0.
DEFAULT_RATES: dict = {
    "llm": {
        # exact keys for every catalogued model — the fuzzy resolver would otherwise let a
        # bare "gpt-5-mini" greedily prefix-match "gpt-5"'s far pricier rate.
        "openai:gpt-5-nano": {"input": 0.05, "output": 0.40, "cached_input": 0.005},
        "openai:gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
        "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
        "openai:gpt-5.4-nano": {"input": 0.20, "output": 1.25, "cached_input": 0.02},
        "openai:gpt-5-mini": {"input": 0.25, "output": 2.00, "cached_input": 0.025},
        "openai:gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
        "openai:gpt-5.4-mini": {"input": 0.75, "output": 4.50, "cached_input": 0.075},
        "openai:gpt-5.6-luna": {"input": 0.20, "output": 1.20, "cached_input": 0.02},
        "openai:gpt-5": {"input": 1.25, "output": 10.00, "cached_input": 0.125},
        "openai:gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
        "openai:gpt-5.6-terra": {"input": 2.00, "output": 12.00, "cached_input": 0.20},
        "openai:gpt-5.6-sol": {"input": 4.00, "output": 20.00, "cached_input": 0.40},  # promo
        "openai:gpt-5.5-pro": {"input": 30.00, "output": 180.00},
        "openai:gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
        # anthropic — real per-MTok rates (Opus 4.6=$5/$25, Sonnet 4.5=$3/$15, Haiku 4.5=$1/$5);
        # the original Opus 4 keeps its launch $15/$75.
        # Anthropic API IDs are hyphenated (claude-opus-4-6, not …4.6) — must match the id the
        # backend reports into the ledger, or the rate resolves to _default (0).
        "anthropic:claude-haiku-4-5": {"input": 1.00, "output": 5.00},
        "anthropic:claude-sonnet-4-0": {"input": 3.00, "output": 15.00},
        "anthropic:claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        "anthropic:claude-opus-4-0": {"input": 15.00, "output": 75.00},
        "anthropic:claude-opus-4-5": {"input": 5.00, "output": 25.00},
        "anthropic:claude-opus-4-6": {"input": 5.00, "output": 25.00},
        # gemini — 2.5 family verified vs ai.google.dev; 3.x preview partly estimated.
        "gemini:gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},   # verified
        "gemini:gemini-2.5-flash": {"input": 0.30, "output": 2.50},        # verified
        "gemini:gemini-3-flash": {"input": 0.50, "output": 3.00},          # input verified, output est.
        "gemini:gemini-2.5-pro": {"input": 1.25, "output": 10.00},         # verified (≤200k prompt)
        "gemini:gemini-3-pro": {"input": 2.00, "output": 12.00},           # est. (aligned to 3.1-pro)
        "gemini:gemini-3.1-pro": {"input": 2.00, "output": 12.00},         # verified (3.1 Pro Preview)
        # grok — verified vs docs.x.ai 2026-08-04. Two corrections to the previous seed:
        #
        # (1) grok-4.20 is NOT fictional (an earlier comment here said so). Three real snapshot
        #     variants ship, all at the 4.3 rate; one prefix key covers them.
        # (2) The retired ids are KEPT, repriced. xAI's 2026-05-15 retirement took grok-3,
        #     grok-4-0709 and the grok-4-1-fast pair, and a request to a retired slug is
        #     REDIRECTED to grok-4.3 and BILLED AT GROK-4.3 RATES. So the old rates here were
        #     over-reporting real spend by up to 6x — and DELETING the keys would be worse than
        #     leaving them wrong: 'grok:grok-4' matches no surviving key (not even by bare
        #     prefix — 'grok-4' does not start with 'grok-4.3'), so it would fall through to
        #     _default = 0 and report paid traffic as FREE. A meter must charge what the
        #     provider charges, including for a slug the provider silently reroutes.
        "grok:grok-3-mini": {"input": 1.25, "output": 2.50},   # retired → served/billed as 4.3
        "grok:grok-3": {"input": 1.25, "output": 2.50},        # retired → served/billed as 4.3
        "grok:grok-4": {"input": 1.25, "output": 2.50},        # retired → served/billed as 4.3
        "grok:grok-4.1-fast": {"input": 1.25, "output": 2.50},  # retired → served/billed as 4.3
        "grok:grok-build-0.1": {"input": 1.00, "output": 2.00},  # verified (coding)
        "grok:grok-4.20": {"input": 1.25, "output": 2.50},     # verified (all -0309-* variants)
        "grok:grok-4.3": {"input": 1.25, "output": 2.50},      # verified
        "grok:grok-4.5": {"input": 2.00, "output": 6.00},      # verified (flagship)
        "ollama:_default": {"input": 0.0, "output": 0.0},  # self-hosted
        "_default": {"input": 0.0, "output": 0.0},
    },
    "embedding": {
        "openai:text-embedding-3-small": 0.02,
        "openai:text-embedding-3-large": 0.13,
        "_default": 0.0,
    },
    "stt": {  # USD per minute
        "openai:whisper-1": 0.006,
        "groq:whisper-large-v3-turbo": 0.04,
        "_default": 0.0,  # self-hosted faster-whisper
    },
    "tts": {  # USD per 1M characters
        "openai:tts-1": 15.00,
        "openai:tts-1-hd": 30.00,
        # Carried at the `tts-1` rate DELIBERATELY, and it is a placeholder, not a measurement:
        # this model is billed by TOKEN (text in + audio out) while this book is per CHARACTER,
        # and the conversion depends on the voice and the language. It is listed anyway because
        # the alternative is worse: without an entry every voiced turn on it logs
        # `rate_uncatalogued` and it silently lands on the same 15.00 floor — same number,
        # unstated, plus a warning per message. Stating the choice makes it reviewable.
        # The DIRECTION of the error is not stated, because the first version stated it wrong.
        # It said "most likely over-states, the safe direction for the platform" — from the
        # headline that this model is cheaper than `tts-1`. Worked the right way for a TTS
        # model, the billable volume is AUDIO-OUTPUT tokens rather than input characters, and a
        # review put the true rate near $17/1M chars for English (≈$14.5–20 across 130–180 wpm):
        # it BRACKETS $15 and centres above it, so this most likely UNDER-reports — which is the
        # failure this file's own doctrine ("refuse to call paid traffic free") exists to
        # prevent. Confirm against a real invoice before anyone leans on it.
        "openai:gpt-4o-mini-tts": 15.00,
        "xai:grok-2-tts": 4.20,
        "_default": 0.0,  # self-hosted Kokoro
    },
}

# Models a reader of the table above would expect to carry ``cached_input`` and which
# deliberately do not, each with the reason. The gate in the tests reads THIS, so adding a model
# without a cache rate is a decision somebody has to write down rather than an omission nobody
# notices. Every name here must exist in the table and must NOT carry the rate — a stale
# exemption is the failure mode of every list like this one, and the test fails on it.
CACHE_RATE_NOT_PUBLISHED: dict = {
    "openai:gpt-5.5-pro": "the Standard table prints '-' in the Cached input column",
}

# Whole providers the gate deliberately does not cover, with the reason it cannot. These are not
# missing numbers, they are missing DIMENSIONS (see the long comment above the table): a column
# here would have to lie about one of them. Kept as data, not prose, so the test can check that
# none of their models has quietly acquired a ``cached_input`` behind the exemption.
CACHE_RATE_NOT_MODELLED: dict = {
    "anthropic": "cache WRITE (1.25x input) and cache READ (0.10x) are billed and reported "
                 "separately; one column cannot carry both, and pricing the read while ignoring "
                 "the write would under-state the bill",
    "gemini": "context caching is billed per token-HOUR of storage plus a discounted read — a "
              "dimension (time) this book does not have",
}

DEFAULT_USD_BRL_RATE = 5.70
DEFAULT_AUDIO_MULTIPLIER = 2.0


@dataclass
class PriceBook:
    rates: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_RATES))
    usd_brl_rate: float = DEFAULT_USD_BRL_RATE
    audio_multiplier: float = DEFAULT_AUDIO_MULTIPLIER  # chars × this → billable tokens
    # Models already reported as having no `cached_input` rate (see `_declare_no_cache_rate`).
    # Not part of the book's identity: excluded from init/repr/eq so two books built from the
    # same mapping stay equal and a log-dedup set never leaks into a comparison.
    _no_cache_rate_declared: set = field(
        default_factory=set, init=False, repr=False, compare=False)

    @classmethod
    def default(cls) -> "PriceBook":
        return cls()

    @classmethod
    def from_mapping(cls, mapping: dict) -> "PriceBook":
        """Build from a host-supplied mapping (the parsed pricing config)."""
        return cls(
            rates={k: v for k, v in mapping.items()
                   if k in ("llm", "embedding", "stt", "tts")},
            usd_brl_rate=float(mapping.get("usd_brl_rate", DEFAULT_USD_BRL_RATE)),
            audio_multiplier=float(mapping.get("audio_multiplier", DEFAULT_AUDIO_MULTIPLIER)),
        )

    # ── rate resolution: exact → prefix-fuzzy → BARE (no 'provider:' prefix) → _default ──
    #
    # The ledger stores the backend's bare model name ('gpt-4o', 'qwen3:8b'), but the rate keys
    # carry a 'provider:' prefix ('openai:gpt-4o'). So besides matching a prefixed model, also
    # match a bare model against the key's model part (exact first so 'gpt-4o' doesn't greedily
    # grab 'gpt-4o-mini', then longest-prefix fuzzy for versioned names like 'gpt-4o-mini-2024…').
    @staticmethod
    def _bare(key: str) -> str:
        return key.split(":", 1)[1] if ":" in key else key

    @staticmethod
    def _resolve(table: dict, model: str):
        if model in table:
            logger.debug("event=rate_resolve model=%s match=exact", model)
            return table[model]
        items = [(k, r) for k, r in table.items() if k != "_default"]
        # Longest-key-first so a short key that string-prefixes a longer one (``gpt-5`` vs
        # ``gpt-5.5-pro``) never steals the more specific model's rate. All fuzzy passes below
        # must be longest-first; step 2 used to iterate in dict-insertion order and mispriced
        # dated builds of flagship models (~19x under-report on the cost-transparency figure).
        prefix_items = sorted(items, key=lambda kv: len(kv[0]), reverse=True)
        for key, rate in prefix_items:                # prefixed model vs prefixed key (versioned)
            if model.startswith(key):
                logger.debug("event=rate_resolve model=%s match=fuzzy key=%s", model, key)
                return rate
        for key, rate in items:                       # bare model == the key's model part
            if model == PriceBook._bare(key):
                logger.debug("event=rate_resolve model=%s match=bare key=%s", model, key)
                return rate
        for key, rate in sorted(items, key=lambda kv: len(PriceBook._bare(kv[0])), reverse=True):
            bare = PriceBook._bare(key)               # bare versioned → longest bare prefix wins
            if bare and model.startswith(bare):
                logger.debug("event=rate_resolve model=%s match=bare_fuzzy key=%s", model, key)
                return rate
        # Per-provider catch-all (``openai:_default``) before the global one, so a host can set a
        # non-zero default per provider instead of every un-catalogued model silently costing 0.
        #
        # ``_provider_of`` answers for the PREFIXED form and for the bare name the ledger really
        # stores; when it answers, the three steps below are exactly the ones this block has
        # always run.
        provider = PriceBook._provider_of(table, model)
        if provider is not None:
            provider_default = table.get(f"{provider}:_default")
            if provider_default is not None:
                logger.debug("event=rate_resolve model=%s match=provider_default", model)
                return provider_default
            # NOTHING declared for this provider, and it is not self-hosted (those carry an
            # explicit ``ollama:_default = 0``). Falling through to the global ``_default``
            # would price the call at ZERO — the exact failure the grok comment above calls
            # out ("report paid traffic as FREE"), except it is not specific to xAI: measured
            # 2026-08-05, an unknown model on EVERY cloud provider metered free. A provider
            # ships a model, a tenant pins it, and the bill is 0 until someone remembers.
            #
            # The floor is the provider's own CHEAPEST catalogued rate, derived from the table
            # rather than guessed, so it stays right as the table changes. Cheapest and not
            # flagship on purpose: this feeds BudgetGuard as well as billing, and the smallest
            # non-zero number cannot wrongly block a tenant. It under-reports a premium model —
            # that is a known, bounded error, and the WARNING is what gets it fixed. Silence
            # was the unbounded one.
            floor = PriceBook._provider_floor(table, provider)
            if floor is not None:
                logger.warning(
                    "event=rate_uncatalogued model=%s provider=%s using_floor=%s — add this "
                    "model to the price book; it is being metered at the provider's cheapest "
                    "known rate, not its real one", model, provider, floor)
                return floor
            # The provider has NO rates at all. Measured 2026-08-05: cogno-synapse ships
            # backends for groq, deepseek, moonshot, xai, openrouter, together and fireworks,
            # and this book prices only openai/anthropic/gemini/grok/ollama — so a tenant on
            # any of the others meters ENTIRELY free, not just on one unknown model.
            #
            # That is a DATA gap (someone must supply real rates) and this cannot invent them.
            # What it can do is refuse to call paid traffic free: fall back to the cheapest
            # CLOUD rate in the whole book — still derived, still the smallest non-zero number,
            # so it cannot wrongly trip a budget — and say so at WARNING with the provider
            # named, which is what gets the rates added.
            cheapest_cloud = PriceBook._cheapest_cloud(table)
            if cheapest_cloud is not None:
                logger.warning(
                    "event=rate_provider_uncatalogued model=%s provider=%s using_floor=%s — "
                    "this provider has NO rates in the price book; its traffic would otherwise "
                    "meter as FREE. Add %s rates.", model, provider, cheapest_cloud, provider)
                return cheapest_cloud
            # A known cloud provider and the book has no priced rate anywhere to floor against.
            # Nothing left to charge, but it is still paid traffic: name it.
            logger.warning(
                "event=rate_unattributed model=%s provider=%s — the price book holds no rate "
                "this call can be floored against, so it is metered at ZERO. Add rates for %s.",
                model, provider, provider)
            return table.get("_default")

        # Nobody bills this model as far as this book can tell. Two ways that is FINE and one
        # way it is the defect, and they are told apart so the warning stays worth reading:
        #
        #   1. the host DECLARED a default for this scheme (``ollama:_default = 0``) — a written
        #      decision, so it is honoured for any ``ollama:<anything>`` and logged at DEBUG;
        #   2. the name has Ollama's native ``model:tag`` shape and no attributable provider, so
        #      it is a model on the user's own hardware (``qwen3:8b``, ``mistral:latest``) —
        #      free, quietly, which is what it has always been and what it must stay;
        #   3. anything else is a model somebody is being charged for and this book cannot name.
        #      It still costs 0 here (inventing a number is the worse trade — see the control in
        #      the tests) but it is no longer SILENT, and the WARNING carries the model, because
        #      a floor that fires and a name nobody logged are equally invisible in a ledger.
        if ":" in model:
            declared = table.get(f"{model.split(':', 1)[0]}:_default")
            if declared is not None:
                logger.debug("event=rate_resolve model=%s match=declared_default", model)
                return declared
        if PriceBook._looks_self_hosted(model):
            logger.debug("event=rate_resolve model=%s match=self_hosted_tag", model)
            self_hosted = table.get("ollama:_default")
            return self_hosted if self_hosted is not None else table.get("_default")
        logger.warning(
            "event=rate_unattributed model=%s — nothing in the price book prices this model and "
            "its name carries no provider this book knows, so it is metered at ZERO. Add the "
            "model, or give the ledger a `provider:` prefix; paid traffic reported as free is "
            "the failure this floor exists to prevent.", model)
        return table.get("_default")

    # ── who bills this model, for a prefixed id AND for the bare name the ledger stores ──
    #
    # Attribution is deliberately one-sided: a miss leaves today's behaviour (zero, now with a
    # warning), a false hit INVENTS cost on a model running on the user's own hardware and
    # inflates that tenant's daily ceiling — trading this defect for a worse one. So every step
    # is derived from data already in this file, and the self-hosted shape wins over all of it.
    @staticmethod
    def _provider_of(table: dict, model: str) -> "str | None":
        if ":" in model and model.split(":", 1)[0] in _CLOUD_PROVIDERS:
            return model.split(":", 1)[0]
        if PriceBook._looks_self_hosted(model):
            return None
        head = PriceBook._head(model.split(":", 1)[0])
        if not head:
            return None
        if head in _CLOUD_PROVIDERS:        # 'anthropic.claude-…-v1:0', 'deepseek-chat'
            return head
        return PriceBook._families(table).get(head)   # 'gpt-6-mini' → openai, via the table

    @staticmethod
    def _looks_self_hosted(model: str) -> bool:
        """Ollama's native name is ``model:tag`` — ``qwen3:8b``, ``mistral:latest``.

        A Bedrock id is ``vendor.family-…-vN:0`` and also carries a colon, so "has a colon"
        decides nothing and a first version of the floor that read it that way started charging
        for local models. The DOT decides: a vendor-qualified id has one before the tag, a plain
        ``name:tag`` does not. This is what stops ``gpt-oss:20b`` — whose leading token is the
        very family that identifies OpenAI — from being attributed and billed."""
        return ":" in model and "." not in model.rsplit(":", 1)[0]

    @staticmethod
    def _head(name: str) -> str:
        return re.split(r"[-./]", name, maxsplit=1)[0].lower()

    @staticmethod
    def _families(table: dict) -> dict:
        """``{family token: the one cloud provider that ships it}``, derived from the book.

        ``gpt`` → openai, ``claude`` → anthropic, ``gemini`` → gemini, ``grok`` → grok. A family
        two providers share (``whisper``, for openai and groq in the stt table) maps to ``None``:
        ambiguous means NOT attributed, because the cost of guessing wrong is a charge invented
        against the wrong tenant."""
        families: dict = {}
        for key in table:
            if ":" not in key or key.endswith("_default"):
                continue
            provider, bare = key.split(":", 1)
            if provider not in _CLOUD_PROVIDERS:
                continue
            head = PriceBook._head(bare)
            if not head:
                continue
            if head in families and families[head] != provider:
                families[head] = None
            else:
                families.setdefault(head, provider)
        return families

    @staticmethod
    def _cheapest_cloud(table: dict):
        """The cheapest non-zero rate in the book — the floor for a provider with no rates."""
        priced = [r for k, r in table.items()
                  if isinstance(r, dict) and not k.endswith("_default")
                  and (r.get("output", 0.0) or 0.0) > 0.0]
        if not priced:
            return None
        return min(priced, key=lambda r: (r.get("output", 0.0), r.get("input", 0.0)))

    @staticmethod
    def _provider_floor(table: dict, provider: str):
        """The provider's cheapest catalogued rate, or None when it has no entries at all."""
        rates = [r for k, r in table.items()
                 if k.startswith(f"{provider}:") and not k.endswith(":_default")]
        if not rates:
            return None
        # Embedding tables hold bare floats; LLM tables hold {input, output} dicts.
        if all(isinstance(r, (int, float)) for r in rates):
            return min(rates)
        priced = [r for r in rates if isinstance(r, dict)]
        return min(priced, key=lambda r: (r.get("output", 0.0), r.get("input", 0.0))) or None

    # ── provider cost (transparency), in USD ──────────────────────────
    def llm_cost_usd(self, model: str, tokens_in: int, tokens_out: int,
                     cached_tokens: int = 0) -> float:
        """Real upstream cost of one LLM call.

        ``cached_tokens`` is the SUBSET of ``tokens_in`` the provider served from its prompt
        cache (0 = unknown/none, and then this is byte-for-byte the old calculation). It is
        priced at the model's ``cached_input`` rate when the book has one; when it does not,
        the whole prompt is charged at the full ``input`` rate — today's behaviour — and the
        omission is DECLARED (see ``_declare_no_cache_rate``). Never a default discount: an
        invented one under-states our own cost, and the daily budget ceiling then fires late.
        """
        rates = self._resolve(self.rates.get("llm", {}), model)
        if not isinstance(rates, dict):
            return 0.0
        input_rate = float(rates.get("input", 0))
        output_cost = (tokens_out / 1_000_000) * float(rates.get("output", 0))
        # Clamped to [0, tokens_in]: the cached part is a SUBSET, and a provider payload that
        # over-reports it must not produce a NEGATIVE fresh count (a credit on our own bill).
        cached = max(0, min(int(cached_tokens or 0), int(tokens_in or 0)))
        cached_rate = rates.get("cached_input")
        if cached <= 0 or cached_rate is None:
            if cached > 0:
                self._declare_no_cache_rate(model, cached)
            return (tokens_in / 1_000_000) * input_rate + output_cost
        fresh = int(tokens_in or 0) - cached
        return (fresh / 1_000_000) * input_rate + \
               (cached / 1_000_000) * float(cached_rate) + output_cost

    def _declare_no_cache_rate(self, model: str, cached: int) -> None:
        """Say, once per model per book, that a real cache discount is being left on the table.

        Once and not per call: the caller is a per-turn hot path and the models this fires for
        are the ones running every turn, so a line per call is a line nobody reads. The dedup
        lives on the INSTANCE (not the module) so a test gets a fresh book and a fresh count.
        """
        if model in self._no_cache_rate_declared:
            return
        self._no_cache_rate_declared.add(model)
        logger.warning(
            "event=cache_rate_uncatalogued model=%s cached_tokens=%d — the provider served part "
            "of this prompt from its cache and the price book has no `cached_input` rate for "
            "this model, so it is being charged at the FULL input rate. Our reported cost is "
            "therefore HIGH (the safe direction). Add the model's published cached-input rate.",
            model, cached)

    def embedding_cost_usd(self, model: str, tokens: int) -> float:
        rate = self._resolve(self.rates.get("embedding", {}), model)
        return (tokens / 1_000_000) * float(rate or 0.0)

    def stt_cost_usd(self, model: str, minutes: float) -> float:
        rate = self._resolve(self.rates.get("stt", {}), model)
        return float(minutes) * float(rate or 0.0)

    def tts_cost_usd(self, model: str, chars: int) -> float:
        rate = self._resolve(self.rates.get("tts", {}), model)
        return (chars / 1_000_000) * float(rate or 0.0)

    def usage_cost_usd(self, rec: UsageRecord) -> float:
        if rec.modality == Modality.LLM:
            return self.llm_cost_usd(rec.model, int(rec.tokens_in or 0), int(rec.tokens_out or 0),
                                     int(getattr(rec, "cached_tokens", 0) or 0))
        if rec.modality == Modality.EMBEDDING:
            return self.embedding_cost_usd(rec.model, int(rec.tokens_in or 0) or int(rec.tokens_out or 0))
        if rec.modality == Modality.STT:
            return self.stt_cost_usd(rec.model, rec.minutes)
        if rec.modality == Modality.TTS:
            return self.tts_cost_usd(rec.model, rec.chars)
        return 0.0

    # ── billable tokens (toward the allowance/overage) ────────────────
    def billable_tokens(self, rec: UsageRecord) -> int:
        """Tokens that count toward the monthly allowance. LLM/embedding use the
        token counts directly; audio is metered by chars × ``audio_multiplier``.

        Token fields are coerced (``None``/missing → 0): one malformed ledger record must not
        raise ``TypeError`` and abort billing for the whole period."""
        tin, tout = int(rec.tokens_in or 0), int(rec.tokens_out or 0)
        if rec.modality == Modality.LLM:
            # ``cached_tokens`` is deliberately NOT subtracted. The plan's allowance is a
            # contract denominated in tokens the customer's turns consumed; whether the
            # provider happened to serve part of the prompt from its own cache changes what
            # WE pay (``llm_cost_usd``), not what the customer used. Cutting the allowance
            # here would silently rewrite every published plan.
            return tin + tout
        # Embedding: mirror the cost path (``tokens_in or tokens_out``) so billable tokens and
        # the priced amount interpret the same record identically — embeddings carry input only.
        if rec.modality == Modality.EMBEDDING:
            return tin or tout
        # STT / TTS — always char-metered, scaled up by the audio multiplier.
        if rec.modality in (Modality.STT, Modality.TTS):
            return int(round((rec.chars or 0) * self.audio_multiplier))
        return 0
