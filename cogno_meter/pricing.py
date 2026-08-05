"""
cogno_meter.pricing — the price book + per-modality cost resolution.

Adapted clean-room from the parent ``cogno/core/pricing.py``: model rates with
exact → fuzzy-prefix (version suffixes) → ``_default`` resolution, USD→currency
conversion, and per-modality cost. Two distinct numbers come out of here:

  * **billable_tokens** — what counts toward the plan allowance/overage (LLM &
    embedding tokens directly; audio = ``chars × audio_multiplier``).
  * **cost_usd** — the real upstream provider cost (transparency only). Local /
    self-hosted models resolve to ``_default = 0``.

The lib ships a default seed (illustrative rates); a host injects its own via
``PriceBook.from_mapping(...)``. No YAML dependency — the host loads its config
(YAML/JSON/DB) and hands over a plain mapping.
"""

from __future__ import annotations

import copy
import logging
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
_CLOUD_PROVIDERS = frozenset({
    "openai", "anthropic", "gemini", "grok", "groq", "bedrock",
    "deepseek", "moonshot", "xai", "openrouter", "together", "fireworks",
})

# stt: USD per minute of audio. tts: USD per 1M characters. _default: self-hosted = 0.
DEFAULT_RATES: dict = {
    "llm": {
        # exact keys for every catalogued model — the fuzzy resolver would otherwise let a
        # bare "gpt-5-mini" greedily prefix-match "gpt-5"'s far pricier rate.
        "openai:gpt-5-nano": {"input": 0.05, "output": 0.40},
        "openai:gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "openai:gpt-5.4-nano": {"input": 0.20, "output": 1.25},
        "openai:gpt-5-mini": {"input": 0.25, "output": 2.00},
        "openai:gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "openai:gpt-5.4-mini": {"input": 0.75, "output": 4.50},
        "openai:gpt-5.6-luna": {"input": 1.00, "output": 6.00},
        "openai:gpt-5": {"input": 1.25, "output": 10.00},
        "openai:gpt-4.1": {"input": 2.00, "output": 8.00},
        "openai:gpt-5.6-terra": {"input": 2.50, "output": 15.00},
        "openai:gpt-5.6-sol": {"input": 5.00, "output": 30.00},
        "openai:gpt-5.5-pro": {"input": 30.00, "output": 180.00},
        "openai:gpt-4o": {"input": 2.50, "output": 10.00},
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
        "xai:grok-2-tts": 4.20,
        "_default": 0.0,  # self-hosted Kokoro
    },
}

DEFAULT_USD_BRL_RATE = 5.70
DEFAULT_AUDIO_MULTIPLIER = 2.0


@dataclass
class PriceBook:
    rates: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_RATES))
    usd_brl_rate: float = DEFAULT_USD_BRL_RATE
    audio_multiplier: float = DEFAULT_AUDIO_MULTIPLIER  # chars × this → billable tokens

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
        if ":" in model and model.split(":", 1)[0] in _CLOUD_PROVIDERS:
            provider = model.split(":", 1)[0]
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
        logger.debug("event=rate_resolve model=%s match=default", model)
        return table.get("_default")

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
    def llm_cost_usd(self, model: str, tokens_in: int, tokens_out: int) -> float:
        rates = self._resolve(self.rates.get("llm", {}), model)
        if not isinstance(rates, dict):
            return 0.0
        return (tokens_in / 1_000_000) * float(rates.get("input", 0)) + \
               (tokens_out / 1_000_000) * float(rates.get("output", 0))

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
            return self.llm_cost_usd(rec.model, int(rec.tokens_in or 0), int(rec.tokens_out or 0))
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
            return tin + tout
        # Embedding: mirror the cost path (``tokens_in or tokens_out``) so billable tokens and
        # the priced amount interpret the same record identically — embeddings carry input only.
        if rec.modality == Modality.EMBEDDING:
            return tin or tout
        # STT / TTS — always char-metered, scaled up by the audio multiplier.
        return int(round((rec.chars or 0) * self.audio_multiplier))
