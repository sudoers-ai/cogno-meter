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
        # grok — verified vs xAI pricing; grok-3-mini estimated. (grok-4.20 was fictional →
        # dropped; real flagship is grok-4.5.)
        "grok:grok-3-mini": {"input": 0.30, "output": 0.50},               # est.
        "grok:grok-3": {"input": 2.00, "output": 10.00},                   # verified (legacy)
        "grok:grok-4": {"input": 3.00, "output": 15.00},                   # verified
        "grok:grok-4.1-fast": {"input": 0.20, "output": 0.50},             # verified
        "grok:grok-4.3": {"input": 1.25, "output": 2.50},                  # verified
        "grok:grok-4.5": {"input": 2.00, "output": 6.00},                  # verified (flagship)
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
        for key, rate in items:                       # prefixed model vs prefixed key (versioned)
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
        logger.debug("event=rate_resolve model=%s match=default", model)
        return table.get("_default")

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
            return self.llm_cost_usd(rec.model, rec.tokens_in, rec.tokens_out)
        if rec.modality == Modality.EMBEDDING:
            return self.embedding_cost_usd(rec.model, rec.tokens_in or rec.tokens_out)
        if rec.modality == Modality.STT:
            return self.stt_cost_usd(rec.model, rec.minutes)
        if rec.modality == Modality.TTS:
            return self.tts_cost_usd(rec.model, rec.chars)
        return 0.0

    # ── billable tokens (toward the allowance/overage) ────────────────
    def billable_tokens(self, rec: UsageRecord) -> int:
        """Tokens that count toward the monthly allowance. LLM/embedding use the
        token counts directly; audio is metered by chars × ``audio_multiplier``."""
        if rec.modality in (Modality.LLM, Modality.EMBEDDING):
            return rec.tokens_in + rec.tokens_out
        # STT / TTS — always char-metered, scaled up by the audio multiplier.
        return int(round(rec.chars * self.audio_multiplier))
