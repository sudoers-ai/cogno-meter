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
        "openai:gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "openai:gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "openai:gpt-4o": {"input": 2.50, "output": 10.00},
        "openai:gpt-4.1": {"input": 2.00, "output": 8.00},
        "anthropic:claude-haiku-4.5": {"input": 1.00, "output": 5.00},
        "anthropic:claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
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

    # ── rate resolution (ported: exact → fuzzy prefix → _default) ──────
    @staticmethod
    def _resolve(table: dict, model: str):
        if model in table:
            logger.debug("event=rate_resolve model=%s match=exact", model)
            return table[model]
        for key, rate in table.items():
            if key == "_default":
                continue
            if model.startswith(key):  # 'gpt-4o-mini-2024-07-18' → 'openai:gpt-4o-mini'
                logger.debug("event=rate_resolve model=%s match=fuzzy key=%s", model, key)
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
