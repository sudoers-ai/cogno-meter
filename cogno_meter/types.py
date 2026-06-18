"""
cogno_meter.types — usage records, plans, and the computed bill.

All cross-modality usage is normalized to **billable tokens** (the unit the
monthly allowance and overage are denominated in). Audio (STT/TTS) is metered by
**characters**, multiplied by a configurable factor so it draws down the
allowance faster (see ``PriceBook.audio_multiplier``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Modality(str, Enum):
    LLM = "llm"
    EMBEDDING = "embedding"
    STT = "stt"
    TTS = "tts"


@dataclass
class UsageRecord:
    """One unit of metered usage (a stage call, a TTS synthesis, …).

    LLM/embedding usage carries token counts; audio (STT/TTS) carries ``chars``
    (always char-metered) and, optionally, ``minutes`` of audio for the
    provider-cost transparency figure (STT is priced per minute upstream).
    """

    modality: Modality
    model: str                 # "provider:model" (the model that actually ran)
    tokens_in: int = 0
    tokens_out: int = 0
    chars: int = 0             # audio: STT transcribed / TTS input characters
    minutes: float = 0.0       # audio duration (optional; STT provider-cost only)


@dataclass
class Plan:
    """A tenant's plan. Values are host-owned (Stripe/DB) — these are just the
    parameters the overage math needs."""

    name: str
    monthly_token_limit: int   # free billable tokens included per period
    overage_price: float       # charge per 1M tokens over the limit (plan currency)
    base_price: float = 0.0    # subscription base for the period (plan currency)


@dataclass
class Bill:
    """The computed bill for a period. ``total`` is what the platform charges the
    customer; ``provider_cost_*`` is transparency only (real upstream cost, paid
    by the client via BYOK for external models — NOT the platform charge)."""

    total_tokens: int          # billable tokens (incl. audio chars × multiplier)
    allowance: int
    overage_tokens: int
    overage_cost: float        # plan currency
    base_price: float
    total: float               # base_price + overage_cost (plan currency)
    provider_cost_usd: float   # transparency
    provider_cost_brl: float   # transparency (usd × usd_brl_rate)
