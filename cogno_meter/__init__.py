"""
cogno-meter — usage metering + pricing/overage for the Cogno stack.

Cross-modality usage (LLM/embedding tokens + STT/TTS characters) is normalized to
**billable tokens** and billed against a plan: a monthly token allowance plus a
per-million **overage** at the plan's rate. Local models cost nothing to run but
still consume the allowance; audio is char-metered with a multiplier so it costs
more. The real upstream **provider cost** is computed alongside, as transparency
only (with BYOK the client pays the provider directly).

Pure code, zero dependencies, zero I/O: the host owns the plan values, the price
book values, the accumulated-usage persistence, and invoicing. Adapted from the
parent cogno's pricing/metering/billing.
"""

from cogno_meter.types import Bill, Modality, Plan, UsageRecord
from cogno_meter.pricing import (
    DEFAULT_AUDIO_MULTIPLIER,
    DEFAULT_RATES,
    DEFAULT_USD_BRL_RATE,
    PriceBook,
)
from cogno_meter.billing import (
    compute_bill,
    meter,
    total_billable_tokens,
    total_provider_cost_usd,
)
from cogno_meter.budget import (
    BLOCKING_STATUSES,
    WARN_THRESHOLD,
    BudgetInputs,
    BudgetReason,
    BudgetVerdict,
    evaluate_budget,
)

__all__ = [
    "Modality",
    "UsageRecord",
    "Plan",
    "Bill",
    "PriceBook",
    "DEFAULT_RATES",
    "DEFAULT_USD_BRL_RATE",
    "DEFAULT_AUDIO_MULTIPLIER",
    "meter",
    "compute_bill",
    "total_billable_tokens",
    "total_provider_cost_usd",
    "evaluate_budget",
    "BudgetInputs",
    "BudgetVerdict",
    "BudgetReason",
    "BLOCKING_STATUSES",
    "WARN_THRESHOLD",
]
