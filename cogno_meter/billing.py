"""
cogno_meter.billing — aggregate usage and compute the bill.

The overage math is ported from the parent ``cogno/core/db_billing.py`` (the
pure calculation, not the DB/Stripe layer): everything counts toward the monthly
token total (local models included, at zero provider cost); the customer is
charged the **plan's** per-million ``overage_price`` on tokens beyond the
allowance — independent of whose API key ran the model (BYOK is not exempt).

The provider cost (price book) is aggregated alongside, purely as transparency.
"""

from __future__ import annotations

from typing import Iterable

from cogno_meter.pricing import PriceBook
from cogno_meter.types import Bill, Plan, UsageRecord


def total_billable_tokens(records: Iterable[UsageRecord], book: PriceBook) -> int:
    return sum(book.billable_tokens(r) for r in records)


def total_provider_cost_usd(records: Iterable[UsageRecord], book: PriceBook) -> float:
    return sum(book.usage_cost_usd(r) for r in records)


def compute_bill(
    *,
    plan: Plan,
    billable_tokens: int,
    provider_cost_usd: float,
    book: PriceBook,
) -> Bill:
    """Apply the plan's allowance + per-million overage to a period's totals."""
    overage_tokens = max(0, billable_tokens - plan.monthly_token_limit)
    overage_cost = (overage_tokens / 1_000_000) * plan.overage_price
    total = plan.base_price + overage_cost
    return Bill(
        total_tokens=billable_tokens,
        allowance=plan.monthly_token_limit,
        overage_tokens=overage_tokens,
        overage_cost=round(overage_cost, 6),
        base_price=plan.base_price,
        total=round(total, 6),
        provider_cost_usd=round(provider_cost_usd, 6),
        provider_cost_brl=round(provider_cost_usd * book.usd_brl_rate, 6),
    )


def meter(records: Iterable[UsageRecord], *, plan: Plan, book: PriceBook) -> Bill:
    """End-to-end: aggregate a period's usage records into a :class:`Bill`."""
    records = list(records)
    return compute_bill(
        plan=plan,
        billable_tokens=total_billable_tokens(records, book),
        provider_cost_usd=total_provider_cost_usd(records, book),
        book=book,
    )
