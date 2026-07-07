"""Budget/quota decision — the pure gate the host wires to its stores.

The parent's ``BudgetGuard`` (``cogno/core/budget_guard.py``) mixed store I/O, currency, dates and
pt-BR messages into one class. Here the **decision** is extracted as pure policy: the host fetches
the numbers (daily spend, monthly usage, subscription state), converts them to the plan currency,
parses expiry, and passes them in; this module applies the layered budget policy and returns a
:class:`BudgetVerdict`. Zero I/O, no dates, no FX, no locale — a blocked verdict names the
:class:`BudgetReason`, and the host renders the localized message.

Layers (order): tenant daily budget → identity daily budget → subscription status/expiry →
monthly token quota (hard-stop ONLY when the plan has no overage). Each daily layer warns at 80%.
A budget of ``None`` / ``<= 0`` is *unset* (no limit) — never blocks. Never raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# Subscription states that block a turn (an active or unset/"inactive" free plan runs). The host
# may override via ``BudgetInputs.blocking_statuses``.
BLOCKING_STATUSES: "frozenset[str]" = frozenset({"canceled", "past_due", "suspended"})
# Soft-warning threshold (fraction of a budget/quota that raises a warning, not a block).
WARN_THRESHOLD = 0.8


class BudgetReason(str, Enum):
    """Why a turn was blocked — the host maps this to a localized user message."""

    TENANT_DAILY = "tenant_daily_budget"
    IDENTITY_DAILY = "identity_daily_budget"
    SUBSCRIPTION_INACTIVE = "subscription_inactive"
    SUBSCRIPTION_EXPIRED = "subscription_expired"
    MONTHLY_TOKENS = "monthly_tokens"


@dataclass(frozen=True)
class BudgetInputs:
    """Everything pre-fetched + normalized by the host. Spends are in the SAME unit as the budgets
    (the host applies any FX before building this); ``subscription_expired`` is pre-computed (the
    host parses the date). A ``None``/non-positive budget or a 0 monthly limit means *unset*."""

    tenant_daily_budget: Optional[float] = None
    tenant_daily_spend: float = 0.0
    identity_daily_budget: Optional[float] = None
    identity_daily_spend: float = 0.0
    subscription_status: Optional[str] = None    # None → no subscription (skip the layer)
    subscription_expired: bool = False
    monthly_token_limit: int = 0
    monthly_tokens_used: int = 0
    overage_price: float = 0.0                    # >0 → paid plan keeps running over the limit
    plan_limits: dict = field(default_factory=dict)   # loop-depth caps to pass through
    call_limit: Optional[float] = None                # per-call ceiling to pass through
    blocking_statuses: "frozenset[str]" = BLOCKING_STATUSES


@dataclass
class BudgetVerdict:
    """The decision. ``plan_limits``/``call_limit`` pass through for the host to apply on an
    allowed turn; ``warnings`` carries 80%-threshold notices (``"<reason>:80pct"``)."""

    blocked: bool = False
    reason: Optional[BudgetReason] = None
    plan_limits: dict = field(default_factory=dict)
    call_limit: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


def _positive(v: Optional[float]) -> bool:
    return v is not None and v > 0


def evaluate_budget(inp: BudgetInputs) -> BudgetVerdict:
    """Apply the layered budget policy → a :class:`BudgetVerdict`. Currency/locale/time-agnostic."""
    warnings: List[str] = []

    def _block(reason: BudgetReason) -> BudgetVerdict:
        return BudgetVerdict(blocked=True, reason=reason, plan_limits=inp.plan_limits,
                             call_limit=inp.call_limit, warnings=warnings)

    # 1) tenant daily budget, then 2) identity daily budget — block at 100%, warn at 80%.
    for budget, spend, reason in (
        (inp.tenant_daily_budget, inp.tenant_daily_spend, BudgetReason.TENANT_DAILY),
        (inp.identity_daily_budget, inp.identity_daily_spend, BudgetReason.IDENTITY_DAILY),
    ):
        if not _positive(budget):
            continue
        if spend >= budget:            # type: ignore[operator]  (guarded by _positive)
            return _block(reason)
        if spend >= budget * WARN_THRESHOLD:   # type: ignore[operator]
            warnings.append(f"{reason.value}:80pct")

    # 3) subscription status / expiry (only when a subscription is present)
    if inp.subscription_status is not None:
        if inp.subscription_status.lower() in inp.blocking_statuses:
            return _block(BudgetReason.SUBSCRIPTION_INACTIVE)
        if inp.subscription_expired:
            return _block(BudgetReason.SUBSCRIPTION_EXPIRED)

    # 4) monthly token quota — hard-stop ONLY for a plan with no overage (Free); paid plans run on
    if inp.monthly_token_limit > 0:
        used, limit = inp.monthly_tokens_used, inp.monthly_token_limit
        if used >= limit and inp.overage_price <= 0:
            return _block(BudgetReason.MONTHLY_TOKENS)
        if used >= limit * WARN_THRESHOLD:
            warnings.append("monthly_tokens:80pct")

    return BudgetVerdict(blocked=False, reason=None, plan_limits=inp.plan_limits,
                         call_limit=inp.call_limit, warnings=warnings)
