"""Unit tests for the pure budget-decision core (evaluate_budget)."""

from cogno_meter import BudgetInputs, BudgetReason, evaluate_budget


def test_all_unset_is_allowed():
    v = evaluate_budget(BudgetInputs())
    assert v.blocked is False and v.reason is None and v.warnings == []


def test_zero_or_none_budget_is_unset_not_block():
    # a budget of 0 / None means "no limit" — spend of 0 must NOT block
    assert evaluate_budget(BudgetInputs(tenant_daily_budget=0.0, tenant_daily_spend=0.0)).blocked \
        is False
    assert evaluate_budget(BudgetInputs(tenant_daily_budget=None, tenant_daily_spend=999)).blocked \
        is False


def test_tenant_daily_blocks_at_100pct():
    v = evaluate_budget(BudgetInputs(tenant_daily_budget=10.0, tenant_daily_spend=10.0))
    assert v.blocked and v.reason is BudgetReason.TENANT_DAILY


def test_identity_daily_blocks_and_tenant_takes_precedence():
    # identity over its budget → blocked with the identity reason
    v = evaluate_budget(BudgetInputs(identity_daily_budget=5.0, identity_daily_spend=6.0))
    assert v.blocked and v.reason is BudgetReason.IDENTITY_DAILY
    # when BOTH are over, tenant is checked first (order fidelity to the parent)
    v2 = evaluate_budget(BudgetInputs(tenant_daily_budget=10.0, tenant_daily_spend=10.0,
                                      identity_daily_budget=5.0, identity_daily_spend=6.0))
    assert v2.reason is BudgetReason.TENANT_DAILY


def test_80pct_warns_without_blocking():
    v = evaluate_budget(BudgetInputs(tenant_daily_budget=10.0, tenant_daily_spend=8.0))
    assert v.blocked is False and "tenant_daily_budget:80pct" in v.warnings


def test_subscription_status_and_expiry():
    assert evaluate_budget(BudgetInputs(subscription_status="canceled")).reason \
        is BudgetReason.SUBSCRIPTION_INACTIVE
    assert evaluate_budget(BudgetInputs(subscription_status="past_due")).reason \
        is BudgetReason.SUBSCRIPTION_INACTIVE
    assert evaluate_budget(BudgetInputs(subscription_status="active",
                                        subscription_expired=True)).reason \
        is BudgetReason.SUBSCRIPTION_EXPIRED
    # active + not expired → not blocked by this layer
    assert evaluate_budget(BudgetInputs(subscription_status="active")).blocked is False
    # unset/"inactive"/free default runs
    assert evaluate_budget(BudgetInputs(subscription_status="inactive")).blocked is False


def test_monthly_tokens_overage_aware():
    # over the limit, no overage → hard stop
    v = evaluate_budget(BudgetInputs(monthly_token_limit=1000, monthly_tokens_used=1000,
                                     overage_price=0.0))
    assert v.blocked and v.reason is BudgetReason.MONTHLY_TOKENS
    # over the limit WITH overage → keeps running (billed), just a warning
    v2 = evaluate_budget(BudgetInputs(monthly_token_limit=1000, monthly_tokens_used=1500,
                                      overage_price=0.05))
    assert v2.blocked is False and "monthly_tokens:80pct" in v2.warnings


def test_plan_limits_and_call_limit_pass_through():
    caps = {"max_fc_steps": 5, "allow_escalation": False}
    v = evaluate_budget(BudgetInputs(plan_limits=caps, call_limit=0.5))
    assert v.plan_limits == caps and v.call_limit == 0.5
    # they pass through even on a block, so the host can still record them
    vb = evaluate_budget(BudgetInputs(tenant_daily_budget=1.0, tenant_daily_spend=2.0,
                                      plan_limits=caps, call_limit=0.5))
    assert vb.blocked and vb.plan_limits == caps and vb.call_limit == 0.5
