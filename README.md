# cogno-meter

**Usage metering + pricing/overage for the [Cogno](https://github.com/sudoers-ai/cogno-anima) cognitive pipeline** — cross-modality billable-token accounting against a plan allowance.

`cogno-meter` turns raw usage (LLM/embedding **tokens** + STT/TTS **characters**) into a bill. Pure code, zero dependencies, zero I/O — the host owns the plan values, the price book values, the accumulated-usage persistence, and invoicing. Adapted clean-room from the parent Cogno SaaS's `pricing` / `metering` / `db_billing`.

> Status: **alpha** — the pricing + overage core and the unit suite are in place.

## The billing model

- **Everything counts as billable tokens.** A **local** model costs nothing to run, but its tokens still draw down the monthly allowance.
- **Audio is char-metered, with a multiplier.** STT/TTS usage is measured in characters and multiplied (default `2×`) so audio is more expensive per the allowance.
- **Overage is per-plan, not per-model.** Beyond the monthly allowance, the customer is charged the **plan's** rate per million tokens — cheap, because the platform runs local models. **BYOK is not exempt**: over the limit you pay the override even with your own API key.
- **Provider cost is transparency only.** The real upstream cost (per-model price book) is computed alongside so the client can track external-LLM spend; for external models under BYOK the client pays the provider directly. It is **not** the platform charge.

## Use

```python
from cogno_meter import meter, Plan, PriceBook, UsageRecord, Modality

plan = Plan(name="Básico", monthly_token_limit=500_000, overage_price=0.05, base_price=20.0)
book = PriceBook.default()                 # or PriceBook.from_mapping(host_pricing_dict)

records = [                                 # a period's usage (from synapse + vox)
    UsageRecord(Modality.LLM, "ollama:mistral:latest", tokens_in=1_000_000, tokens_out=200_000),
    UsageRecord(Modality.TTS, "local:kokoro", chars=50_000),     # 50k × 2 = 100k billable tokens
]

bill = meter(records, plan=plan, book=book)
# bill.total_tokens=1_300_000  overage_tokens=800_000  overage_cost=0.04
# bill.total=20.04  provider_cost_brl=0.0   (all local → free upstream)
```

External, paid model (transparency cost appears, charge stays plan-based):

```python
records = [UsageRecord(Modality.LLM, "openai:gpt-4.1-mini", tokens_in=1_000_000, tokens_out=500_000)]
bill = meter(records, plan=plan, book=book)
# overage_cost=0.05  total=20.05  (platform charge — plan rate)
# provider_cost_usd=1.20  provider_cost_brl=6.84  (transparency — client pays OpenAI via BYOK)
```

## The price book

`PriceBook` resolves a model's rate by exact match → fuzzy version-prefix → provider `_default` (self-hosted = 0), and converts USD→currency. The lib ships an **illustrative** seed (`DEFAULT_RATES`); a host injects its own via `PriceBook.from_mapping(...)` (load your YAML/JSON/DB and pass a plain mapping — no YAML dependency here). Units: LLM/embedding USD per 1M tokens, STT USD per minute, TTS USD per 1M characters.

## Boundary

| Concern | Owner |
| --- | --- |
| Price book resolution, per-modality cost, audio char×multiplier, overage math | **cogno-meter** |
| Plan/allowance/overage-price values, price book values | **host** |
| Accumulated-usage persistence (the token ledger), invoicing, Stripe | **host** |
| Per-tenant model overrides, BYOK key storage | **host** |

`meter` consumes usage produced by `cogno-synapse` (tokens) and `cogno-vox` (audio chars); it never imports them — usage is handed in as plain `UsageRecord`s.

## Test

```bash
pip install -e ".[dev]"
pytest tests/unit -q
```
