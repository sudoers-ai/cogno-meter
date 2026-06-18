# Host Integration Guide

How to wire `cogno-meter` into a real application. The library ships the
**pricing + overage math**; the host owns the **values, the persistence, and the
invoicing**. This is the human-facing companion to `examples/host_min.py`.

> TL;DR — feed `meter(...)` a period's `UsageRecord`s (tokens from cogno-synapse,
> chars from cogno-vox), a `Plan`, and a `PriceBook`. It returns a `Bill`: what to
> charge (plan allowance + per-million overage) plus the provider cost for
> transparency.

---

## 1. The boundary

| Concern | Owner |
| --- | --- |
| Price book resolution, per-modality cost, USD→currency, audio char×multiplier, overage math | **meter** |
| Plan/allowance/overage-price **values**; price book **values** | **host** |
| Accumulated-usage persistence (the token ledger), billing period boundaries | **host** |
| Per-tenant model overrides, BYOK key storage, invoicing/Stripe | **host** |

`meter` is pure (no I/O) and consumes plain `UsageRecord`s — it never imports
`cogno-synapse` or `cogno-vox`.

---

## 2. Build the inputs

**Price book** — the lib ships an illustrative `DEFAULT_RATES` seed; inject your
own (prices drift; keep them host-side):

```python
import yaml
from cogno_meter import PriceBook

book = PriceBook.from_mapping(yaml.safe_load(open("pricing.yaml")))
# mapping keys: llm/embedding/stt/tts + usd_brl_rate + audio_multiplier
```

Resolution is exact → fuzzy version-prefix → provider `_default` (self-hosted = 0).
Units: LLM/embedding USD per 1M tokens, STT USD per minute, TTS USD per 1M chars.

**Plan** — host-owned (Stripe/DB):

```python
from cogno_meter import Plan
plan = Plan(name="Premium", monthly_token_limit=5_000_000, overage_price=0.04, base_price=99.0)
```

**Usage** — translate your producers into `UsageRecord`s:

```python
from cogno_meter import UsageRecord, Modality

# from a cogno-synapse / cogno-anima StageMetrics:
UsageRecord(Modality.LLM, sm.model, tokens_in=sm.tokens_in, tokens_out=sm.tokens_out)
# from a cogno-vox result (audio is char-metered):
UsageRecord(Modality.STT, "local:faster-whisper", chars=transcription.chars)
UsageRecord(Modality.TTS, "local:kokoro", chars=synthesis.chars)
```

---

## 3. Compute the bill

```python
from cogno_meter import meter
bill = meter(period_records, plan=plan, book=book)
```

`Bill` fields:

- `total_tokens` — billable tokens for the period (audio already `chars × multiplier`).
- `overage_tokens` / `overage_cost` — tokens beyond the allowance, charged at the
  **plan's** per-million rate (the platform charge).
- `total` — `base_price + overage_cost` (what the customer pays).
- `provider_cost_usd` / `provider_cost_brl` — the real upstream cost, **transparency
  only** (with BYOK the client pays the provider directly; this is not the charge).

---

## 4. The model in one paragraph

Everything counts as billable tokens. A **local** model costs nothing to run but
still draws down the monthly allowance. Audio is **char-metered** and multiplied
(default `2×`) so it costs more. Beyond the allowance, the customer pays the
**plan's** override per million tokens — cheap, because the platform runs local
models — and **BYOK is not exempt** (over the limit you pay the override even with
your own key). The provider price book yields a separate number used only so the
client can see external-LLM spend.

---

## 5. Accumulation across a period

`meter` is stateless: you decide what "the period" is. Two common shapes:

1. **Recompute** — persist each turn's `UsageRecord`s (your token ledger) and call
   `meter(all_records_this_month, ...)` for the current bill.
2. **Running totals** — keep the period's `total_billable_tokens` and
   `provider_cost_usd` accumulators (use `total_billable_tokens(...)` /
   `total_provider_cost_usd(...)` per turn) and call `compute_bill(...)` at billing
   time.

Either way the persistence (the token ledger) and the period boundaries are the
host's — `cogno-meter` only does the arithmetic.
