"""
Minimal host wiring for cogno-meter: turn a period's usage into a bill.

Standalone — no network, no deps:  python examples/host_min.py

It shows the host's job:
  1. build a PriceBook (default seed, or your own mapping from YAML/JSON/DB),
  2. define the tenant's Plan (allowance + per-million overage),
  3. collect the period's UsageRecords (from cogno-synapse tokens + cogno-vox chars),
  4. call meter(...) → Bill (what to charge + the transparency provider cost).
"""

from __future__ import annotations

from cogno_meter import Modality, Plan, PriceBook, UsageRecord, meter


def main() -> None:
    # 1) Price book — the default seed here; a host injects its own:
    #    book = PriceBook.from_mapping(yaml.safe_load(open("pricing.yaml")))
    book = PriceBook.default()                       # audio_multiplier defaults to 2x

    # 2) The tenant's plan (host-owned values — Stripe/DB):
    plan = Plan(name="Básico", monthly_token_limit=500_000, overage_price=0.05, base_price=20.0)

    # 3) The period's usage. In a real host these come from:
    #    - cogno-synapse StageMetrics → tokens_in/out per stage
    #    - cogno-vox TranscriptionResult.chars / SynthesisResult.chars → audio chars
    records = [
        UsageRecord(Modality.LLM, "ollama:mistral:latest", tokens_in=900_000, tokens_out=180_000),
        UsageRecord(Modality.LLM, "openai:gpt-4.1-mini", tokens_in=200_000, tokens_out=100_000),
        UsageRecord(Modality.STT, "local:faster-whisper", chars=20_000),   # × 2 = 40k tokens
        UsageRecord(Modality.TTS, "local:kokoro", chars=30_000),           # × 2 = 60k tokens
    ]

    # 4) Compute the bill:
    bill = meter(records, plan=plan, book=book)

    print(f"plano            : {plan.name}  (incluso {plan.monthly_token_limit:,} tokens)")
    print(f"tokens cobráveis : {bill.total_tokens:,}  (áudio já × {book.audio_multiplier:g})")
    print(f"excedente        : {bill.overage_tokens:,} tokens")
    print(f"override (cobra)  : R$ {bill.overage_cost:.4f}")
    print(f"TOTAL ao cliente  : R$ {bill.total:.2f}  (mensalidade {plan.base_price:.2f} + override)")
    print(f"custo fornecedor  : US$ {bill.provider_cost_usd:.4f} / R$ {bill.provider_cost_brl:.4f}"
          f"   ← transparência (BYOK paga direto ao fornecedor; não é a cobrança da plataforma)")


if __name__ == "__main__":
    main()
