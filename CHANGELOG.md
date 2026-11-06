# Changelog

## 0.1.1 — 2026-11-06 — release to PyPI of the changes merged since 0.1.0

- fix(pricing): o chão que nunca podia disparar — o livro guarda o nome NU, e o que ninguém sabe cobrar passa a ser NOMEADO (#18)
- fix(pricing): gpt-5.6 lê-se na célula certa — Standard, contexto curto — e o silêncio sobre a taxa de cache passa a ser uma decisão escrita (#17)
- feat(pricing): o desconto de cache do provider entra no CUSTO — e o modelo que não foi medido paga cheio (#15)
- fix(metering): enforce parity for uncatalogued modalities and fix billable_tokens fallthrough
- feat(pricing): gpt-4o-mini-tts ganha taxa DECLARADA (era piso silencioso + aviso por turno) (#14)
- fix(pricing): an uncatalogued cloud model metered as FREE — on every provider
- fix(pricing): grok rates were over-reporting up to 6x; keep the retired ids, reprice them

## 0.1.0 — 2026-07-25

First public release on PyPI.

Usage metering + pricing/overage for the Cogno cognitive pipeline — cross-modality (LLM/embedding tokens + STT/TTS chars) billable-token accounting against a plan allowance
