# Logging — convenção desta lib

Esta biblioteca **emite** logs; o **host configura** (handlers, formato, nível,
contexto de tenant). Regras:

1. Use `logging.getLogger(__name__)` no topo do módulo. Nada de handlers,
   formatters, `basicConfig` ou um `get_logger` próprio.
2. Mensagem = só o fato de domínio, em `key=value`, sempre lazy:
   `logger.debug("event=meter total_tokens=%d overage_tokens=%d", t, o)`.
   NÃO coloque tenant_id / timestamp / channel na mensagem — o host injeta
   via contextvars + Filter no root logger (carimbado em todo LogRecord).
3. Níveis:
   - **ERROR**  → nunca aqui; erro fatal vira exceção e propaga (host loga ERROR).
   - **WARNING**→ condição recuperada/tratada (fallback, parse coercion, verify falho).
   - **INFO**   → marco caro e raro; NÃO happy-path por request.
   - **DEBUG**  → trace de fidelidade total (cálculos). DEV-ONLY, jamais ligado
                  em produção multi-tenant. Redija secrets (apikey).
4. Controle de nível é por pacote: `logging.getLogger("cogno_meter").setLevel(...)`.

O host anexa o handler (TenantFilter + JsonFormatter) ao root logger real;
veja `cogno/core/logging.py` no host como referência.

## Nota específica do cogno-meter

São **funções puras** (recebem dados, retornam `Bill`/custo). Não há WARNING
(a decisão de bloquear por orçamento é do host) nem INFO por chamada (quem chama
loga o resultado retornado). Apenas **DEBUG** para inspecionar o cálculo:
resolução de modelo na tabela (exact/fuzzy/default) e custo por modalidade.
