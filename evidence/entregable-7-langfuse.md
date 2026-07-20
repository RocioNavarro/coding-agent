# Evidencia — Entregable #7: traza completa en Langfuse

**Link público de la traza real** (proyecto `coding-agent-tp`, Langfuse Cloud):

https://cloud.langfuse.com/project/cmrtdlml403ejad0ddjaypy0i/traces?traceTab=log&peek=3e0152d1c882876c&observation=3e0152d1c882876c&traceId=305dbfe787d6a2d92281f090ae027f88&timestamp=2026-07-20T15%3A58%3A17.140Z

Corresponde a una corrida real completada (`status: completed`) contra el workspace de
PrintScript, con la jerarquía completa registrada: `orchestrated-task` (raíz) →
`explorer`/`researcher` (agentes) → `llm-completion` (llamadas reales al LLM) →
`rag-retrieval` (consulta al índice RAG) → `orchestrated-task-finished` (resultado, con
`sources`, `iterations` y `error_count`).

**Nota sobre el campo "Output" en Langfuse**: aparece como `undefined` en todos los eventos
de esta integración, incluidos los exitosos — es una simplificación de
`core/observability.py`, que vuelca todo el contenido relevante (`result`, `sources`,
`status`, etc.) en el campo `input` de cada observación y nunca completa `output`. No indica
ningún fallo; el contenido real está en `input`/`result`.
