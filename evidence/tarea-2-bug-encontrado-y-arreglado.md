# Evidencia — Tarea 2: bug real encontrado, arreglado y verificado en vivo

**Fecha:** 2026-07-20
**Pedido:** "Explicame cómo PrintScript maneja la concurrencia y el multithreading en su
intérprete, citando la especificación del lenguaje."
**Modelo:** `gpt-5-nano` (OpenAI Responses API, real)
**Traza en Langfuse:** `task_id` = `d71c9989-787f-47e6-9499-0885eada2255` (proyecto
`coding-agent-tp` en Langfuse, span raíz `orchestrated-task`, 2026-07-20 15:56 UTC) —
buscar por ese `task_id` en Tracing para ver la traza completa en vivo.

Esta tarea documenta un ciclo completo de detección → diagnóstico → arreglo → verificación
de un bug real de arquitectura, encontrado en vivo mientras se buscaba la segunda corrida
del entregable.

## Qué falló (antes del arreglo)

Al pedirle al agente algo más específico y menos cubierto por el contexto inicial que
Explorer arma (arquitectura general), el modelo decidió pedir **8 tool calls** (`find_files`
buscando "concurrencia", "multithreading", "interpreter", "language specification", etc.)
en vez de responder directo con JSON.

**El problema real**: `BaseAgent.run()` le ofrecía tools al LLM (`allowed_tools` +
schemas), pero **nunca las ejecutaba**. Sólo las validaba contra la allowlist y después
intentaba parsear `response.text` como JSON — que quedaba vacío porque el modelo terminó su
turno pidiendo tools, no con texto. Resultado: `AgentExecutionError("El agente 'explorer'
devolvió una respuesta JSON inválida.")`, reproducible de forma determinista con esta
pregunta (falló 4 veces seguidas, siempre por el mismo motivo, confirmado inspeccionando
`response.tool_calls` directamente).

## Diagnóstico

Se instrumentó `OpenAILLMClient.complete()` para loguear cada llamada real (tamaño del
pedido, tools ofrecidas, éxito/fallo), confirmando que **el mismo código, en el mismo
proceso**, sí ejecutaba llamadas reales a OpenAI — descartando que fuera un problema de
cuota o de la API. La inspección directa de `response.tool_calls` confirmó las 8 tool
calls pedidas y nunca atendidas.

## Arreglo aplicado

`agents/base.py`: se agregó `BaseAgent._complete_until_text()`, que ejecuta de verdad las
tool calls que pida el modelo (usando el mismo patrón que ya existía en
`core/harness.py` para el loop principal), acotado a `_MAX_TOOL_ROUNDS = 3` rondas antes de
forzar una respuesta final sin tools disponibles. Se mantiene además `_MAX_OUTPUT_ATTEMPTS
= 2` como red de seguridad para salidas mal formadas que no involucren tools.

Validado:
- Suite completa (389 tests) sigue en verde después del cambio.
- La misma pregunta que fallaba 4/4 veces ahora **completa** de forma consistente.

## Resultado con el arreglo aplicado (evidencia real)

**Estado:** `completed` — **Agentes:** `explorer`, `researcher`

> **Explorer**: La especificación de PrintScript no define concurrencia ni multithreading;
> se centra en un procesamiento streaming incremental entre las etapas de lexer, parser e
> interpreter, y no describe primitivas del lenguaje para concurrencia.

> **Researcher**: [...] La evidencia sugiere un procesamiento secuencial con streaming
> incremental entre lexer, parser e interpreter y no documenta primitivas de concurrencia en
> el lenguaje o en el runtime. No hay fragmentos que describan usos de hilos, corutinas o
> mecanismos asíncronos en el intérprete, y la especificación mencionada indica
> explícitamente ausencia de primitivas de concurrencia.

### Trazabilidad de fuentes

- **RAG**: 2 chunks de `kotlinlang.org/docs/basic-syntax.html` recuperados y usados (score
  0.383 y 0.315), evidencia evaluada como suficiente (confianza 0.67) — sin fallback web.
- **Memoria persistida**: reutilizó 5 `previous_tasks`/`session_summaries` de corridas
  anteriores.
- **Repositorio**: citó archivos reales (incluida la especificación
  `docs/printscript-language-spec.md`) para fundamentar la ausencia de concurrencia.

## Qué se observa / conclusión

- El sistema **no alucinó** una respuesta sobre concurrencia inexistente — reportó
  honestamente que la especificación no la cubre, citando evidencia concreta. Esto es
  exactamente el comportamiento de "reconocer cuándo no tiene evidencia suficiente" que pide
  la consigna, aunque terminó en `completed` (con una conclusión honesta de ausencia) en vez
  de en `blocked`.
- El bug encontrado y arreglado es un ejemplo real de "qué falló" para la reflexión del
  entregable #8: un gap de arquitectura genuino (tools ofrecidas sin loop de ejecución),
  no aleatoriedad del modelo — y se resolvió con una implementación acotada y sin romper
  ningún test existente.
