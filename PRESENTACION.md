# TP Final — Coding Agent Avanzado
> Presentación oral (10 min) — versión final, TP completo
> Documento de apoyo — no versionado (ver .gitignore)

✅ **Los 8 entregables de la consigna están cubiertos.** Código completo (389 tests en
verde), README, caso de uso con objetivo y criterio de éxito, arquitectura documentada, RAG
documentado, 2 tareas reales ejecutadas con evidencia (`evidence/`), traza real en Langfuse,
y reflexión honesta con hallazgos reales (dos bugs encontrados y arreglados en vivo).

---

## Slide 1 — Consigna del TP y punto de partida

- Objetivo: evolucionar el coding agent de clase hacia un **sistema multiagente**, sin
  frameworks de orquestación (LangChain/LangGraph/CrewAI/AutoGen), combinando tools locales +
  RAG + memoria persistente + subagentes especializados + políticas de seguridad +
  observabilidad.
- Punto de partida conservado de la Parte 1: el harness (`core/harness.py`) y las tools base
  (`read_file`, `write_file`, `list_files`, `run_command`, `web_search`) — nunca se tocaron.
- Todo lo nuevo se construyó **por composición**, en una capa aparte (`agents/`, `rag/`,
  `security/policy_engine.py`), sin reemplazar el bootstrap interactivo de `main.py`.

*Tiempo sugerido: 1 min*

---

## Slide 2 — Caso de uso concreto: PrintScript, ya ejecutado dos veces de verdad

- **Objetivo elegido**: "Analizar un repositorio desconocido" (uno de los dos que ofrece la
  consigna, completo por sí solo) — un checkout externo real de PrintScript (Kotlin/Gradle:
  lexer, parser, interpreter, formatter, linter, runner, cli), referenciado como workspace
  externo, sin copiarlo dentro de este repo.
- **Alcance acotado a propósito**: sólo Explorer + Researcher, sin Implementer/Tester/
  Reviewer — decisión consciente para no depender de que Gradle/Kotlin compilen en la máquina
  de evaluación, sin sacrificar ningún requisito obligatorio.
- **Ya corrido dos veces, con LLM real (`gpt-5-nano`) y Langfuse real activo**:
  - **Tarea 1**: "Explicame la arquitectura, módulos y diseño de PrintScript" → `completed`,
    citando RAG real (spec de PrintScript + docs de Kotlin) y memoria persistida de corridas
    previas.
  - **Tarea 2**: una pregunta sobre concurrencia/multithreading que la spec no cubre →
    el agente **no alucinó**: reportó honestamente la ausencia de evidencia, citando la
    especificación real.
- Evidencia completa en `evidence/` (2 tareas + traza de Langfuse + reflexión).

*Tiempo sugerido: 1.5 min*

---

## Slide 3 — Arquitectura de agentes: agente principal

- **`agents/orchestrator.py` → `MainAgent`**: recibe el pedido, mantiene el `TaskState` y
  coordina subagentes.
- Flujo: clasificar tarea → consultar memoria → explorar (completo o incremental) →
  investigar si aplica → generar plan → aprobación humana → operaciones planificadas →
  **preflight de políticas** → evidencia → implementación → testing → review → resultado.
- Estados de salida explícitos: `completed`, `rejected`, `blocked`, `max_iterations`.

*Tiempo sugerido: 1 min*

---

## Slide 4 — Los 5 subagentes pedidos por la consigna

| Subagente | Archivo | Responsabilidad |
|---|---|---|
| Explorer | `agents/explorer.py` | Estructura, tecnologías, convenciones — con evidencia, no supuestos |
| Researcher | `agents/researcher.py` | Memoria → RAG → evaluación de suficiencia → web como fallback |
| Implementer | `agents/implementer.py` | Propone reemplazos exactos; sólo aplica con plan + evidencia `sufficient` |
| Tester | `agents/tester.py` | Ejecuta comandos respaldados, aplica `PolicyEngine`, registra progreso |
| Reviewer | `agents/reviewer.py` | Aprueba, pide cambios, bloquea o marca evidencia insuficiente |

- Todos heredan de `agents/base.py::BaseAgent`, con `allowed_tools` propio (no comparten
  permisos) y dependencias 100% inyectables (LLM, memoria, RAG, web y observabilidad
  pueden ser fakes en tests).

*Tiempo sugerido: 1 min*

---

## Slide 5 — RAG: pipeline completo, con contenido real en dos niveles

- **Chunking** (`rag/processing.py`): reconoce encabezados Markdown; se agregó un
  `HtmlDocumentParser` propio para extraer texto limpio de páginas HTML reales sin ruido de
  etiquetas.
- **Embeddings** (`rag/embeddings.py`): interfaz intercambiable; `HashEmbeddingProvider`
  local y determinista.
- **Vector store** (`rag/vector_store.py::JsonVectorStore`): índice versionado, reindexa sólo
  lo que cambió por hash.
- **Contenido real indexado, en dos niveles**: (1) la especificación de PrintScript —el
  lenguaje que se implementa— entregada por la cátedra; (2) documentación oficial de
  Kotlin —el lenguaje de implementación— descargada en vivo de kotlinlang.org. 68 chunks
  reales, verificados.
- **Recuperación con fallback web**: memoria → RAG → evaluación de suficiencia → web sólo si
  falta evidencia, priorizando fuentes oficiales.

*Tiempo sugerido: 1.5 min*

---

## Slide 6 — Trazabilidad, memoria y políticas

- Trazabilidad explícita repo/memoria/RAG/web/inferencia vía `SourceReference.origin`,
  expuesta al usuario como "Trazabilidad RAG (recuperado/utilizado)".
- `ProjectMemory`: JSON versionado por workspace, con exploración incremental (fingerprints
  por tamaño/`mtime_ns`); **reutilizó de verdad resúmenes de corridas anteriores** en las
  corridas reales contra PrintScript.
- `security/policy_engine.py::PolicyEngine`: punto único de autorización, combinando config
  global, permisos por agente y reglas de perfil.
- `EvidenceSufficiencyPolicy` + `ProgressMonitor`: reconocen evidencia insuficiente y
  detectan estancamiento — implementados y testeados (`tests/integration/`), aunque no se
  ejercitó `ProgressMonitor` en la corrida real (el caso de uso no conectó Tester).

*Tiempo sugerido: 1.5 min*

---

## Slide 7 — Observabilidad: Langfuse real, no simulado

- `core/observability.py`: eventos tipados con jerarquía completa (`task_id`/`event_id`/
  `parent_event_id`), redacción automática de secretos, fallback seguro a `NoOp` si faltan
  credenciales.
- **Corrida real con Langfuse activo**: traza completa visible en el dashboard —
  `orchestrated-task` → `explorer`/`researcher` → `llm-completion` → `rag-retrieval` →
  `orchestrated-task-finished` — con prompts, modelo, tokens, latencia y resultado real.
- `estimated_cost`: tabla de precios públicos por modelo agregada; sigue siendo `None` para
  modelos desconocidos — nunca se inventa un costo.

*Tiempo sugerido: 1 min — mostrar el link a la traza si hay tiempo*

---

## Slide 8 — Dos bugs reales encontrados y arreglados en vivo

Buscando la evidencia de las 2 tareas, aparecieron —y se resolvieron— dos bugs genuinos:

1. **El prompt no restringía los valores válidos de `origin`** en las fuentes citadas: el LLM
   devolvía un valor fuera del enum esperado y la validación estricta rompía la ejecución.
   Arreglado explicitando los 5 valores literales permitidos en la instrucción de salida.
2. **Los subagentes ofrecían tools al LLM que nunca se ejecutaban**: ante preguntas más
   específicas, el modelo pedía tool calls (ej. buscar archivos sobre "concurrencia") y el
   framework las validaba pero nunca las ejecutaba ni retomaba la conversación — la respuesta
   quedaba vacía y el parseo de JSON fallaba siempre, de forma determinista y reproducible
   (confirmado inspeccionando `response.tool_calls` directamente, no por sospecha). Se
   arregló implementando un loop real de ejecución de tools en `BaseAgent`, reutilizando el
   mismo patrón que ya existía en el harness de la Parte 1, acotado a 3 rondas. Verificado:
   la misma pregunta que fallaba 4/4 veces ahora completa siempre.

Ambos arreglos validados contra la suite completa (389 tests, sin romper nada) y contra una
corrida real repetida.

*Tiempo sugerido: 1.5 min*

---

## Slide 9 — Auditoría de calidad de código (SOLID / acoplamiento)

Antes de la entrega se hizo una revisión dedicada sobre `agents/`, `core/`, `security/` y
`rag/`. Corregido:

- **Dependencia circular real entre `rag/` y `agents/`**: resuelta moviendo los contratos
  compartidos (`EvidenceFragment`, `KnowledgeRetriever`, etc.) a un módulo neutral
  (`core/research_ports.py`). Prueba empírica: los imports tardíos que antes eran
  obligatorios para poder importar el proyecto dejaron de serlo.
- **Manejo de errores inconsistente entre subagentes**: Tester y Reviewer no registraban sus
  fallos en `TaskState`, a diferencia de los otros tres — unificado con un context manager
  compartido (`BaseAgent._error_guard`).
- **Funciones duplicadas** (`_domain()` en 3 archivos, validadores de texto repetidos):
  unificadas en `core/validation.py`.

Documentado y **no** corregido a propósito (queda como reflexión honesta, no oculto):
`MainAgent.run()` mide ~300 líneas; `BaseAgent` tiene una violación de LSP (cada subagente
devuelve un tipo distinto), mitigada en la práctica porque el orquestador ya usa `Protocol`s
estructurales en vez de depender del tipo de la base.

*Tiempo sugerido: 1.5 min*

---

## Slide 10 — Cierre: qué demuestra este proyecto

- Un sistema multiagente real, sin frameworks, que:
  1. analiza un repositorio externo real que nunca había visto;
  2. fundamenta sus conclusiones citando RAG real, distinguiendo fuente/memoria/inferencia;
  3. **no alucina** cuando no tiene evidencia — lo dice explícitamente;
  4. queda completamente trazado en Langfuse;
  5. y cuyos propios bugs de integración con un LLM real se encontraron, diagnosticaron y
     arreglaron en el mismo proceso de generar la evidencia — no se escondieron.
- Reflexión completa, con qué funcionó, qué falló y qué se mejoraría, en
  `evidence/reflexion-final.md`.

*Tiempo sugerido: 1 min — cierre y preguntas*
