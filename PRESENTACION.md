# TP Final — Coding Agent Avanzado
> Presentación oral (10 min) — estructurada según la consigna del TP Final
> Documento de apoyo — no versionado (ver .gitignore)
> Actualizado con los commits: `feat: add policy preflight and incremental exploration`,
> `test: add generic end-to-end agent scenarios`, `docs: complete setup and architecture documentation`,
> más la incorporación real de PrintScript como workspace y una auditoría de calidad de código aplicada.

⚠️ **Único punto realmente pendiente** (ver Slide 14): correr una tarea real completa contra PrintScript.
El workspace, el perfil, el RAG (con la especificación real del lenguaje) y el script de composición
(`run_agent.py`) ya están armados y probados contra la API real de OpenAI — sólo falta que la cuenta usada
tenga cuota disponible.

---

## Slide 1 — Consigna del TP y punto de partida

- Objetivo del TP: evolucionar el coding agent de clase hacia un **sistema multiagente**, sin frameworks
  de orquestación (LangChain/LangGraph/CrewAI/AutoGen), que combine tools locales + RAG + memoria
  persistente + subagentes especializados + políticas de seguridad + observabilidad.
- Punto de partida conservado de la Parte 1: el harness (`core/harness.py`) y las tools base
  (`read_file`, `write_file`, `list_files`, `run_command`, `web_search`).
- Todo lo nuevo se construyó **sin tocar esos contratos base** — se extendió por composición. El README
  lo deja explícito: "la arquitectura multiagente... todavía no reemplaza al bootstrap interactivo de
  `main.py`" — son dos capas conscientemente separadas.

*Tiempo sugerido: 1 min*

---

## Slide 2 — Caso de uso concreto: PrintScript (ya incorporado, no sólo planeado)

- **PrintScript real** (Kotlin/Gradle: lexer, parser, interpreter, formatter, linter, runner, cli), fuera
  de este repo, referenciado como workspace externo vía `agent.config.yaml` (`workspace.path`, sólo
  lectura por ahora: `write: false`, `run_commands: false`).
- **`profiles/printscript.yaml`**: perfil real con la especificación del lenguaje entregada por la
  cátedra como fuente RAG (no el README del checkout, que no tenía contenido sustancial).
- **RAG indexado con contenido real y en dos niveles**: (1) la especificación de PrintScript —el lenguaje
  que se implementa— (11 chunks); (2) documentación oficial de Kotlin —el lenguaje de implementación—
  descargada en vivo desde kotlinlang.org (sintaxis básica, null safety, sealed classes; 57 chunks limpios).
  Se agregó `HtmlDocumentParser` (`rag/processing.py`) para extraer texto plano de HTML real sin ruido de
  etiquetas, coherente con la arquitectura de parsers intercambiables ya existente (`sections`/`plain`/`html`).
- **`run_agent.py`**: script de composición que arma `MainAgent` (Explorer + Researcher) desde
  `agent.config.yaml` — validado de punta a punta contra la API real de OpenAI (llegó a llamarla; sólo
  falló por cuota agotada de la cuenta de prueba, no por un error de código).
- **Alcance deliberadamente acotado a "Analizar un repositorio desconocido"** (uno de los dos objetivos
  que ofrece la consigna, completo por sí solo): sólo Explorer + Researcher. No se conectan
  Implementer/Tester/Reviewer a este caso de uso — decisión consciente, no una limitación por descubrir.
  Evita depender de que Gradle/Kotlin compilen en la máquina de evaluación, sin sacrificar ningún
  requisito obligatorio (RAG con fuentes citadas, memoria, políticas, observabilidad).

*Tiempo sugerido: 1.5 min*

---

## Slide 3 — Arquitectura de agentes: agente principal

- **`agents/orchestrator.py` → `MainAgent`**: recibe el pedido del usuario, mantiene el `TaskState` y
  coordina el trabajo de los subagentes.
- Flujo completo (ver diagrama Mermaid en README): clasificar tarea → consultar memoria → explorar
  (completo o incremental) → investigar si aplica → generar plan → aprobación humana → operaciones
  planificadas → **preflight de políticas** → evidencia → implementación → testing → review → resultado.
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

- Todos heredan de `agents/base.py::BaseAgent`, con `allowed_tools` propio y dependencias explícitas
  (LLM, memoria, RAG, web, aprobación y observabilidad pueden inyectarse falsos en tests).

*Tiempo sugerido: 1.5 min*

---

## Slide 5 — Novedad: exploración incremental (commit `policy preflight and incremental exploration`)

- El `Explorer` ahora puede evitar re-explorar todo el repo si ya existe una exploración previa vigente en
  `ProjectMemory`.
- Usa **fingerprints por tamaño y `mtime_ns`** de archivos; revalida una muestra estable, inspecciona
  archivos nuevos o modificados, y **vuelve a exploración completa** si la memoria falta, está corrupta o
  el fingerprint no es confiable.
- Limitación documentada honestamente en el README: un cambio que preserve tamaño y `mtime_ns` podría
  requerir exploración completa manual — no es mágico, es una heurística explícita.

*Tiempo sugerido: 1 min*

---

## Slide 6 — Novedad central: intención estructurada y `PolicyPreflight`

- **`core/planned_operations.py`**, módulo nuevo — resuelve un riesgo que antes no estaba cubierto:
  ¿qué pasa si el plan en texto libre "sugiere" una operación sensible que nadie estructuró?
- **`PlannedOperation`**: cada operación (`write_file`, `run_command`, `git_operation`, etc.) tiene un
  `fingerprint` (hash de tipo + target + parámetros + versión del plan) — una aprobación sólo vale para
  esa operación exacta, en esa versión exacta del plan.
- **`StructuredPlannedOperationProvider`**: sólo convierte en operación lo que ya viene respaldado por
  campos estructurados (`files_relevant` del Explorer, `changes` del Implementer, comandos ya registrados
  en `TaskState`). **El texto libre del plan nunca se interpreta como acción por heurísticas.**
- **`PolicyPreflight.evaluate()`**: corre `PolicyEngine` sobre cada operación planificada *antes* de tocar
  evidencia o escritura — `allow` / `deny` / `require_approval` / `insufficient_structured_intent`.
- Si cambia el plan, la aprobación de preflight se invalida (el fingerprint incluye `plan_version`).

*Tiempo sugerido: 1.5 min*

---

## Slide 7 — Estado compartido de la tarea

- **`core/task_state.py` → `TaskState`**: objeto único compartido, ahora también con operaciones
  planificadas y fingerprints de preflight aprobados (además de lo ya existente: pedido, fase, resultados
  de subagentes, fuentes, archivos modificados, tool calls, errores, observaciones, plan aprobado y
  `EvidenceAssessment` ligado a la versión vigente del plan).
- Cambiar el plan invalida tanto la evidencia como el preflight — el estado nunca queda con aprobaciones
  "viejas" aplicadas a un plan distinto.

*Tiempo sugerido: 1 min*

---

## Slide 8 — Memoria persistente del proyecto

- **`agents/project_memory.py` → `ProjectMemory`**: JSON versionado por workspace, reemplazo atómico y
  permisos de archivo restrictivos.
- Guarda arquitectura, tecnologías, módulos, archivos importantes, dependencias, comandos, convenciones,
  decisiones, bugs, tareas previas, resúmenes **y ahora también fingerprints de exploración** (para la
  exploración incremental).
- Redacción de secretos incorporada antes de persistir. "La memoria es una pista, no una verdad actual"
  (cita textual del README) — siempre se revalida contra el estado real del repo.

*Tiempo sugerido: 1 min*

---

## Slide 9 — RAG: pipeline completo

- **Chunking** (`rag/processing.py`): reconoce encabezados Markdown, normaliza espacios, tamaño máximo y
  solapamiento configurables (1500 / 150 caracteres por defecto).
- **Embeddings** (`rag/embeddings.py`): interfaz intercambiable; `HashEmbeddingProvider` local y
  determinista para no depender de un proveedor externo (el README aclara: no equivale semánticamente a un
  modelo entrenado).
- **Vector store** (`rag/vector_store.py::JsonVectorStore`): índice versionado, reindexa sólo lo que
  cambió por hash, deduplica y puede podar documentos eliminados.
- **Recuperación con fallback web**: orden memoria → RAG → evaluación de suficiencia → web, priorizando
  dominios oficiales configurados por perfil.
- CLI dedicada: `python -m rag.cli manifest.json [--prune]`.

*Tiempo sugerido: 1.5 min*

---

## Slide 10 — Trazabilidad de fuentes y detección de estancamiento

- Trazabilidad explícita repo/memoria/RAG/web/inferencia vía `SourceReference.origin`, expuesta al usuario
  como "Trazabilidad RAG (recuperado/utilizado)" y "Trazabilidad web (encontrado/utilizado)".
- **`core/progress.py::ProgressMonitor`**: detecta comandos repetidos con el mismo error, relecturas sin
  novedad, búsquedas repetidas, diffs idénticos, ciclos entre agentes e iteraciones sin evidencia nueva.
  Recomienda `retry_with_new_strategy`, `replan`, `ask_user` o `stop`.
- **`EvidenceSufficiencyPolicy`**: `sufficient` / `partial` / `insufficient`, con faltantes y acción
  recomendada explícitos.

*Tiempo sugerido: 1 min*

---

## Slide 11 — Configuración, políticas y observabilidad

- `agent.config.yaml`: workspace, permisos por tool, límites, RAG, memoria, web, observabilidad — loader
  estricto (rechaza YAML inválido, campos desconocidos, escapes de workspace).
- `security/policy_engine.py::PolicyEngine`: punto único de autorización, combinando config global,
  permisos por agente y reglas de perfil (`denied_tools`, `protected_paths`, `require_approval_tools`).
  Ahora también es el motor detrás del `PolicyPreflight` (Slide 6).
- Observabilidad Langfuse (`core/observability.py`): eventos tipados con jerarquía completa
  (`task_id`/`event_id`/`parent_event_id`), redacción automática de secretos, fallback seguro a `NoOp` si
  faltan credenciales, `estimated_cost` nunca inventado si no hay tabla de precios explícita.

*Tiempo sugerido: 1.5 min*

---

## Slide 12 — Evidencia reproducible: escenarios integrales (commit `end-to-end agent scenarios`)

- Nueva carpeta **`tests/integration/`**: 6 escenarios end-to-end con repos temporales aislados
  (`tmp_path`) y fakes deterministas (sin red, sin OpenAI/Tavily/Langfuse reales):
  - `test_analysis_scenario.py` — análisis sin escritura.
  - `test_simple_change_scenario.py` — cambio localizado aplicado.
  - `test_rag_scenario.py` — RAG sin necesidad de fallback web.
  - `test_memory_scenario.py` — recarga real de memoria persistida.
  - `test_failed_command_scenario.py` — comando fallido + detección de progreso.
  - `test_blocked_operation_scenario.py` — *safe stop* por preflight.
- El propio README es explícito: **"las aserciones de los tests son la evidencia reproducible"** — no
  reemplazan una corrida real contra PrintScript, pero sí demuestran cada capacidad pedida por la consigna
  de forma verificable y repetible en el momento del examen (`python -m pytest tests/integration -q`).

*Tiempo sugerido: 1 min — correr uno en vivo si hay tiempo*

---

## Slide 13 — Auditoría de calidad de código (SOLID / acoplamiento) y qué se corrigió

Antes de la entrega se hizo una revisión de código dedicada sobre `agents/`, `core/`, `security/` y `rag/`.
Hallazgo más importante y **ya corregido**:

- **Dependencia circular real entre `rag/` y `agents/`**: `rag/retriever.py` importaba tipos
  (`EvidenceFragment`, `KnowledgeRetriever`) desde `agents/researcher.py`, obligando a imports tardíos en
  `core/profiles.py` y `core/settings.py` sólo para que el proyecto pudiera importarse. Se resolvió
  moviendo esos contratos a un módulo neutral (`core/research_ports.py`), del que ahora dependen tanto
  `agents/` como `rag/` sin ciclo. **Prueba empírica**: los imports tardíos ya no son necesarios y se
  eliminaron; toda la suite (389 tests) sigue pasando igual.
- **Manejo de errores inconsistente entre subagentes**: Tester y Reviewer no registraban sus fallos en
  `TaskState.errors` a diferencia de Explorer/Researcher/Implementer. Se extrajo un context manager
  compartido (`BaseAgent._error_guard`) y se aplicó a los 5 subagentes por igual.
- **Función `_domain()` de normalización de dominios, duplicada 3 veces** (`core/profiles.py`,
  `core/config.py`, `agents/web_research.py`): unificada en `core/validation.py`, cada módulo conserva su
  propia excepción específica.

Hallazgos **documentados pero no corregidos** (quedan como reflexión honesta, no ocultos): `MainAgent.run()`
mide ~300 líneas y concentra demasiadas fases; `BaseAgent` tiene una violación de LSP (cada subagente
devuelve un tipo distinto pese a heredar la misma firma) — mitigada en la práctica porque el orquestador ya
usa `Protocol`s estructurales en vez de depender del tipo de la base.

*Tiempo sugerido: 1.5 min*

---

## Slide 14 — Lo único que falta cerrar

1. **Cuota de OpenAI**: el pipeline completo (config → perfil → RAG → `MainAgent` → LLM real) ya se probó
   de punta a punta; sólo falta que la cuenta usada tenga crédito disponible para completar una corrida.
   Nada más depende de esto — el alcance elegido (sólo análisis) no necesita Gradle, Kotlin compilado ni
   ningún comando de build/test de PrintScript.
2. **Evidencia formal de 2 tareas + captura de Langfuse**: en cuanto haya cuota, correr
   `python run_agent.py "<pedido>"` con 2 pedidos de análisis distintos y documentar output, fuentes RAG
   citadas (spec de PrintScript + docs de Kotlin) y la traza.
3. Sin esto, la presentación ya puede apoyarse en: arquitectura implementada, políticas, observabilidad,
   auditoría de calidad con hallazgos corregidos y documentados, y los 6 escenarios integrales como
   evidencia reproducible (`python -m pytest tests/integration -q`).

*Tiempo sugerido: 1 min — cierre y preguntas*
