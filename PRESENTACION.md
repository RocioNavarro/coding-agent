# TP Final — Coding Agent Avanzado
> Presentación oral (10 min) — estructurada según la consigna del TP Final
> Documento de apoyo — no versionado (ver .gitignore)

⚠️ **Pendiente antes del examen** (ver Slide 12): definir el caso de uso concreto, ejecutar y documentar
al menos 2 corridas, y tomar capturas de Langfuse. El resto de los requisitos de la consigna ya está
implementado en el código — se detalla abajo con archivo/módulo concreto para poder mostrarlo en vivo.

---

## Slide 1 — Consigna del TP y punto de partida

- Objetivo del TP: evolucionar el coding agent de clase hacia un **sistema multiagente**, sin frameworks
  de orquestación (LangChain/LangGraph/CrewAI/AutoGen), que combine tools locales + RAG + memoria
  persistente + subagentes especializados + políticas de seguridad + observabilidad.
- Punto de partida conservado de la Parte 1: el harness (`core/harness.py`) y las tools base
  (`read_file`, `write_file`, `list_files`, `run_command`, `web_search`).
- Todo lo nuevo se construyó **sin tocar esos contratos base** — se extendió por composición.

*Tiempo sugerido: 1 min*

---

## Slide 2 — Arquitectura de agentes: agente principal

- **`agents/orchestrator.py` → `MainAgent`**: recibe el pedido del usuario, mantiene el `TaskState` y
  coordina el trabajo de los subagentes. Puede bloquear la tarea o ejecutar directamente cuando corresponde.
- Flujo que orquesta: **análisis de tarea → exploración → investigación → planificación (con aprobación
  humana) → evaluación de evidencia → implementación → testing → review → resultado final**.
- Estados de salida explícitos y no ambiguos: `completed`, `rejected`, `blocked`, `max_iterations`.

*Tiempo sugerido: 1 min*

---

## Slide 3 — Arquitectura de agentes: los 5 subagentes pedidos

| Subagente | Archivo | Responsabilidad (según consigna) |
|---|---|---|
| Explorer | `agents/explorer.py` | Entiende el repo: estructura, dependencias, convenciones, archivos relevantes |
| Researcher | `agents/researcher.py` | Busca evidencia en el RAG y, si falta, en la web |
| Implementer | `agents/implementer.py` | Evalúa evidencia y propone/aplica cambios de código |
| Tester | `agents/tester.py` | Corre validaciones (tests/build/lint) sobre el cambio aplicado |
| Reviewer | `agents/reviewer.py` | Aprueba, rechaza o pide replanificar el diff final |

- Cada subagente hereda de **`agents/base.py` (`BaseAgent`)**: contexto acotado explícito, `allowed_tools`
  propio (no todos comparten las mismas tools/permisos, tal como pide la consigna), salida normalizada a JSON.

*Tiempo sugerido: 1.5 min*

---

## Slide 4 — Estado compartido de la tarea

- **`core/task_state.py` → `TaskState`**: objeto único compartido entre `MainAgent` y subagentes.
- Registra como mínimo lo exigido por la consigna: pedido original, avance/fase actual, resultados de cada
  subagente (`subagent_results`), fuentes consultadas (`sources`), archivos modificados (`files_modified`),
  observaciones (`observations`), advertencias y errores.
- Se pasa explícitamente en cada llamada — ningún subagente accede a estado no delegado, evitando
  acoplamiento implícito.

*Tiempo sugerido: 1 min*

---

## Slide 5 — Memoria persistente del proyecto

- **`agents/project_memory.py` → `ProjectMemory`**: persistencia local en disco, aislada por workspace,
  auditable (JSON versionado con `schema_version`).
- Guarda lo pedido por la consigna: resumen del proyecto, tecnologías, arquitectura detectada, módulos,
  archivos importantes, dependencias, comandos conocidos, convenciones y resúmenes de tareas previas.
- **Redacción de secretos incorporada** (`SECRET_KEY`, `SECRET_TOKEN`, rutas sensibles) antes de escribir a
  disco — nunca persiste credenciales.
- El `MainAgent` la carga al iniciar cada tarea y guarda el resumen al finalizar (`orchestrator.py::_result`).

*Tiempo sugerido: 1 min*

---

## Slide 6 — RAG: pipeline completo

Carpeta `rag/` — pipeline propio, sin librerías de orquestación:

- **Chunking configurable** (`rag/processing.py::ConfigurableChunker`): respeta encabezados Markdown,
  parámetros de tamaño máximo y solapamiento configurables (por defecto 1500 caracteres, 150 de overlap).
- **Embeddings desacoplados** (`rag/embeddings.py`): interfaz `EmbeddingProvider` intercambiable;
  implementación local determinística (`HashEmbeddingProvider`) para no depender de un proveedor externo.
- **Vector store persistente** (`rag/vector_store.py::JsonVectorStore`): índice versionado en disco,
  detecta cambios por hash de documento para reindexar sólo lo modificado.
- **Recuperación** (`rag/retriever.py`) + **fallback web** cuando no hay evidencia suficiente en el RAG
  (implementado en `researcher.py`, con prioridad a documentación oficial vía `priority_web_domains` del perfil).
- CLI dedicada: `coding-agent-index` (script instalable, `rag/cli.py`).

*Tiempo sugerido: 1.5 min*

---

## Slide 7 — Trazabilidad de fuentes (requisito explícito de la consigna)

- La consigna pide **diferenciar** repositorio / memoria / RAG / web / inferencia propia — se resuelve con
  `SourceReference.origin` en `core/task_state.py` y se expone al usuario en el resultado final
  (`TextResultPresenter` en `orchestrator.py`):
  - "Trazabilidad RAG (recuperado/utilizado)"
  - "Trazabilidad web (encontrado/utilizado)"
  - Fuentes marcadas como `inferido` vs `utilizado`.
- Esto permite mostrar, para cualquier corrida, exactamente qué fragmento de qué documento se usó.

*Tiempo sugerido: 1 min*

---

## Slide 8 — Manejo de contexto y detección de estancamiento

- **Contexto acotado** (`agents/context_manager.py::StateContextManager`): selecciona un subconjunto
  relevante del `TaskState` por subagente (máx. caracteres, máx. fuentes, máx. archivos) — nunca se manda
  el repo o el historial completo al modelo.
- **Detección de loops/falta de progreso** (`core/progress.py::ProgressMonitor`): detecta repetición de
  errores de comando, relecturas sin novedad, búsquedas repetidas, ciclos entre agentes o iteraciones sin
  evidencia nueva. Recomienda `retry_with_new_strategy`, `replan`, `ask_user` o `stop` — tal como pide la
  consigna ("cambiar de estrategia, replanificar, detenerse o pedir ayuda").
- **Reconocimiento de evidencia insuficiente**: `security/evidence_policy.py::EvidenceSufficiencyPolicy`
  bloquea la implementación ante pedidos ambiguos, falta de documentación, permisos insuficientes o riesgo
  excesivo, explicando qué falta (`missing_information`) y qué acción recomienda.

*Tiempo sugerido: 1.5 min*

---

## Slide 9 — Configuración y políticas del agente

- **`agent.config.example.yaml`**: workspace, permisos por tool (read/write/run_commands/web_search),
  límites (iteraciones, contexto, timeout, resultados RAG/web), comandos permitidos, observabilidad.
- **`security/policy_engine.py::PolicyEngine.evaluate()`**: punto único de validación antes de cada tool
  call — combina config global, permisos por agente y reglas de **perfil de proyecto**
  (`profiles/*.yaml`: `denied_tools`, `denied_commands`, `protected_paths`, `require_approval_tools`).
  Devuelve `allow` / `deny` / `require_approval`, igual que el ejemplo de la consigna.
- Reglas base siempre activas (independientes de la config): `security/command_policy.py` y
  `security/paths.py` bloquean `.env`, `secrets/**`, `*.pem`, `git push`, `rm -rf`, escapes de workspace.

*Tiempo sugerido: 1 min*

---

## Slide 10 — Observabilidad (Langfuse)

- **`core/observability.py`**: eventos tipados con jerarquía (`task_id`/`event_id`/`parent_event_id`) para
  cada prompt, llamada al LLM, tool, documento RAG recuperado, búsqueda web, iteración, error y resultado
  final — exactamente el mínimo pedido por la consigna.
- **`LangfuseObservabilityClient`**: adaptador al SDK v4 de Langfuse; se activa sólo con
  `CODING_AGENT_OBSERVABILITY_ENABLED=true` + credenciales; si faltan, cae a `NoOpObservabilityClient`
  sin romper la ejecución.
- **Redacción automática de secretos** antes de emitir cualquier evento (`sanitize_observability_data`).
- Registra tokens, latencia y costo estimado por evento (`ObservabilityEvent`).

*Tiempo sugerido: 1 min — mostrar acá, si es posible, una traza real en Langfuse*

---

## Slide 11 — Extra opcional: registro de tools sin tocar el núcleo

- `tools/registry.py` ya centraliza descubrimiento y validación de tools mediante `ToolDefinition`
  (nombre, schema, función, `modifies_system`) — habilitar una tool nueva no requiere tocar el harness.
- Mencionar como plus si el tiempo alcanza; no es el foco central de la exposición.

*Tiempo sugerido: 0.5 min (opcional, recortable)*

---

## Slide 12 — Lo que falta cerrar antes del examen

La arquitectura y las políticas están implementadas; lo que exige la consigna y **todavía no está resuelto**:

1. **Caso de uso concreto** (obligatorio): elegir el lenguaje/framework/ecosistema y el objetivo verificable
   (ej. "analizar repo desconocido" o "agregar funcionalidad concreta"). Hoy `profiles/default.yaml` está
   vacío — falta completar un perfil real con `rag_sources` apuntando a documentación de ese ecosistema.
2. **Evidencia de 2 tareas ejecutadas**: correr el agente sobre el caso elegido, guardar output, fuentes
   recuperadas y observaciones de cada corrida.
3. **Capturas de Langfuse**: al menos una traza completa de ejecución.
4. **Reflexión breve**: qué funcionó, qué falló, cuándo se detectaron loops o falta de evidencia (esto se
   puede armar directamente a partir de lo que `ProgressMonitor` y `EvidenceSufficiencyPolicy` reporten en
   las corridas reales).

*Tiempo sugerido: 1 min — cierre y preguntas*
