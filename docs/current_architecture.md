# Arquitectura actual y evolución propuesta

## Alcance del relevamiento

Este documento describe la implementación actual de la Parte 1. La propuesta final
es solamente una organización futura: en esta etapa no se implementan subagentes,
persistencia, RAG ni cambios en el comportamiento de la CLI.

## Estructura actual

```text
coding-agent/
├── main.py                    # composición, CLI y loop externo
├── core/
│   ├── harness.py            # planificación y loop interno
│   ├── llm_client.py         # contrato LLM y adaptador de OpenAI
│   ├── models.py             # mensajes y resultados normalizados
│   ├── settings.py           # configuración mutable de la sesión
│   └── supervision.py        # aprobación de tools modificadoras
├── tools/
│   ├── definitions.py        # definición, schema y resultado de una tool
│   ├── registry.py           # catálogo, validación y despacho
│   ├── file_tools.py         # lectura, escritura y listado
│   ├── command_tools.py      # ejecución de procesos
│   └── web_tools.py          # búsqueda mediante Tavily
├── security/
│   ├── paths.py              # confinamiento de rutas
│   └── command_policy.py     # reglas defensivas para comandos
├── tests/                    # tests unitarios con LLM y APIs simulados
└── workspace/                # raíz permitida para tools locales y demos
```

### Entry point

El entry point ejecutable es `main.py`, mediante `python main.py`. El bloque
`if __name__ == "__main__"` llama a `main()`, que:

1. carga variables con `load_environment()`;
2. construye `AgentSettings`;
3. crea `OpenAILLMClient`;
4. crea un `ToolRegistry` nuevo con `build_default_registry()`;
5. inicia `run_chat()`.

`pyproject.toml` empaqueta `main` como módulo, pero no declara un script de consola.
Por lo tanto no existe otro entry point público instalado.

## Flujo actual

```text
stdin
  -> main.run_chat (loop externo e historial de sesión)
       -> comando local: cambia settings, informa estado o termina
       -> pedido del usuario
            -> core.harness.run_planning_loop, si Plan mode está activo
                 -> LLM sin schemas de tools
                 -> aprobar / rechazar / solicitar modificación
            -> core.harness.run_internal_loop
                 -> LLM con schemas de tools
                 -> SupervisedToolExecutor
                      -> validar argumentos
                      -> confirmar si corresponde
                      -> ToolRegistry.execute
                           -> file / command / web tool
                           -> políticas de seguridad aplicables
                 -> resultado correlacionado en el historial
                 -> repetir hasta respuesta sin tool calls
       -> imprimir respuesta final
```

### Recepción del pedido y loop externo

`main.run_chat()` crea una lista `history` con el `SYSTEM_PROMPT` y permanece en un
`while True`. Lee `Usuario>`, ignora entradas vacías y procesa localmente `/exit`,
`/status`, `/plan on|off` y `/supervision on|off`; esos comandos no llegan al LLM ni
se guardan en el historial. Un pedido normal se agrega como `Message(role="user")`.

La configuración comienza con Plan mode y Supervision mode activos, máximo de 20
iteraciones y timeout declarado de 60 segundos. Los toggles mutan la misma instancia
de `AgentSettings` durante la sesión.

### Generación y revisión del plan

Con Plan mode activo, `run_chat()` llama a `run_planning_loop()` antes del loop de
ejecución. Éste copia el historial principal, agrega una instrucción `developer` y
llama a `LLMClient.complete(planning_history, ())`: no entrega schemas, por lo que
las tools no están disponibles en planificación. Una tool call o un plan vacío se
consideran errores.

El callback interactivo permite:

- **aprobar**: devuelve el plan y habilita la ejecución;
- **rechazar**: cancela ese pedido, agrega al historial principal un mensaje de
  cancelación del asistente y vuelve al prompt;
- **modificar**: agrega a la copia de planificación la respuesta anterior y una
  instrucción con el cambio pedido, luego solicita un plan completo nuevo.

Las revisiones consumen el mismo valor `max_iterations` usado por el loop interno,
aunque son límites conceptualmente diferentes. Los planes intermedios y pedidos de
modificación no pasan al historial principal. Al aprobar, sí se agregan el plan como
mensaje `assistant` y una instrucción `developer` que ordena ejecutarlo.

Con Plan mode desactivado se omite toda esta fase: el pedido entra directamente al
loop interno y no hay aprobación global previa a las tools.

### Ejecución, llamadas al LLM y llamadas a tools

`run_internal_loop()` construye un `SupervisedToolExecutor` y, en cada iteración:

1. registra visiblemente el número de iteración;
2. llama al LLM con todo el historial y todos los schemas del registro;
3. agrega el mensaje normalizado del asistente al mismo historial;
4. si no hay tool calls, devuelve la respuesta final;
5. si hay una o más, las procesa secuencialmente en el orden recibido;
6. registra intención, nombre, argumentos redactados y resultado resumido;
7. agrega por cada ejecución un `Message(role="tool")` JSON, correlacionado mediante
   `tool_call_id`, y vuelve a llamar al LLM.

Antes de ejecutar, `SupervisedToolExecutor` busca la tool y valida argumentos. Si
Supervision mode está activo y `modifies_system=True`, solicita confirmación. El
rechazo o error se convierte en un resultado controlado que también vuelve al LLM.
`ToolRegistry.execute()` vuelve a validar, invoca el executor y captura excepciones.
El loop falla con `MaxIterationsError` si todas las iteraciones terminan en tool
calls. `run_chat()` muestra errores sin traceback y conserva el historial parcial.

`OpenAILLMClient` es el límite con el proveedor: traduce los modelos propios a la
Responses API, convierte schemas, normaliza tool calls, uso de tokens, modelo y
latencia, y traduce errores del SDK a errores propios.

## Tools y permisos actuales

| Tool | Efecto declarado | Aprobación con supervisión | Restricción efectiva |
|---|---|---|---|
| `read_file(path)` | lectura | no | UTF-8, ruta relativa dentro de `workspace/`, sin `.env` ni symlinks de escape |
| `list_files(path=".")` | lectura | no | directorio dentro de `workspace/`, listado no recursivo |
| `write_file(path, content)` | modificación | sí | reemplaza/crea texto y padres dentro de `workspace/`; sin `.env` ni escapes |
| `run_command(command)` | modificación potencial | sí | `cwd=workspace/`, `shell=False`, timeout, política de comandos y rutas |
| `web_search(query, max_results=5)` | lectura externa | no | Tavily, de 1 a 10 resultados, snippets acotados; requiere `TAVILY_API_KEY` |

La autorización actual tiene tres niveles independientes:

1. Plan mode, si está activo, exige aprobación global antes de exponer tools.
2. Supervision mode exige confirmación individual sólo para las tools cuyo booleano
   `modifies_system` es verdadero (`write_file` y `run_command`).
3. Las políticas de rutas y comandos se aplican siempre, aun sin supervisión.

`run_command` bloquea ejecutables destructivos conocidos, wrappers, `git push`,
`git reset --hard`, evaluación inline con intérpretes, rutas externas y ciertos
nombres sensibles. Es una lista defensiva, no un sandbox del sistema operativo.
`web_search` sale a la red pero está clasificada como no modificadora y no pide
confirmación.

## Historial y estado de conversación

El historial vive exclusivamente en la variable local `history: list[Message]` de
`run_chat()`. Se pasa por referencia al loop interno, que agrega respuestas y salidas
de tools. Persiste entre pedidos mientras el proceso y esa llamada a `run_chat()`
continúan, y se devuelve al finalizar principalmente para facilitar tests.

No se serializa a disco ni a base de datos, no tiene identificador de sesión,
versionado, compactación ni recuperación tras reinicio. Tampoco existe un objeto de
estado que reúna historial, plan, resultados, artefactos y métricas. La otra porción
de estado mutable es `AgentSettings`; las revisiones de planificación usan una copia
temporal separada del historial.

## Componentes reutilizables

### Agente principal

- `run_internal_loop()` ya contiene el ciclo básico LLM/tool calling.
- `run_planning_loop()` y los modelos `PlanReview`/`PlanningResult` sirven como
  servicio de planificación con aprobación humana.
- `LLMClient` permite inyectar proveedores o clientes falsos.
- La composición explícita de `main.main()` es una base sencilla para un futuro
  bootstrap sin introducir un framework de orquestación.

### Subagentes

- `LLMClient`, `Message`, `LLMResponse`, `ToolCall`, `ToolRegistry` y
  `SupervisedToolExecutor` no dependen de la consola y pueden reutilizarse por cada
  runtime de agente.
- `build_default_registry()` puede evolucionar a una fábrica de registros por rol o
  capacidad, evitando entregar todas las tools a todos los agentes.
- El algoritmo de `run_internal_loop()` puede extraerse a una clase configurable;
  hoy no modela identidad, objetivo delegado, presupuesto, jerarquía ni resultado de
  subagente.

### Estado compartido

- `Message` es un buen formato de intercambio normalizado y los resultados de tools
  ya son JSON serializable.
- Hace falta agregar un `RunState`/`TaskState` explícito y un repositorio con control
  de concurrencia. La lista actual no debe compartirse directamente entre agentes:
  sus mutaciones carecen de aislamiento, ownership y correlación de ejecución.

### Memoria persistente

- El contrato `Message` y las métricas de `LLMResponse` son datos persistibles.
- Conviene definir puertos separados para sesiones, mensajes y memorias; la memoria
  no debería quedar embebida en `run_chat()` ni acoplada inicialmente a un motor
  concreto. Actualmente no hay implementación reutilizable de almacenamiento.

### RAG

- `read_file`, `list_files`, `web_search` y el confinamiento de rutas son fuentes
  reutilizables para ingesta controlada.
- `ToolRegistry` puede publicar futuras tools de búsqueda/recuperación.
- Aún faltan contratos de documento y fragmento, ingesta, embeddings, índice,
  recuperación, citas y políticas de actualización; `web_search` por sí sola no es
  RAG.

### Políticas

- `security.paths` y `security.command_policy` ya separan dos políticas puras y
  testeables.
- `SupervisedToolExecutor` centraliza el punto de enforcement humano.
- `ToolDefinition.modifies_system` es un inicio de metadatos de capacidad, pero un
  booleano no alcanza para permisos por agente, alcance, red, costo, sensibilidad o
  tipo de efecto. Debe evolucionar sin duplicar las validaciones de seguridad de
  cada executor.

### Observabilidad

- `LLMResponse` ya conserva modelo, tokens y latencia.
- El harness registra iteraciones, tool calls, argumentos redactados y resultados
  acotados mediante un callback de salida inyectable.
- Estos eventos hoy son texto para consola: no tienen tipo, timestamp, run/agent ID,
  duración de tools, acumulación de uso ni sink persistente. La redacción sólo cubre
  nombres y patrones comunes de secretos.

## Riesgos y responsabilidades mezcladas

1. **`main.py` mezcla interfaz y aplicación.** Compone dependencias, mantiene la
   sesión, interpreta comandos, implementa diálogos de aprobación y formatea salida.
   Esto dificulta sumar otra interfaz sin replicar el flujo.
2. **`core/harness.py` concentra orquestación y presentación.** Decide el ciclo,
   muta historial, ejecuta tools, serializa resultados, redacta secretos y produce
   texto específico de CLI.
3. **El historial es una lista compartida implícita.** No hay límites de contexto,
   transacciones ni rollback. Un error conserva mensajes parciales y una ejecución
   multiagente produciría escrituras concurrentes ambiguas.
4. **Plan y ejecución no son entidades de dominio.** El plan aprobado se codifica
   como dos mensajes; no hay pasos, estado, trazabilidad entre paso y tool call ni
   replanificación estructurada.
5. **Permisos demasiado gruesos.** `modifies_system` decide toda supervisión. No
   expresa que `web_search` usa red ni distingue lectura, creación, proceso, costo o
   alcance por rol.
6. **Registro acoplado a implementaciones concretas.** `tools.registry` importa todas
   las tools para construir el catálogo y además expone un singleton global
   `TOOL_REGISTRY`, aunque la aplicación crea otra instancia.
7. **Errores con semántica inconsistente.** Las file tools devuelven strings que
   empiezan con `Error`, por lo que el registro los marca como `success=True`; otras
   tools lanzan excepciones o retornan `exit_code=-1`. Esto complica decisiones y
   métricas confiables.
8. **Configuración no fluye completamente.** `AgentSettings.command_timeout_seconds`
   se muestra en CLI, pero `run_command` toma al importar un valor constante desde
   `DEFAULT_SETTINGS`; cambiar una instancia de settings no cambia el timeout real.
9. **Observabilidad incompleta.** Tokens y latencia se normalizan pero no se muestran,
   acumulan ni emiten. El logging mediante `print`/callback no permite consultas ni
   correlación.
10. **Seguridad limitada por diseño.** La allow/deny policy de comandos no puede
    determinar el comportamiento interno de todo binario permitido. `write_file` no
    es atómica y no conserva una copia recuperable. La carga de `.env` está fuera de
    las tools, pero ocurre dentro del proceso que ejecuta el agente.
11. **Límites reutilizados.** Un único `max_iterations` controla revisiones de plan y
    vueltas del tool loop; futuros agentes necesitarán presupuestos separados por
    tarea, profundidad, tokens, tiempo y tools.
12. **No existe persistencia ni RAG.** Confundir el historial en RAM o la búsqueda
    web con esas capacidades llevaría a una arquitectura difícil de evolucionar.

## Propuesta de estructura futura

La migración puede hacerse por extracción de responsabilidades, preservando los
contratos y tests existentes. Una estructura objetivo posible es:

```text
coding-agent/
├── main.py                         # wrapper compatible: delega al bootstrap
├── coding_agent/
│   ├── bootstrap.py                # composición de dependencias
│   ├── application/
│   │   ├── chat_service.py         # casos de uso de sesión/turno
│   │   ├── planning_service.py     # generar, revisar y aprobar planes
│   │   └── orchestration_service.py# coordinar principal y futuras delegaciones
│   ├── agents/
│   │   ├── runtime.py              # loop LLM/tools extraído del harness
│   │   ├── main_agent.py           # política y capacidades del coordinador
│   │   ├── definitions.py          # AgentDefinition, objetivo y capacidades
│   │   └── delegation.py           # contratos futuros; sin CLI
│   ├── domain/
│   │   ├── messages.py             # Message, ToolCall y respuestas
│   │   ├── plans.py                # plan, pasos y decisiones
│   │   ├── state.py                # RunState/TaskState y correlaciones
│   │   └── events.py               # eventos tipados de ejecución
│   ├── llm/
│   │   ├── base.py                 # protocolo LLMClient
│   │   └── openai_client.py        # adaptador actual
│   ├── tools/
│   │   ├── definitions.py          # contratos y metadatos de capacidad
│   │   ├── registry.py             # registro sin imports concretos
│   │   ├── factory.py              # registros por rol/política
│   │   ├── files.py
│   │   ├── commands.py
│   │   └── web.py
│   ├── policies/
│   │   ├── engine.py               # decisión uniforme allow/confirm/deny
│   │   ├── paths.py
│   │   ├── commands.py
│   │   └── supervision.py
│   ├── state/
│   │   ├── repository.py           # puerto para estado compartido
│   │   └── in_memory.py            # comportamiento inicial equivalente
│   ├── memory/
│   │   ├── repository.py           # puertos de sesión/memoria
│   │   └── persistent.py           # adaptador futuro
│   ├── rag/
│   │   ├── documents.py            # documentos, chunks y citas
│   │   ├── ingestion.py
│   │   ├── retriever.py            # puerto independiente del índice
│   │   └── indexes/                 # adaptadores futuros
│   ├── observability/
│   │   ├── events.py               # emitter y redacción
│   │   ├── metrics.py              # tokens, latencias y contadores
│   │   └── sinks.py                # consola y persistencia
│   └── interfaces/
│       └── cli.py                   # comandos/prompts actuales, sin lógica de dominio
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
└── workspace/
```

`main.py` debería conservarse inicialmente como fachada para mantener exactamente el
comando y la experiencia actual. Los módulos existentes pueden trasladarse detrás
de imports compatibles o envolverse antes de moverlos físicamente. El orden de
extracción recomendado para etapas posteriores es: eventos y estado explícito;
runtime desacoplado de la consola; política/capacidades; repositorios; recién después
delegación, persistencia y RAG. Cada etapa puede conservar los tests actuales como
contratos de regresión y agregar tests específicos, sin incorporar un framework de
orquestación.
