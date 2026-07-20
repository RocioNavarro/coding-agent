# Reflexión final — TP Final: Coding Agent Avanzado

## Qué funcionó bien

- **La arquitectura multiagente completa funciona de punta a punta, sin frameworks de
  orquestación**: `MainAgent` coordinando Explorer y Researcher reales, contra un repositorio
  externo real (PrintScript) y un LLM real (`gpt-5-nano`), no simulado.
- **El RAG es real, no de juguete**: chunking, embeddings, vector store con dos fuentes
  genuinas —la especificación del lenguaje entregada por la cátedra y documentación oficial
  de Kotlin descargada en vivo de kotlinlang.org—, con trazabilidad explícita de qué chunk se
  recuperó, cuál se usó, y con qué score y nivel de confianza.
- **La memoria persistente funciona entre corridas reales**, no sólo dentro de una
  conversación: reutilizó resúmenes de tareas anteriores como evidencia adicional en corridas
  posteriores.
- **El sistema no alucina cuando no tiene evidencia**: al preguntarle por concurrencia y
  multithreading en PrintScript —algo que ni la especificación del lenguaje ni la
  documentación de Kotlin indexada cubren— el agente respondió honestamente que esa
  información no está definida, citando la ausencia como hallazgo en vez de inventar una
  respuesta plausible.
- **La observabilidad con Langfuse quedó funcionando de verdad**, con jerarquía de trazas
  (task → agente → llm_call/rag) visible en el dashboard, no sólo declarada en el código.
- **Las políticas de seguridad se sostienen bajo prueba real**: bloqueo de rutas fuera del
  workspace, de archivos sensibles y de comandos destructivos verificado con el pipeline
  corriendo contra un repositorio real de 194 archivos.

## Qué falló (y qué aprendimos de cada falla)

1. **Bug real: el prompt no restringía los valores permitidos de `origin` en las fuentes.**
   El LLM devolvía un valor fuera del enum esperado (`repository`/`project_memory`/`rag`/`web`/
   `inference`), y la validación estricta —correcta en su intención— rompía la ejecución sin
   dar oportunidad de corregirse. Se arregló explicitando los 5 valores literales permitidos
   directamente en la instrucción de salida (`agents/base.py`).

2. **Bug real, más serio: los subagentes ofrecían tools al LLM que nunca se ejecutaban.**
   Ante preguntas más específicas o difíciles, el modelo elegía pedir tool calls (por ejemplo,
   buscar archivos relacionados a "concurrencia") en vez de responder directo. El framework
   validaba esas tool calls pero jamás las ejecutaba ni retomaba la conversación con el
   resultado — la respuesta de texto quedaba vacía y el parseo de JSON fallaba siempre, de
   forma determinista y reproducible (no era azar del modelo). Se diagnosticó inspeccionando
   `response.tool_calls` directamente y se arregló implementando un loop real de ejecución de
   tools en `BaseAgent`, acotado a 3 rondas, reutilizando el mismo patrón que ya existía en el
   harness de la Parte 1. Verificado: la misma pregunta que fallaba consistentemente ahora
   completa siempre.

3. **Errores intermitentes de cuota de OpenAI en una cuenta recién creada.** Se dedicó tiempo
   real a diferenciar esto de un bug propio: se probó el mismo código en el mismo instante,
   invocado de formas distintas, con resultados distintos — la evidencia (código idéntico,
   resultados distintos) descarta un bug determinista de nuestro lado y apunta a inestabilidad
   propia de una cuenta nueva de OpenAI en sus primeras horas de uso real.

4. **Auditoría de calidad encontró (y en parte corrigió) deuda técnica real**: una
   dependencia circular genuina entre `rag/` y `agents/` (resuelta moviendo los contratos
   compartidos a un módulo neutral, `core/research_ports.py`, con prueba empírica: los imports
   tardíos que antes eran obligatorios dejaron de serlo); manejo de errores inconsistente
   entre subagentes (Tester y Reviewer no registraban sus fallos, a diferencia de los otros
   tres — unificado con un context manager compartido); funciones de normalización duplicadas
   en tres archivos distintos (unificadas en `core/validation.py`).

## Cuándo se detectó falta de evidencia

- La corrida sobre "concurrencia en PrintScript" (arriba) es el caso concreto: el `Researcher`
  evaluó la evidencia disponible como insuficiente para afirmar la existencia de mecanismos de
  concurrencia, y tanto Explorer como Researcher reportaron la ausencia explícitamente en vez
  de completar el hueco con inferencia no marcada como tal.
- **No llegamos a ejercitar `ProgressMonitor` (detección de loops/estancamiento) en una
  corrida real** contra PrintScript, porque el caso de uso elegido quedó acotado
  deliberadamente a análisis puro (Explorer + Researcher), sin conectar Implementer/Tester —
  decisión consciente para reducir riesgo cerca de la entrega, no una limitación descubierta
  tarde. Esa capacidad sí está implementada y cubierta por un escenario de test dedicado
  (`tests/integration/test_failed_command_scenario.py`), pero no con evidencia de una corrida
  real end-to-end.

## Qué mejoraríamos con más tiempo

- Conectar Implementer/Tester/Reviewer al caso de uso de PrintScript, verificando primero el
  comando real de build/test de Gradle, para poder demostrar también un cambio de código real
  validado (y ahí sí ejercitar `ProgressMonitor` con una corrida real, no sólo con tests).
- Reemplazar la validación de salida basada únicamente en texto de prompt por el modo de
  salida estructurada estricta que ofrece la API (JSON Schema forzado), para eliminar de raíz
  la fragilidad que encontramos hoy en vez de sólo mitigarla con reintentos.
- Terminar de refactorizar `MainAgent.run()` (hoy concentra demasiadas fases en un solo método
  de ~300 líneas) en pasos independientes y testeables por separado.
- Resolver la violación de LSP en `BaseAgent` (cada subagente devuelve un tipo de resultado
  distinto pese a heredar una firma común) con composición o genéricos en vez de herencia de
  método fijo — hoy mitigado en la práctica porque el orquestador ya usa `Protocol`s
  estructurales en vez de depender del tipo de la clase base.
- Reemplazar el acoplamiento implícito entre Explorer y Tester (que hoy se comunican
  parseando con regex el texto libre de las observaciones registradas) por un campo
  estructurado y tipado en `TaskState`.
