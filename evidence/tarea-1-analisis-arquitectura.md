# Evidencia — Tarea 1: análisis de arquitectura de PrintScript

**Fecha:** 2026-07-20
**Comando:** composición de `MainAgent` (Explorer + Researcher) desde `agent.config.yaml`, vía `run_agent.py`
**Modelo:** `gpt-5-nano` (OpenAI Responses API, real, no simulado)
**Estado final:** `completed`
**Agentes usados:** `explorer`, `researcher`

## Pedido

> Explicame la arquitectura, los módulos principales y el diseño de PrintScript, citando las
> fuentes RAG que uses (tanto la especificación del lenguaje como documentación de Kotlin si
> corresponde).

## Output relevante (resumido — el output completo lista los 194 archivos del repo)

### Explorer

> PrintScript es un proyecto Kotlin modular gestionado con Gradle. Presenta dos puntos de
> entrada (CLI y lexer) y una arquitectura basada en módulos claros (lexer, parser,
> interpreter, formatter, linter, runner, CLI adapters, etc.). La documentación clave incluye
> README y la especificación del lenguaje, y hay una batería de tests (56) distribuidos por
> módulos. La evidencia se sostiene en la estructura de archivos, las rutas de entry points y
> la presencia de archivos de configuración y docs.

Datos concretos detectados (con evidencia = rutas reales del repositorio, no inferencia):
- **194 archivos, 149 directorios**; lenguaje: Kotlin; build: Gradle; 2 entry points; 56 tests.
- **Módulos**: `lexer/`, `parser/`, `interpreter/`, `formatter/`, `linter/`, `runner/`, `cli/`,
  `token/`, `common/`, `buildSrc/` — cada uno con su propio `build.gradle`.
- **Entry points**: `cli/src/main/kotlin/org/printscript/cli/Main.kt`,
  `lexer/src/main/kotlin/org/printscript/lexer/Main.kt`.
- **Convenciones detectadas**: tests con sufijo `Test`, separación estricta de `src/`/`src/test`
  por módulo.
- **CI detectado**: `.github/workflows/ci.yml`, `.github/workflows/publish.yml`.

### Researcher

> PrintScript es un proyecto Kotlin/Gradle organizado en módulos que sugieren un pipeline
> típico de compilación/interpretación: Lexer, Parser, Formatter, Interpreter, Linter, CLI y
> runner [...] contiene referencias explícitas a una especificación del lenguaje
> (docs/printscript-language-spec.md) y a documentación de Kotlin para conceptos del lenguaje
> [...] No se identifican incompatibilidades explícitas entre fragmentos; sí hay incertidumbre
> sobre el diseño detallado y las interfaces entre módulos.

## Trazabilidad de fuentes (diferenciando origen, tal como pide la consigna)

| Origen | Cantidad | Ejemplos |
|---|---|---|
| `repository` (hallazgo directo del repo) | ~140 | `cli/src/main/kotlin/org/printscript/cli/Main.kt`, `build.gradle.kts`, `lexer/src/main/kotlin/org/printscript/lexer/Lexer.kt` |
| `project_memory` (memoria persistida de tareas previas) | 5 | `memory://printscript-.../previous_tasks/8`, `.../session_summaries/8` |
| `rag` (RAG real, recuperado y citado) | 2 | 2 chunks de `https://kotlinlang.org/docs/basic-syntax.html` (scores 0.393 y 0.327) |
| `inference` (razonamiento propio, marcado como tal) | 1 | síntesis final del researcher |

**Traza RAG completa**: query enviada al retriever, 2 chunks recuperados y 2 utilizados (100% de
aprovechamiento), ambos del documento `kotlinlang.org/docs/basic-syntax.html`, evaluados como
evidencia **suficiente** (confianza 0.68) — por eso **nunca se activó el fallback web**, tal
como exige el diseño (RAG primero, web sólo si falta evidencia).

## Qué se observa / conclusión

- El sistema diferenció correctamente 4 orígenes de información (repo, memoria, RAG, inferencia)
  en una sola corrida, citando rutas y URLs reales, no inventadas.
- La memoria persistente ya contenía resúmenes de sesiones anteriores y los reutilizó como
  evidencia adicional — confirma que `ProjectMemory` funciona entre corridas, no sólo dentro de
  una conversación.
- El Researcher marcó explícitamente la incertidumbre que no pudo resolver ("no hay diagrama de
  interacción entre módulos") en vez de inventar una respuesta — comportamiento esperado de
  `EvidenceSufficiencyPolicy`/`ThresholdSufficiencyEvaluator`.
- Pendiente: esta corrida usó `NoOpObservabilityClient` (sin Langfuse configurado todavía) —
  falta una corrida con observabilidad real para la captura del entregable #7.
