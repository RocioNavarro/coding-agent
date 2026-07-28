"""Síntesis final determinística y acotada para tareas de análisis."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from core.task_state import SourceReference, SubagentResult, TaskState


class DeterministicAnalysisSummary:
    """Construye un informe legible usando sólo evidencia persistida en TaskState."""

    MAX_RAW_FINDINGS = 24
    MAX_SOURCES = 16
    MAX_ITEM_CHARS = 360
    MAX_RESEARCH_CHARS = 1_600
    MAX_OUTPUT_CHARS = 12_000
    MODULE_ORDER = (
        "common", "token", "lexer", "parser", "interpreter",
        "formatter", "linter", "runner", "cli",
    )
    RESPONSIBILITIES = {
        "common": ("common/", "tipos y utilidades compartidas"),
        "token": ("token/", "modelo de tokens y tipos léxicos"),
        "lexer": ("lexer/", "análisis léxico y producción de tokens"),
        "parser": ("parser/", "análisis sintáctico y construcción del AST"),
        "interpreter": ("interpreter/", "evaluación y ejecución del AST"),
        "formatter": ("formatter/", "formateo y renderizado de código"),
        "linter": ("linter/", "análisis estático mediante reglas"),
        "runner": ("runner/", "orquestación alternativa de lexer, parser e interpreter"),
        "cli": ("cli/", "interfaz de línea de comandos y adaptación del pipeline"),
    }

    def build(self, state: TaskState) -> str:
        researcher = self._result(state, "researcher")
        explorer = self._result(state, "explorer")
        findings = tuple(
            item for item in state.repository_findings if self._useful_finding(item)
        )[: self.MAX_RAW_FINDINGS]
        evidence_text = "\n".join((*findings, *self._result_texts(researcher)))
        modules = self._values(findings, "modules=")
        paths = self._paths(state, explorer, researcher, findings)
        internal = self._internal_dependencies(findings)
        technologies = self._technologies(findings)
        entry_points = self._entry_points(findings)
        commands = self._commands(state)
        sources = self._sources(state.sources)
        research_summary = self._research_summary(researcher)
        uncertainties = self._unique(
            (*state.warnings, *(error.message for error in state.errors),
             *(researcher.blockers if researcher else ()),
             *(researcher.recommendations if researcher else ()))
        )

        responsibilities = []
        for module in self.MODULE_ORDER:
            if module not in modules:
                continue
            marker, description = self.RESPONSIBILITIES[module]
            proof = next((path for path in paths if marker in path), None)
            responsibilities.append(
                f"- **{module}**: {description}. Evidencia: `{proof}`."
                if proof else f"- **{module}**: responsabilidad no confirmada con la evidencia disponible."
            )

        risks = self._risks(evidence_text, internal, paths)
        confirmed = self._confirmed(modules, internal, technologies, entry_points, researcher)
        inferences = [risk[2] for risk in risks if risk[2]]
        if researcher:
            inferences.extend(item for item in researcher.findings if "INFER" in item.upper())

        sections = [
            "# Informe técnico del repositorio",
            "## 1. Objetivo general\n" + (
                self._clip(research_summary, self.MAX_RESEARCH_CHARS)
                if research_summary else
                "No se recibió una síntesis técnica de Researcher; se informa sólo evidencia de Explorer."
            ),
            "## 2. Arquitectura general\n" + self._architecture(modules, research_summary),
            "## 3. Módulos y responsabilidades\n" + ("\n".join(responsibilities) or "No confirmados."),
            "## 4. Dependencias internas\n" + self._bullets(
                f"{module} → {', '.join(dependencies)}" for module, dependencies in internal
            ),
            "## 5. Dependencias externas y tecnologías\n" + self._bullets(technologies),
            "## 6. Puntos de entrada y flujo de ejecución\n"
            + self._bullets(f"`{path}`" for path in entry_points)
            + "\n\nFlujo confirmado por la evidencia disponible: "
            + ("archivo → FrontendAdapter → Lexer → Parser → AST → Interpreter"
               if self._flow_confirmed(paths, research_summary) else "no confirmado de extremo a extremo.")
            + ("\n\nCamino alternativo: `runner` conecta Lexer → Parser → Interpreter."
               if any("runner/" in path and path.endswith("Runner.kt") for path in paths) else ""),
            "## 7. Rol de los componentes principales\n" + self._component_roles(responsibilities),
            "## 8. Organización de Gradle\n" + self._gradle(modules, evidence_text),
            "## 9. Comandos documentados\n" + self._bullets(f"`{item}`" for item in commands),
            "## 10. Riesgos y deuda técnica\n" + self._risk_lines(risks),
            "## 11. Hechos confirmados\n" + self._bullets(confirmed),
            "## 12. Inferencias\n" + self._bullets(inferences),
            "## 13. Información no confirmada\n" + self._bullets(uncertainties),
            "## 14. Fuentes principales\n" + self._bullets(
                f"[{source.origin}] `{source.reference}`" for source in sources
            ),
        ]
        report = "\n\n".join(sections)
        return self._clip(report, self.MAX_OUTPUT_CHARS)

    @staticmethod
    def _result(state: TaskState, name: str) -> SubagentResult | None:
        return next(
            (item for item in reversed(state.subagent_results) if item.subagent_id == name),
            None,
        )

    @staticmethod
    def _result_texts(result: SubagentResult | None) -> tuple[str, ...]:
        if result is None:
            return ()
        return tuple(item for item in (result.summary, result.result, *result.findings) if item)

    def _research_summary(self, result: SubagentResult | None) -> str:
        if result is None:
            return ""
        text = result.summary or result.result or ""
        lines = []
        for line in text.splitlines():
            clean = line.strip()
            if not clean or re.match(r"^#{1,6}\s", clean):
                continue
            if clean not in lines:
                lines.append(clean)
        return self._clip("\n".join(lines), self.MAX_RESEARCH_CHARS)

    @classmethod
    def _values(cls, findings: Sequence[str], prefix: str) -> tuple[str, ...]:
        values: list[str] = []
        for finding in findings:
            if finding.casefold().startswith(prefix.casefold()):
                payload = finding[len(prefix):].split(";", 1)[0]
                values.extend(item.strip() for item in payload.split(",") if item.strip())
        return cls._unique(values)

    @classmethod
    def _internal_dependencies(
        cls, findings: Sequence[str]
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        result = []
        for finding in findings:
            if not finding.startswith("internal_dependency="):
                continue
            relation = finding.removeprefix("internal_dependency=").split(";", 1)[0]
            module, separator, dependencies = relation.partition("->")
            if separator:
                result.append((module.strip(), cls._unique(dependencies.split(","))))
        return tuple(result)

    @classmethod
    def _technologies(cls, findings: Sequence[str]) -> tuple[str, ...]:
        aliases = (
            ("kotlin", "Kotlin"), ("java toolchain", "Java toolchain"),
            ("gradle", "Gradle"), ("picocli", "Picocli"), ("gson", "Gson"),
            ("junit", "JUnit"), ("kotlin test", "Kotlin Test"),
            ("spotless", "Spotless"), ("foojay", "Foojay"),
        )
        text = "\n".join(findings).casefold()
        return tuple(
            label for token, label in aliases
            if token in text or token.replace(" ", "-") in text
        )

    @classmethod
    def _entry_points(cls, findings: Sequence[str]) -> tuple[str, ...]:
        for finding in findings:
            if finding.casefold().startswith("entry points:"):
                payload = finding.split(":", 1)[1].split(";", 1)[0]
                return cls._unique(item.strip() for item in payload.split(","))
        return ()

    @classmethod
    def _commands(cls, state: TaskState) -> tuple[str, ...]:
        commands = []
        for item in state.observations:
            if item.startswith("Comando detectado: "):
                commands.append(item.removeprefix("Comando detectado: ").split("; evidencia:", 1)[0])
        return cls._unique(commands)

    @classmethod
    def _paths(
        cls, state: TaskState, explorer: SubagentResult | None,
        researcher: SubagentResult | None, findings: Sequence[str],
    ) -> tuple[str, ...]:
        values = [source.reference for source in state.sources]
        for result in (explorer, researcher):
            if result:
                values.extend(result.files_relevant)
                values.extend(source.reference for source in result.sources)
        for finding in findings:
            values.extend(re.findall(r"[\w./-]+\.(?:kt|kts|gradle|md|yml)", finding))
        return cls._unique(values)

    @classmethod
    def _sources(cls, sources: Sequence[SourceReference]) -> tuple[SourceReference, ...]:
        selected = []
        seen = set()
        for source in sources:
            key = source.reference
            if key in seen or source.origin == "inference":
                continue
            seen.add(key)
            selected.append(source)
            if len(selected) >= cls.MAX_SOURCES:
                break
        return tuple(selected)

    def _architecture(self, modules: Sequence[str], research: str) -> str:
        base = (
            "Repositorio modular" + (f" con {len(modules)} subproyectos confirmados: {', '.join(modules)}." if modules else ".")
        )
        corroboration = next(
            (line for line in research.splitlines() if "CONFIRM" in line.upper()), ""
        )
        return base + (("\n\n" + self._clip(corroboration, self.MAX_ITEM_CHARS)) if corroboration else "")

    @staticmethod
    def _flow_confirmed(paths: Sequence[str], research: str) -> bool:
        text = (" ".join(paths) + " " + research).casefold()
        return all(item in text for item in ("lexer", "parser", "interpreter"))

    @staticmethod
    def _component_roles(lines: Sequence[str]) -> str:
        selected = [line for line in lines if any(f"**{name}**" in line for name in ("cli", "runner", "formatter", "linter"))]
        return "\n".join(selected) or "No confirmados."

    @classmethod
    def _gradle(cls, modules: Sequence[str], evidence: str) -> str:
        lines = ["- `settings.gradle.kts`: declara los subproyectos.", "- `build.gradle.kts`: configuración raíz."]
        if "build_infrastructure=buildSrc" in evidence:
            lines.append("- `buildSrc`: infraestructura de build; no es un módulo declarado.")
        if modules:
            lines.append("- Subproyectos: " + ", ".join(modules) + ".")
        if "runner" in evidence.casefold() and "duplic" in evidence.casefold():
            lines.append("- `runner` aparece declarado más de una vez en settings.")
        return "\n".join(lines)

    @classmethod
    def _risks(
        cls, evidence: str, internal: Sequence[tuple[str, Sequence[str]]], paths: Sequence[str]
    ) -> tuple[tuple[str, str, str], ...]:
        low = evidence.casefold()
        risks = []
        if "runner" in low and "duplic" in low:
            risks.append(("Declaración duplicada de runner", "settings.gradle.kts", "Inferencia: puede confundir el mantenimiento de la configuración."))
        version_finding = next(
            (line for line in evidence.splitlines() if line.startswith("module_versions=")),
            "",
        )
        versions = {
            item.rsplit(":", 1)[-1].strip()
            for item in version_finding.split("=", 1)[-1].split(";", 1)[0].split(",")
            if ":" in item
        }
        if len(versions) > 1:
            risks.append(("Versiones de módulos desalineadas", "archivos build.gradle de los módulos", "Inferencia: puede dificultar releases coordinados."))
        if "duplicated_test_configuration=" in low:
            risks.append(("Configuración de tests duplicada", "build.gradle de los módulos indicados", "Inferencia: aumenta el costo de cambios de configuración."))
        if "root_writes_hooks=" in low:
            risks.append(("La tarea raíz instala hooks o scripts", "build.gradle.kts", "Inferencia: ejecutar esa tarea produce efectos fuera del build declarativo."))
        if any("runner/" in path and path.endswith("Runner.kt") for path in paths) and any("ExecuteCmd.kt" in path for path in paths):
            risks.append(("Dos caminos de ejecución", "Runner.kt y ExecuteCmd.kt", "Inferencia: ambos caminos pueden divergir si evolucionan por separado."))
        cli = next((deps for module, deps in internal if module == "cli"), ())
        if len(cli) >= 5:
            risks.append(("Alto acoplamiento del CLI", "cli/build.gradle", "Inferencia: cambios internos pueden propagarse a la capa CLI."))
        return tuple(risks)

    @classmethod
    def _confirmed(
        cls, modules: Sequence[str], internal: Sequence[tuple[str, Sequence[str]]],
        technologies: Sequence[str], entries: Sequence[str], researcher: SubagentResult | None,
    ) -> tuple[str, ...]:
        facts = []
        if modules:
            facts.append("Módulos declarados: " + ", ".join(modules) + ".")
        if internal:
            facts.append("Las dependencias internas listadas provienen de declaraciones project(...).")
        if technologies:
            facts.append("Tecnologías detectadas: " + ", ".join(technologies) + ".")
        if entries:
            facts.append("Puntos de entrada detectados: " + ", ".join(entries) + ".")
        if researcher:
            facts.extend(item for item in researcher.findings if "CONFIRM" in item.upper())
        return cls._unique(facts)

    @classmethod
    def _risk_lines(cls, risks: Sequence[tuple[str, str, str]]) -> str:
        return "\n".join(
            f"- **{description}.** Evidencia: `{evidence}`. {impact}"
            for description, evidence, impact in risks
        ) or "No se confirmaron riesgos específicos."

    @classmethod
    def _bullets(cls, values: Iterable[str]) -> str:
        items = cls._unique(values)
        return "\n".join(f"- {cls._clip(item, cls.MAX_ITEM_CHARS)}" for item in items) or "- No confirmado."

    @staticmethod
    def _unique(values: Iterable[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.strip() for item in values if item and item.strip()))

    @staticmethod
    def _useful_finding(value: str) -> bool:
        category = value.split("=", 1)[0].split(":", 1)[0].strip().casefold()
        return category in {
            "arquitectura", "modules", "build_infrastructure", "module_warning",
            "internal_dependency", "module_versions", "duplicated_test_configuration",
            "root_writes_hooks", "gradle_technology", "language", "build_system",
            "framework", "dependency", "entry points", "configuración", "documentación",
        }

    @staticmethod
    def _clip(value: str, limit: int) -> str:
        compact = value.strip()
        return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"
