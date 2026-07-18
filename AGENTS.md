# Coding Agent - Parte 1

## Objetivo

Implementar desde cero un coding agent en Python, sin frameworks de
orquestación como LangChain, LangGraph, CrewAI, AutoGen ni similares.

El programa debe conectar un LLM con herramientas locales mediante
tool calling.

## Requisitos obligatorios

El programa debe tener dos loops:

1. Un loop externo que funcione como chat interactivo, reciba mensajes
   nuevos y conserve el historial.
2. Un loop interno que llame al LLM, detecte tool calls, ejecute las
   herramientas, devuelva sus resultados y repita hasta obtener una
   respuesta final sin tool calls.

Debe implementar estas tools:

- read_file: lee un archivo.
- write_file: reemplaza el contenido de un archivo.
- list_files: lista los archivos de un directorio.
- run_command: ejecuta un comando y devuelve exit code, stdout y stderr.
- web_search: busca información en la web.

También debe incluir:

- Plan mode activable y desactivable.
- Aprobación del plan antes de ejecutar tools.
- Supervision mode activable y desactivable.
- Con supervisión activa, pedir confirmación antes de write_file y
  run_command.
- Las tools de lectura se pueden ejecutar sin aprobación.
- Historial entre mensajes.
- Manejo de errores.
- Límite de iteraciones.
- Registro visible de tools e iteraciones.
- Dos tareas diferentes de demostración.

## Seguridad

- Las tools locales solo pueden operar dentro de workspace/.
- Bloquear rutas que intenten escapar mediante ../.
- No acceder a .env ni secretos.
- No ejecutar comandos destructivos.
- No hacer git push.
- No incluir API keys en el código.
- Leer secretos desde variables de entorno.
- No afirmar que algo funciona sin ejecutar pruebas.

## Calidad

- Usar Python 3.10 o superior.
- Usar type hints.
- Agregar docstrings relevantes.
- Separar responsabilidades en módulos.
- Agregar tests con pytest.
- Trabajar por etapas.
- Al finalizar cada etapa, ejecutar tests y mostrar git diff.
- No avanzar automáticamente a la etapa siguiente.
