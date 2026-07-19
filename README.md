# Coding Agent — Parte 1

## Objetivo

Coding agent educativo implementado desde cero en Python, sin frameworks de
orquestación. Conecta la Responses API de OpenAI con tools locales mediante tool
calling, conserva el historial y ofrece planificación y supervisión interactivas.

El alcance de la Parte 1 es permitir que un usuario converse con el agente para
explorar y modificar proyectos ubicados en `workspace/`, ejecutar verificaciones
y consultar la web, manteniendo control explícito sobre la planificación y las
operaciones sensibles.

## Requisitos

- Python 3.10 o superior.
- Una API key de OpenAI y un modelo con soporte de tool calling.
- Una API key de Tavily para `web_search`.

## Instalación

Crear y activar un entorno virtual:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

En Windows PowerShell, la activación equivalente es:

```powershell
.venv\Scripts\Activate.ps1
```

`pyproject.toml` es la fuente principal de dependencias. Para instalar el proyecto
en modo editable con las dependencias de desarrollo:

```bash
python -m pip install -e ".[dev]"
```

Como alternativa, `requirements.txt` mantiene los mismos rangos de versiones:

```bash
python -m pip install -r requirements.txt
```

## Variables de entorno

El programa carga automáticamente el archivo `.env` de la raíz al iniciar. Se
puede crear a partir del ejemplo:

```bash
cp .env.example .env
# Completar .env sin commitearlo.
```

Las variables ya exportadas en el entorno tienen prioridad sobre los valores de
`.env`.

Variables requeridas:

- `OPENAI_API_KEY`: credencial de la API de OpenAI.
- `OPENAI_MODEL`: identificador del modelo que usará Responses API.
- `TAVILY_API_KEY`: credencial utilizada por la tool `web_search`.

No deben incluirse secretos reales en el código ni en archivos versionados.

## Ejecución

```bash
python main.py
```

Comandos interactivos:

- `/exit`: cerrar el chat.
- `/status`: mostrar la configuración activa.
- `/plan on` y `/plan off`: activar o desactivar plan mode.
- `/supervision on` y `/supervision off`: activar o desactivar supervisión.

Los mensajes vacíos se ignoran. `Ctrl+C` y EOF cierran la sesión de forma
controlada.

## Arquitectura

- `main.py`: loop externo, comandos y diálogo interactivo.
- `core/harness.py`: planificación y loop interno de tool calling.
- `core/llm_client.py`: protocolo propio y adaptador de OpenAI.
- `core/models.py`: modelos independientes del SDK del proveedor.
- `core/supervision.py`: aprobación de operaciones modificadoras.
- `tools/`: implementaciones y registro de tools.
- `security/`: políticas de rutas y comandos.
- `workspace/`: único directorio de trabajo permitido para las tools locales.
- `tests/`: suite unitaria con pytest y proveedores externos simulados.

### Loop externo

Inicializa el historial con un system prompt, recibe mensajes, procesa comandos
locales, agrega cada pedido y ejecuta un turno. Después muestra la respuesta final
y la cantidad de iteraciones, conservando el historial para el mensaje siguiente.

### Loop interno

Envía el historial y los schemas al LLM, registra la respuesta, valida y ejecuta
las tool calls, agrega sus resultados correlacionados al historial y repite hasta
obtener una respuesta sin tools. `max_iterations` evita ciclos indefinidos.

## Tools disponibles

- `read_file`: lee un archivo UTF-8 dentro de `workspace/`.
- `write_file`: reemplaza un archivo dentro de `workspace/`.
- `list_files`: lista un directorio dentro de `workspace/`.
- `run_command`: ejecuta un comando validado con `cwd=workspace/` y `shell=False`.
- `web_search`: consulta Tavily y devuelve resultados breves y estructurados.

## Plan mode

Cuando está activo, el LLM propone un plan sin recibir schemas de tools. El usuario
puede aprobarlo, rechazarlo o pedir una modificación. Ninguna tool se expone ni se
ejecuta antes de aprobar. Sólo el plan aprobado se incorpora al contexto principal.

## Supervision mode

Cuando está activo, `write_file` y `run_command` requieren confirmación previa.
`read_file`, `list_files` y `web_search` se consideran de sólo lectura y no piden
aprobación. Desactivar supervisión no desactiva las políticas de seguridad.

## Seguridad del workspace

Las file tools resuelven rutas dentro de `workspace/`, rechazan escapes mediante
`..`, rutas absolutas externas, symlinks que escapen y archivos `.env`.

`run_command` usa una política previa que revisa el comando completo. Bloquea
operaciones destructivas conocidas, `git push`, `git reset --hard`, ejecución de
código inline mediante intérpretes, wrappers de subcomandos, secretos configurados
y rutas externas. Siempre usa `shell=False`, timeout y `cwd=workspace/`.

Esta es una política de validación defensiva para la Parte 1, no un sandbox fuerte
del sistema operativo. Un ejecutable o script permitido puede tener comportamiento
arbitrario que el análisis de argumentos no detecte. Para ejecutar código no
confiable se necesitaría aislamiento adicional mediante contenedores, namespaces,
permisos del sistema operativo u otra tecnología fuera del alcance de esta parte.

## Tests

```bash
python -m pytest tests
```

Los clientes de OpenAI y Tavily se prueban con mocks; la suite no consume APIs ni
internet. Los tests cubren loops, historial, planificación, supervisión, tools,
límites, errores y políticas de seguridad.

## Ejemplos de uso

Los siguientes pedidos se pueden ingresar en el chat después de ejecutar
`python main.py`. Los resultados dependen del modelo configurado y deben revisarse
antes de aprobar planes u operaciones sensibles.

### Ejemplo 1: diagnosticar y corregir un bug

El proyecto inicial está en `workspace/bug_demo/`. Prompt sugerido:

> Explorá el proyecto bug_demo, encontrá por qué fallan los tests, corregí el
> problema y verificá que todos los tests pasen.

### Ejemplo 2: implementar funcionalidad nueva

El proyecto inicial está en `workspace/palindrome_demo/`. Prompt sugerido:

> En palindrome_demo, creá una función is_palindrome que ignore mayúsculas,
> espacios y signos de puntuación. Agregá tests y verificá que pasen.

Estos ejemplos no documentan resultados todavía. Las dos corridas entregables se
registrarán una vez terminada la implementación completa de la Parte 1.

## Evidencias

Esta sección queda preparada para pegar evidencias reales. No completar campos sin
haber ejecutado y verificado la corrida correspondiente.

### Corrida 1

- Output o enlace a la evidencia:
- Cantidad de iteraciones:
- Qué salió bien:
- Qué salió mal:

### Corrida 2

- Output o enlace a la evidencia:
- Cantidad de iteraciones:
- Qué salió bien:
- Qué salió mal:

### Mejoras futuras

- Pendiente de completar después de analizar ambas corridas.

## Limitaciones y trabajo futuro

- `run_command` aplica reglas explícitas, pero no es un sandbox de sistema operativo.
- `.env` se carga al iniciar y debe permanecer fuera del control de versiones.
- La calidad de planes y tool calls depende del modelo configurado.
- Las corridas demostrativas todavía no tienen evidencias ni resultados documentados.
- Funcionalidades adicionales quedan fuera del alcance de la Parte 1.
