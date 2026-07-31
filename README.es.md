# local-img

[English](README.md) · **Español**

Generación local de imágenes a partir de texto. Prompt en el navegador, PNG a la
salida. Sin API keys, sin nube, sin filtrado de contenido. Después de la primera
descarga de pesos funciona completamente offline.

Los pesos vienen del Hugging Face Hub — repositorios públicos, sin token y sin
cuota — y cada imagen se genera en tu propia GPU. Acá Hugging Face es solo el
servidor de archivos, nunca una API de inferencia.

![La interfaz de local-img: prompt y configuración del modelo a la izquierda, el render
seleccionado con sus parámetros en el centro, y el historial local abajo](docs/screenshot.png)

## Descargar la app

La vía rápida: un archivo, sin terminal y sin Python.

| Plataforma | Archivo |
|---|---|
| macOS (Apple Silicon) | `.dmg` |
| Windows 10/11 (64 bits) | `.msi` |
| Linux (64 bits) | `.AppImage` o `.deb` |

**[Descargar la última versión →](https://github.com/giancarlobrusca/local-img/releases/latest)**

En el primer arranque la app descarga su propio motor de Python — alrededor de
1 GB en una Mac, o hasta 7 GB en una PC con placa NVIDIA, donde PyTorch trae la
build de CUDA, mucho más pesada — mide la máquina, recomienda un modelo que
entre y pregunta antes de descargarlo. De ahí en adelante funciona sin
conexión. Las imágenes se guardan en `Pictures/local-img`.

Las Mac con Intel no están soportadas: sin Metal cada imagen se generaría en la
CPU, que como explica la [nota sobre CPU](#requisitos) más abajo no es lento
sino impracticable.

### El aviso de seguridad

Los builds **no están firmados** — los certificados de firma cuestan dinero por
año y esto es una herramienta gratuita. Los dos sistemas avisan, una vez:

- **macOS** — "no se puede abrir local-img porque no se puede verificar al
  desarrollador". Hacé clic derecho sobre la app en Aplicaciones, elegí **Abrir**,
  y después **Abrir** otra vez en el diálogo. macOS se acuerda de la decisión.
- **Windows** — "Windows protegió su PC". Hacé clic en **Más información** y
  después en **Ejecutar de todas formas**.

Si preferís no pasar por eso, la instalación desde el código de abajo compila
todo en tu propia máquina.

## Requisitos

| | |
|---|---|
| Python | 3.11 o 3.12, **solo para la instalación desde el código** — la app trae el suyo |
| GPU | Apple Silicon (Metal/MPS) o NVIDIA (CUDA) |
| Memoria | Una GPU de 4 GB u 8 GB de memoria unificada alcanzan para SD 1.5; 16 GB para todos los modelos SDXL; 36 GB o más para la capa flux |
| Disco | ~2-4 GB por modelo SD 1.5, ~7 GB por modelo SDXL, 26-34 GB por modelo flux, más ~2.5 GB de dependencias de Python |

El backend se detecta automáticamente — CUDA, después MPS, después CPU — y se
muestra junto al título en la interfaz. Se puede forzar con
`LOCAL_IMG_DEVICE=cpu ./run.sh`.

**La CPU funciona, pero no es práctica.** No hay kernels fp16 para buena parte de
la UNet, así que SDXL cae a fp32: ~14 GB de RAM y varios minutos por imagen. Sin
GPU disponible, conviene usar LCM DreamShaper o DreamShaper 8 a 512px y tener
paciencia.

En GPU, la memoria es la restricción real. En la primera ejecución la app mide la
máquina — chip, núcleos de GPU, cuánta memoria va a entregar realmente Metal o
CUDA, y disco libre — y después muestra solo los modelos que entran, con el resto
listado junto al número que los bloquea. Nada sale de la computadora: el
resultado se guarda en `.local-img/profile.json` y se puede rehacer desde la barra
lateral cuando quieras. También se puede saltear el análisis y ver el catálogo
completo sin filtrar.

Se mantiene **un solo** pipeline residente por vez y se libera el anterior al
cambiar — dos SDXL no entran en 16 GB. Los modelos de arquitectura flux se cargan
con `enable_model_cpu_offload()`, así que solo un componente está en la GPU por
vez; por eso necesitan menos memoria de GPU que su tamaño de descarga, pero mucha
más RAM del sistema.

**Los tiempos empiezan siendo estimaciones.** Cada modelo muestra un tiempo por
imagen escalado desde una máquina de referencia (M1 Pro, GPU de 16 núcleos).
Después de generar tres imágenes con un modelo, la app pasa a usar la mediana de
tus propios tiempos registrados y lo aclara. Los factores de escalado para NVIDIA
siguen siendo mayormente conjeturas: se midió exactamente una máquina NVIDIA (una
RTX 3050 Laptop — ver las bases más abajo), y resultó cerca del **doble de lenta**
de lo que `cuda_perf_factor()` predice para una placa que no reconoce. Tomá el
número de la primera corrida en cualquier GPU NVIDIA como optimista hasta que tus
tres renders lo reemplacen.

## Desde el código

El flujo del repositorio, sin cambios: todo lo que hace la app de escritorio,
hecho a mano, en cualquier máquina que ya tenga Python 3.11 o 3.12.

```bash
./setup.sh                      # venv de Python 3.12 + torch/diffusers (~2.5 GB, unos minutos)
./download.sh                   # descarga previa del modelo por defecto (~6.9 GB, reanudable)
./run.sh                        # → http://127.0.0.1:7788
```

`setup.sh` busca Python 3.12 u 3.11 de forma explícita, en vez de usar lo que
apunte `python3`, porque PyTorch todavía no publica wheels para 3.13+.

La descarga previa es opcional pero recomendada: si se corta la conexión durante
una transferencia de 7 GB, el problema aparece después como un prompt fallido.
Ambos caminos reintentan y retoman desde la caché de Hugging Face, así que volver
a ejecutar nunca empieza de cero.

```bash
./download.sh juggernaut-xl-v9  # un modelo específico
./download.sh all               # todos los modelos (~104 GB)
./download.sh prune             # borra archivos cacheados que ningún pipeline carga
```

### Windows

Los scripts de shell son solo para macOS/Linux — un venv de Windows deja su
intérprete en `.venv\Scripts\`, no en `.venv/bin/`. El equivalente, en PowerShell:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download.py dreamshaper-8
.venv\Scripts\python.exe app.py                    # → http://127.0.0.1:7788
```

Instalar torch desde el índice de CUDA **primero** es el paso que importa. El
wheel de torch para Windows en PyPI es solo-CPU, así que un `pip install -r
requirements.txt` a secas deja una app que arranca, funciona y reporta en
silencio `device: cpu` — imágenes correctas a una fracción de la velocidad, sin
ningún error que explique por qué. Hacerlo en este orden conserva la build de
CUDA, porque el `torch>=2.4` posterior ya queda satisfecho. Elegí la etiqueta de
índice que corresponda a la versión de CUDA de tu driver, que `nvidia-smi`
imprime arriba a la derecha.

Nombrá el intérprete 3.12 explícitamente en vez de usar `py` o `python`: en una
máquina donde 3.13 es el default, torch no tiene wheel y la instalación falla al
resolver dependencias. Para forzar un dispositivo, definí la variable para la
sesión antes: `$env:LOCAL_IMG_DEVICE = "cpu"`.

## Modelos

Diez modelos, desde GPUs de portátil de 4 GB hasta estaciones de trabajo de 64 GB.
Todos los repositorios son públicos y resuelven sin token de Hugging Face. Los
repos con acceso restringido — FLUX.1-schnell, FLUX.1-dev y la familia Stable
Diffusion 3.5 — quedan deliberadamente afuera: devuelven `GatedRepoError: 401` de
forma anónima, y esta app no tiene dónde poner un token. La capa flux usa
derivados Apache-2.0 sin restricción en su lugar.

| Modelo | Arq. | Disco | Requiere | Base | Notas |
|---|---|---|---|---|---|
| LCM DreamShaper v7 | SD 1.5 | 4.3 GB | 3.5 GB GPU / 8 GB RAM | ~4 s | Destilado de consistencia latente, 4-8 pasos. El camino más rápido a 768px en una placa chica. MIT. |
| DreamShaper 8 | SD 1.5 | 2.1 GB | 3 GB GPU / 8 GB RAM | **~18 s** *(medido)* | La descarga más chica. Cobertura de conceptos muy amplia, ecosistema enorme de LoRA. |
| SD Turbo | SD 2.1 | 2.6 GB | 3 GB GPU / 8 GB RAM | ~2 s | 1-4 pasos a 512px, la huella de memoria más chica del set. Base entrenada con datos filtrados. |
| SDXL Turbo | SDXL | 6.9 GB | 9.5 GB GPU / 16 GB RAM | ~5 s | Previsualizaciones de 3 pasos a 512px. Ignora prompts negativos y CFG por diseño. |
| **DreamShaper XL v2 Turbo** | SDXL | 6.9 GB | 9.5 GB GPU / 16 GB RAM | **~40 s** *(medido)* | La mejor calidad por segundo. SDXL a 1024px en 7 pasos. |
| Playground v2.5 | SDXL | 7.0 GB | 9.5 GB GPU / 16 GB RAM | ~200 s | Entrenado desde cero para estética. Trae un scheduler EDM que necesita. |
| Juggernaut XL v9 | SDXL | 7.1 GB | 9.5 GB GPU / 16 GB RAM | ~180 s | Especialista en fotorrealismo. Muestreo completo de 30 pasos. |
| RealVisXL V4.0 | SDXL | 6.9 GB | 9.5 GB GPU / 16 GB RAM | ~180 s | El otro fine-tune fotorrealista de SDXL; mejor con personas. |
| Flex.1 alpha | flux, 8B | 26.3 GB | 20 GB GPU / 36 GB RAM | ~300 s | Derivado Apache-2.0 de schnell con un guidance embedder entrenado. |
| Shuttle 3 Diffusion | flux, 12B | 33.7 GB | 26 GB GPU / 48 GB RAM | ~120 s | Derivado Apache-2.0 de FLUX.1-schnell. La mejor adherencia al prompt, en 4 pasos. |

No hay un modelo por defecto fijo. La app recomienda el de mayor calidad que entre
en la máquina donde corre — DreamShaper XL v2 Turbo en un M1 Pro de 16 GB,
Shuttle 3 Diffusion en un M4 Max de 64 GB, DreamShaper 8 en una Mac de 8 GB.

Los tiempos base son por imagen en un **M1 Pro con 16 GB de memoria unificada** —
un punto de referencia, no una especificación, y la interfaz los escala a tu
máquina. Medido ahí: DreamShaper 8 a 512×512/20 pasos → **18.8 s**; DreamShaper
XL Turbo a 1024×1024/7 pasos → **39.5 s** y **42.8 s**. El resto está extrapolado
de esos costos por paso. Los números de flux son los menos confiables: ningún
modelo flux entra en los 11.8 GB utilizables de esta máquina, así que ese camino
no está probado acá y sus requisitos de memoria son estimaciones conservadoras.

Desde entonces se midió una máquina NVIDIA — una **RTX 3050 Laptop, 4 GB de
VRAM** (Windows 11, driver 577, CUDA 12.9), la placa más chica que el catálogo
admite. SD Turbo a 512×512/2 pasos → **1.1 s**; DreamShaper 8 a 512×768/25 pasos
→ **12.4 s**; LCM DreamShaper v7 a 768×768/6 pasos → **8.4 s**. El pico de
memoria de GPU sobre toda la placa, escritorio incluido, fue 2.9 / 3.6 / 3.8 GB:
las tres entradas de la familia SD 1.5 sí entran en una placa de 4 GB, pero la de
768px deja menos de 200 MB de margen, así que tener otra aplicación pesada en GPU
abierta al mismo tiempo es la diferencia entre un render y un error por falta de
memoria. Contra la única base que fue medida, el factor de escalado real de esa
placa es de alrededor de **1.4** — bastante por debajo del 3.0 que `hardware.py`
asume para una placa que no reconoce.

Los tamaños en disco son lo que el pipeline carga realmente. Los repositorios en
sí son mucho más grandes: dreamshaper-xl-v2-turbo son 41.6 GB completo, porque
además incluye tres checkpoints standalone de archivo único y copias fp32 de cada
componente que el pipeline fp16 nunca toca. `download.sh` descarga solo los
archivos necesarios, y `./download.sh prune` recupera el resto si alguna
herramienta más amplia ya los bajó.

### Sobre el filtrado de contenido

Nada en este stack filtra prompts ni imágenes:

- Los pipelines de SDXL no definen ningún safety checker.
- El pipeline de SD 1.5 se construye con `safety_checker=None, requires_safety_checker=False`
  (`app.py`), así que ninguna salida se difumina ni se reemplaza.
- No hay coincidencia de palabras clave sobre el prompt en ninguna parte de este código.

Los fine-tunes de la comunidad (DreamShaper, Juggernaut, RealVis) también se
entrenaron sin filtrado de contenido, así que la restricción no está apenas
desactivada en tiempo de ejecución: tampoco está en los pesos. Vale la pena
entenderlo, porque es una propiedad de *los pesos*, no un interruptor. FLUX.1-dev,
en cambio, se entrenó con datos curados, así que no puede producir lo que nunca
vio, sin importar la configuración del pipeline — los modelos de arquitectura
flux que hay acá son derivados de schnell, que están menos curados pero no sin
curar.

La salida es tuya, y la responsabilidad también. Generar en local elimina las
barreras del proveedor, no la ley: las imágenes no consentidas de personas reales
y el contenido sexual que involucra menores son delitos en prácticamente
cualquier jurisdicción, y que «corrió en mi laptop» no cambia nada.

## Uso

En la primera ejecución un asistente breve mide la máquina y muestra el catálogo
dividido en lo que recomienda, lo que también funciona, y lo que no entra y por
qué. Después de eso la barra lateral lleva un resumen del hardware en una línea;
al hacer clic se abre el perfil completo, un botón de *Re-analizar*, y una casilla
que muestra los modelos que no entran.

Panel izquierdo: prompt, modelo, prompt negativo y un desplegable de Settings con
pasos, guidance, dimensiones, cantidad de imágenes y seed. Al elegir un modelo se
cargan sus valores por defecto recomendados, y la cantidad de imágenes queda
limitada según la memoria medida. Las imágenes se generan de a una, así que el
tope limita cuántos renders encolás, no cuánta memoria necesita cada uno.
`Cmd+Enter` en el campo de prompt genera.

La barra de estado transmite el progreso paso a paso en vivo por SSE — útil,
porque un render SDXL de 30 pasos tarda un par de minutos. Las imágenes generadas
van a `outputs/` como PNG, más un `.json` hermano con todos los parámetros
(incluida la seed, así que cualquier imagen se puede reproducir). La tira de
miniaturas de abajo es el historial local; al hacer clic en una se reabre con su
metadata. Borrar elimina el PNG y su `.json` de forma definitiva: hay que pasar el
cursor por una miniatura para ver su `×`, o usar el botón `Delete` sobre la imagen
abierta.

La seed `-1` aleatoriza. Fijar una seed y variar un solo parámetro es la forma más
rápida de entender a qué responde un modelo.

## Estructura

```
app.py           servidor FastAPI — caché de pipelines, cola de trabajos, progreso por SSE
models.py        registro de modelos (repos, valores por defecto, tamaños, requisitos)
hardware.py      detección de la máquina, reglas de compatibilidad, estimaciones de tiempo
paths.py         dónde viven el perfil y los renders, a partir de dos variables de entorno
download.py      descarga previa de pesos, reanudable
delete_test.py   chequeos offline de la ruta de borrado y de la sesión
hardware_test.py chequeos offline de detección, fit, estimaciones y rutas
paths_test.py    chequeos offline de la resolución de rutas en los dos modos
shell_test.py    chequeos offline del puerto y del watchdog del proceso padre
web/index.html   toda la interfaz, sin paso de build
desktop/         el shell Tauri — instala Python, corre app.py, sin clonar nada
outputs/         PNGs generados + sus archivos de parámetros (ignorados por git)
.local-img/      el perfil de hardware detectado (ignorado por git)
```

## Notas

- **La primera generación con un modelo es lenta** — descarga los pesos y después
  carga ~7 GB en memoria. Los prompts siguientes reutilizan el pipeline residente.
- **Cambiar de modelo** descarga el anterior de memoria y recarga desde disco
  (~15–30 s). Conviene agrupar el trabajo por modelo.
- **Presión de memoria**: cerrar las sesiones pesadas del navegador antes de un
  render SDXL a 1024px. En memoria unificada sobre todo, cuando el sistema empieza
  a hacer swap, el tiempo de generación se duplica.
- **Los tamaños no cuadrados** deben ser múltiplos de 8 (la app redondea hacia
  abajo). SDXL se entrenó a ~1 megapíxel: 1024×1024, 1152×896 y 896×1152 son los
  que mejor se comportan; pasar de ~1536 en cualquier eje produce sujetos
  duplicados y arriesga un OOM.
- Agregar un modelo es una entrada nueva en `models.py`, siempre que el repo esté
  en formato diffusers (tenga un `model_index.json`) y sea arquitectura SD 1.5,
  SDXL o flux.
- **La app y el repositorio comparten los pesos.** Los dos usan
  `~/.cache/huggingface`, así que instalar la app después de haber usado el
  flujo del código no vuelve a descargar nada. La app guarda su propio perfil
  de hardware y pone los renders en `Pictures/local-img`.

## No despliegues esto

Corrélo en tu propia máquina. Está hecho para exactamente un usuario en
localhost, y hostearlo públicamente se rompe en varias direcciones a la vez:

- El servidor escucha en `127.0.0.1` y **no tiene autenticación ni límite de
  peticiones**. Un endpoint abierto que quema 40 s de GPU por request es una
  denegación de servicio gratis.
- `outputs/` es **global**. La galería le muestra a cada visitante las imágenes de
  todos los demás, y la ruta de borrado permite que cualquiera elimine archivos ajenos.
- Un único `GEN_LOCK` serializa toda la generación, así que los usuarios hacen cola
  unos detrás de otros.
- Los hosts serverless quedan descartados por completo: no hay disco persistente
  para 7 GB de pesos, además de límites de tamaño y timeouts de segundos.
- Cualquier cosa sin GPU cae a CPU y se arrastra, como se describe más arriba.

Volverlo multiusuario implica una cola real, almacenamiento por sesión,
autenticación y cuotas: otro proyecto. Y un generador de imágenes público, anónimo
y sin filtros conlleva una exposición legal muy distinta a la de la misma
herramienta en tu propia laptop.

## Licencia

MIT — ver [LICENSE](LICENSE). Los pesos de los modelos **no** están cubiertos por
ella; cada uno tiene su propia licencia en su repositorio de Hugging Face (los
modelos de Stability se publican bajo la licencia CreativeML OpenRAIL++-M, que
restringe algunos usos). Conviene revisar la del modelo que se vaya a usar, sobre
todo con fines comerciales.
