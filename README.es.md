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

## Requisitos

| | |
|---|---|
| Python | 3.11 o 3.12 (PyTorch no publica wheels para 3.13/3.14) |
| GPU | Apple Silicon (Metal/MPS) o NVIDIA (CUDA) |
| Memoria | 16 GB unificados, u 8 GB de VRAM, para los modelos SDXL; aproximadamente la mitad para SD 1.5 |
| Disco | ~2 GB por modelo SD 1.5, ~7 GB por modelo SDXL, más ~2.5 GB de dependencias de Python |

El backend se detecta automáticamente — CUDA, después MPS, después CPU — y se
muestra junto al título en la interfaz. Se puede forzar con
`LOCAL_IMG_DEVICE=cpu ./run.sh`.

**La CPU funciona, pero no es práctica.** No hay kernels fp16 para buena parte de
la UNet, así que SDXL cae a fp32: ~14 GB de RAM y varios minutos por imagen. Sin
GPU disponible, conviene usar DreamShaper 8 a 512px y tener paciencia.

En GPU, la memoria es la restricción real. SDXL en fp16 son ~7 GB de pesos más
activaciones, así que la app mantiene **un solo** pipeline residente por vez y
libera el anterior al cambiar — dos SDXL no entran en 16 GB. FLUX (12B) queda
deliberadamente afuera: necesita cuantización de 4 bits para entrar en este tipo
de hardware y aun así tarda varios minutos por imagen.

## Instalación

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
./download.sh all               # todos los modelos (~30 GB)
./download.sh prune             # borra archivos cacheados que ningún pipeline carga
```

## Modelos

Elegidos para entrar en una GPU de portátil de 16 GB — todo lo de acá se mantiene
en memoria y termina en un tiempo razonable.

| Modelo | Disco | Velocidad | Notas |
|---|---|---|---|
| **DreamShaper XL v2 Turbo** *(por defecto)* | 6.9 GB | **~40 s** *(medido)* | El mejor equilibrio calidad/velocidad. SDXL a 1024px en 7 pasos. |
| DreamShaper 8 (SD 1.5) | 2.1 GB | **~18 s** *(medido)* | El más chico y el más rápido. Cobertura de conceptos muy amplia, ecosistema enorme de LoRA. |
| Juggernaut XL v9 | 7.1 GB | ~3 min *(estimado)* | Especialista en fotorrealismo. Muestreo completo de 30 pasos. |
| RealVisXL V4.0 | 6.9 GB | ~3 min *(estimado)* | El otro fine-tune fotorrealista de SDXL; mejor con personas. |
| SDXL Turbo | 6.9 GB | ~5 s *(estimado)* | Previsualizaciones de 3 pasos a 512px. Ignora prompts negativos y CFG por diseño. |

Los tiempos son de un **M1 Pro con 16 GB de memoria unificada** — hay que tomarlos
como punto de referencia, no como especificación. Medido ahí: DreamShaper 8 a
512×512/20 pasos → **18.8 s**; DreamShaper XL Turbo a 1024×1024/7 pasos →
**39.5 s** y **42.8 s**. Los dos modelos fotorrealistas están extrapolados del
costo por paso del XL Turbo, así que esos números son estimados. Una placa NVIDIA
reciente es bastante más rápida en todos los casos.

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
vio, sin importar la configuración del pipeline.

La salida es tuya, y la responsabilidad también. Generar en local elimina las
barreras del proveedor, no la ley: las imágenes no consentidas de personas reales
y el contenido sexual que involucra menores son delitos en prácticamente
cualquier jurisdicción, y que «corrió en mi laptop» no cambia nada.

## Uso

Panel izquierdo: prompt, modelo, prompt negativo y un desplegable de Settings con
pasos, guidance, dimensiones, cantidad de imágenes y seed. Al elegir un modelo se
cargan sus valores por defecto recomendados. `Cmd+Enter` en el campo de prompt
genera.

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
models.py        registro de modelos (ids de repo, valores por defecto, tamaños)
download.py      descarga previa de pesos, reanudable
delete_test.py   chequeos offline de la ruta de borrado (sin trabajo de GPU)
web/index.html   toda la interfaz, sin paso de build
outputs/         PNGs generados + sus archivos de parámetros (ignorados por git)
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
  en formato diffusers (tenga un `model_index.json`) y sea arquitectura SD 1.5 o SDXL.

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
