# Módulo 01: Preparación de datos

*Documento del informe y guía de estudio. Un solo MD por módulo: propósito, preconceptos, decisiones, resultados, limitaciones y glosario. Última actualización: miércoles 11 de noviembre de 2026.*

## 1. Propósito

Dejar los datos listos para los demás módulos, con tres garantías:

1. **Partición sin fuga:** ninguna foto (ni copia de ella) aparece en dos conjuntos.
2. **Cachés reproducibles** para la CNN (módulo 02) y la U-Net (módulo 03).
3. **Máscaras de segmentación** para la U-Net, porque SylFishBD solo trae la etiqueta de clase.

Entradas: 9075 imágenes JPG de 500×500×3 en 9 clases. Salidas: `splits.csv`, `excluidas.csv`, cachés `.npz`, `masks.zip`, `mask_qc.csv`, `mask_params.json`, y las tablas y figuras de `reports/`.

## 2. Preconceptos

### 2.1 Fuga de datos

El objetivo es medir qué tan bien generaliza el sistema a imágenes que nunca vio. Si una foto (o una copia casi idéntica) está en entrenamiento y en prueba, el modelo la "reconoce" en lugar de generalizar y la métrica sale inflada. A esto se le llama **fuga de datos** (*data leakage*). SylFishBD contiene muchas copias y casi copias, por lo que una partición aleatoria por imagen no es válida y hay que partir **por grupos**.

### 2.2 Duplicados exactos: MD5

Una **función hash** convierte un archivo en una cadena corta de longitud fija (huella). **MD5** produce 128 bits (32 caracteres hexadecimales). Dos archivos idénticos byte a byte tienen el mismo MD5, y un solo byte distinto lo cambia por completo. Detecta **copias exactas**, no la misma foto recomprimida, recortada o redimensionada.

### 2.3 Casi duplicados: dHash

El **dHash** (*difference hash*) es una huella que depende de cómo se ve la imagen:

1. Se pasa a escala de grises y se reduce a $9 \times 8$ píxeles (9 columnas, 8 filas).
2. Para cada fila $i$ y cada par de vecinos horizontales, el bit es $b_{i,j} = 1$ si $p_{i,j} > p_{i,j+1}$ y $0$ en caso contrario.
3. Resultan $8 \times 8 = 64$ bits.

Dos huellas se comparan con la **distancia de Hamming**, el número de bits en que difieren:

$$d_H(h_1, h_2) = \sum_{k=1}^{64} \mathbb{1}\big[\,b_k^{(1)} \neq b_k^{(2)}\,\big]$$

Como una misma foto puede aparecer espejada, cada imagen se compara también contra la versión volteada de la otra, y se toma la menor distancia:

$$d(I, J) = \min\big(d_H(h(I), h(J)),\; d_H(h(I), h(\mathrm{flip}(J)))\big)$$

Como los fondos son casi uniformes, dos peces **distintos** pueden dar huellas parecidas (falso positivo). Por eso toda decisión basada en dHash se validó con revisión visual.

### 2.4 Grupos: componentes conexas

Se construye un grafo no dirigido: cada **nodo** es una imagen y hay una **arista** entre dos imágenes si su distancia es $d \le \tau$ (o tienen el mismo MD5). Un **grupo** es una **componente conexa** de ese grafo, es decir, un conjunto de imágenes alcanzables entre sí por cadenas de aristas. Todas las imágenes de un grupo son "la misma foto" en alguna versión y deben ir juntas a una misma partición.

### 2.5 Partición estratificada por grupos

`StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)` reparte **grupos completos** en 20 pliegues (*folds*) intentando que la proporción de clases sea parecida en cada uno. Asignación: 3 pliegues a test ($3/20 = 0.15$), 3 a validación (0.15) y 14 a entrenamiento (0.70).

### 2.6 Cachés

Leer y redimensionar miles de JPG en cada época sería lento. Se preprocesa una vez y se guarda en `.npz` por partición.

- **CNN:** $224 \times 224$ RGB `uint8`, con `INTER_AREA` (promedia píxeles al reducir). Se guarda sin normalizar, para que sirva a cualquier red.
- **U-Net:** imágenes a $256 \times 256$ y máscaras con `INTER_NEAREST`, que no inventa valores intermedios (la máscara solo vale 0 o 255).

### 2.7 Máscaras con visión clásica

- **Máscara de segmentación:** imagen del mismo tamaño que la original donde cada píxel indica pez (255) o fondo (0).
- **Pseudo-máscara:** máscara generada automáticamente, no dibujada a mano. No es la verdad del terreno.
- **GrabCut:** algoritmo de OpenCV. Se le da un rectángulo: lo de afuera es fondo seguro. Modela el color del fondo y del primer plano con mezclas de gaussianas (*GMM*) y decide la frontera con un **corte de grafo** que penaliza que píxeles vecinos parecidos reciban etiquetas distintas. Tolera fondos con textura mejor que un umbral de color.
- **Otsu (método descartado):** elige el umbral $t^*$ que maximiza la varianza entre las dos clases:

$$t^* = \arg\max_t \; \omega_0(t)\,\omega_1(t)\,\big[\mu_0(t) - \mu_1(t)\big]^2$$

  donde $\omega_k$ es la proporción de píxeles de cada clase y $\mu_k$ su media.
- **Morfología:** el *cierre* (dilatar y luego erosionar) rellena huecos; la *apertura* (erosionar y luego dilatar) elimina manchas aisladas.
- **Componente conexa mayor:** se conserva solo la región de mayor área, que se asume es el pez, y se rellenan sus huecos con el contorno externo.
- **Control de calidad (QC):** reglas sobre la máscara ya generada (área y ocupación de una banda interior). Marca las sospechosas; no las corrige.

## 3. Decisiones y datos

| Fecha | Decisión | Motivo y evidencia |
|---|---|---|
| 2026-11-06 | Partición por **grupos**, no por imagen | 610 copias exactas (≈ 6.7 %) por MD5 y 4134 pares casi duplicados con $d_H \le 5$ |
| 2026-11-07 | Umbral de agrupación $\tau = 5$ | Con 2 o 3 se escapan entre 600 y 950 grupos que son la misma foto. Sin grupos gigantes (máx. 14). Costo: excluir 5 grupos multiclase más |
| 2026-11-07 | Excluir grupos con más de una clase y el bloque `rui_791` a `rui_810` | Son copias de imágenes `mrigel`: ruido de etiqueta. 75 + 64 − 39 (solape) = 100 excluidas |
| 2026-11-07 | `StratifiedGroupKFold` de 20 pliegues (3 test, 3 val, 14 train), semilla 42 | Respeta grupos y aproxima 70/15/15 |
| 2026-11-09 | No cambiar la semilla para acomodar `kalibaush` (0.647 / 0.164 / 0.189) | Rompería el contrato de reproducibilidad. Se compensará con pesos de clase en el 02 |
| 2026-11-10 | **Máscaras con GrabCut** en lugar de distancia al color de fondo con Otsu | Otsu falló en 3 de 8 imágenes de prueba (`boal_094`, `rui_1551`, `katla_060`) con fondos con textura (lonas manchadas, platos) |
| 2026-11-10 | Parámetros: 196 px, margen 0.05, 8 iteraciones, apertura 5 | Con 20 iteraciones el resultado fue prácticamente el mismo, pero con ~0.87 s por imagen contra ~1.7 s |
| 2026-11-10 | Generación en **hilos** (`ThreadPoolExecutor`), no procesos | `ProcessPoolExecutor` se colgó (bloqueo de OpenCV con `fork`). Con 2 hilos, ≈ 1.9 imágenes por segundo |
| 2026-11-10 | QC con tres reglas: área mínima, área máxima y ocupación de la banda interior (≤ 0.30) | Con GrabCut el pez nunca toca los 10 px del borde, así que se mide una banda de 5 px dentro del rectángulo |
| 2026-11-10 | `area_min = 0.02` | De 0.011 a 0.020 son fragmentos de aleta; de 0.021 a 0.027 son peces pequeños bien segmentados (1 de 8 era solo una cola) |
| 2026-11-10 | **Plan B:** las máscaras descartadas se excluyen de la U-Net (entrenamiento y evaluación), no de la CNN. El filtro se aplica al cargar en el 03 con `mask_qc.csv` | Tres variantes de reinicio (`rect_12`, `prior`, `centro`) no recuperaron las vacías de forma fiable. No se rehacen cachés |
| 2026-11-11 | **`area_max = 0.35`** (antes provisional en 0.70) | El histograma de `area_frac` tiene una campana con moda cerca de 0.13 que se agota hacia 0.30 y una cola casi plana hasta 0.65. En 0.45 a 0.70 el contorno encierra el plato, el balde o la mesa con el pez dentro (sección 4.4): entrenar con eso enseña a segmentar el soporte. Se eligió 0.35 y no 0.30 para conservar más imágenes en el tramo intermedio |
| 2026-11-11 | Cuaderno 01 **idempotente** (ejecutable de arriba abajo) | Cada paso largo carga su resultado de Drive si existe y es consistente (`REHACER` fuerza un paso). Escritura atómica (archivo temporal y renombrado) para que una desconexión no deje archivos a medias. `splits.csv` nunca se sobrescribe: se recalcula, se compara y se avisa si difiere |
| 2026-11-11 | Verificación de cachés recalculando 3 imágenes desde el JPG | Una versión intermedia de la celda de la caché de la CNN sumaba 99 a los píxeles (desbordamiento en `uint8`). La verificación detecta una caché así y la rehace. Confirmar en la corrida real si hubo que rehacer alguna |
| 2026-11-11 | Semilla fija (`cv2.setRNGSeed`) dentro de `generar_mascara` | La inicialización k-means de GrabCut es aleatoria. Solo afecta a máscaras nuevas: `masks.zip` sigue siendo la referencia |

## 4. Resultados

### 4.1 Partición

- Imágenes: train 6246, val 1359, test 1370 (suman 8975). Proporciones 0.696 / 0.151 / 0.153.
- Máximo de particiones por grupo: 1 (no hay fuga).
- MD5 de `splits.csv`: `5b174a510317656566c44916ff8da4a7`.

| Clase | Excluidas | train | val | test |
|---|---|---|---|---|
| boal | 1 | 621 | 127 | 136 |
| ilish | 0 | 549 | 127 | 113 |
| kalibaush | 6 | 336 | 85 | 98 |
| katla | 10 | 791 | 163 | 169 |
| koi | 4 | 411 | 84 | 93 |
| mrigel | 37 | 859 | 209 | 188 |
| pabda | 2 | 610 | 135 | 115 |
| rui | 34 | 1143 | 239 | 254 |
| telapia | 6 | 926 | 190 | 204 |

### 4.2 Cachés

- CNN: `cnn_224_{train,val,test}.npz` con formas $(6246, 224, 224, 3)$, $(1359, \ldots)$ y $(1370, \ldots)$, `uint8`.
- U-Net: `unet_256_{train,val,test}.npz` con las mismas cantidades a $256 \times 256$; `M` solo contiene $\{0, 255\}$. Fracción media de pez: 0.181, 0.186 y 0.180. Revisión visual de 8 imágenes de val: alineación correcta.
- La caché de la U-Net contiene **todas** las máscaras, también las descartadas por el QC.

### 4.3 Control de calidad de las máscaras

Primera clasificación, con `area_max = 0.70` (antes del ajuste de la sección 4.3.1): máscaras generadas 8975, válidas **8237**, descartadas **738**.

| Motivo | Imágenes |
|---|---|
| Vacía | 640 |
| Área chica ($< 0.02$) | 83 |
| Llena la banda interior | 11 |
| Área grande y llena la banda | 2 |
| Área grande | 2 |

Las vacías no son aleatorias: `rui` 183, `katla` 153, `telapia` 111, `boal` 93, `ilish` 56, `kalibaush` 27, `pabda` 8, `koi` 7, `mrigel` 2. Al regenerar 30 de ellas una por una (sin hilos), 26 siguieron vacías: es una **falla del método**, no de la ejecución. Las imágenes tienen fondo de piedra o icopor manchado y un pez gris o blanco de color parecido.

Intentos de rescate (solo sobre las vacías), todos descartados:

| Variante | Resultado |
|---|---|
| `rect_12` (margen 0.12) | Recupera 6 de 26 por área, pero visualmente casi ninguna era correcta |
| `prior` (candidatos por distancia al fondo) | 8 de 26 por área; contornea el plato o solo la cola |
| `centro` (núcleo central como pez seguro) | ~5 de 12 bien, 1 dudosa y 6 malas |

Válidas para la U-Net por clase y partición con `area_max = 0.70` (referencia, ya no vigente):

| Clase | train | val | test |
|---|---|---|---|
| boal | 543 | 112 | 124 |
| ilish | 506 | 122 | 103 |
| kalibaush | 304 | 84 | 94 |
| katla | 662 | 141 | 150 |
| koi | 404 | 84 | 92 |
| mrigel | 858 | 208 | 186 |
| pabda | 595 | 131 | 111 |
| rui | 1006 | 209 | 221 |
| telapia | 828 | 169 | 190 |

Retención por partición con `area_max = 0.70`: 91.4 % en train, 92.7 % en val y 92.8 % en test.

#### 4.3.1 Con los umbrales finales ($0.02 \le$ área $\le 0.35$, banda $\le 0.30$)

Máscaras generadas: 8975. Válidas: **7118**. Descartadas: **1857** (vacías 640, área chica 83 y 1134 por área grande o banda llena; el desglose exacto está en `reports/tables/01_qc_motivos.csv`). Con `area_max = 0.70` se descartaban 738, así que bajar el umbral a 0.35 descarta 1119 más.

Retención por partición: 79.0 % en train (4933 de 6246), 78.4 % en val (1066 de 1359) y 81.7 % en test (1119 de 1370).

| Clase | train | val | test | Total válidas | Retención |
|---|---|---|---|---|---|
| boal | 388 | 81 | 91 | 560 de 884 | 63.3 % |
| ilish | 332 | 82 | 80 | 494 de 789 | 62.6 % |
| kalibaush | 283 | 76 | 87 | 446 de 519 | 85.9 % |
| katla | 579 | 113 | 137 | 829 de 1123 | 73.8 % |
| koi | 389 | 83 | 91 | 563 de 588 | 95.7 % |
| mrigel | 698 | 165 | 135 | 998 de 1256 | 79.5 % |
| pabda | 582 | 127 | 110 | 819 de 860 | 95.2 % |
| rui | 895 | 175 | 202 | 1272 de 1636 | 77.8 % |
| telapia | 787 | 164 | 186 | 1137 de 1320 | 86.1 % |

`boal` e `ilish` son las clases que más pierden (cerca del 37 %), por sus escenas con plato o balde y, en `boal`, 93 vacías. `koi` y `pabda` casi no pierden.

Las máscaras con área entre 0.35 y 0.70 no entran al entrenamiento ni a las métricas de la U-Net; quedan disponibles para la evaluación cualitativa de la demo.

### 4.4 Desborde de las máscaras aprobadas (revisión visual de 12 por rango)

| Área de la máscara | Aprobadas | Observación visual (conteo a ojo, 12 por rango) |
|---|---|---|
| 0.15 a 0.30 | 2918 | ~10 de 12 siguen la silueta del pez |
| 0.30 a 0.45 | 823 | ~2 o 3 de 12; el resto cubre plato, balde o mesa con el pez dentro |
| 0.45 a 0.70 | 617 | ~1 de 12; casi todas son el plato completo |
| 0.02 a 0.15 | 3879 | Sin revisar en bloque (solo una muestra de 0.04 a 0.06, correcta) |

### 4.5 Pendiente de cerrar

Cerrado el 2026-11-11: `area_max = 0.35` aplicado (se recalcularon `mask_qc.csv` y `mask_params.json` sin regenerar máscaras), cuaderno reorganizado y tablas y figuras guardadas en `reports/`.

Pendiente:

- Revisar una muestra del rango $0.02$ a $0.15$ (3879 imágenes); por ahora solo hay una muestra de 0.04 a 0.06.
- Aplicar el filtro `qc_ok` al cargar las cachés en el cuaderno 03.

## 5. Limitaciones

1. **Pseudo-máscaras.** El IoU y el Dice de la U-Net medirán el acuerdo con máscaras de GrabCut, no con la verdad del terreno. Un falso positivo puede ser error de la U-Net o de la pseudo-máscara.
2. **Sesgo de selección.** Las métricas se calcularán solo sobre imágenes con máscara válida, que son las más fáciles. Con `area_max = 0.35` quedan fuera de las métricas las escenas con plato, balde o mesa grandes; el modelo no se evalúa en ellas.
3. **Retención desigual por clase.** Con `area_max = 0.70`, `katla` retenía 84.9 %, `rui` 87.8 % y `boal` 88.1 %, frente a `mrigel` 99.7 % y `koi` 98.6 %. Con `area_max = 0.35` la brecha crece: `boal` 63.3 % e `ilish` 62.6 %, frente a `koi` 95.7 % y `pabda` 95.2 % (tabla de 4.3.1). El modelo verá menos escenas difíciles de esas dos clases, y su IoU por clase será más optimista que el de las demás.
4. **El QC no detecta toda falla.** Detecta máscaras vacías, diminutas o desbordadas, pero no cuerpos incompletos con área plausible (por ejemplo, sin cabeza ni cola) ni desbordes parciales como el borde del plato en `mrigel_068`.
5. **Máscaras con desborde sobre el fondo.** En imágenes con plato o mesa, la máscara contiene al pez pero también el soporte. Tienen alto recall y baja precisión.
6. **dHash y falsos positivos.** En $d_H = 4$ y $5$ conviven falsos positivos (fondos uniformes) y duplicados reales. Los 8 pares `kalibaush`↔`katla` ($d_H \le 2$) no se inspeccionaron visualmente; al cruzar clases quedaron en grupos multiclase excluidos.
7. **Reproducibilidad de las máscaras.** La primera pasada no fijó la semilla interna de GrabCut (la inicialización k-means es aleatoria), por lo que regenerarlas daría resultados ligeramente distintos. El cuaderno ya la fija para máscaras nuevas, pero el `masks.zip` en Drive es el entregable de referencia y no se regenera.
8. **Revisión visual de pocas muestras.** Las proporciones de la sección 4.4 salen de 12 imágenes por rango y son estimaciones.

## 6. Glosario

### Datos y etiquetas

| Término | Definición breve |
|---|---|
| Dataset | Conjunto de datos del proyecto: las imágenes de SylFishBD (9075 en origen, 8975 tras excluir ruido) |
| Clase | Especie de pez a predecir. Hay 9: boal, ilish, kalibaush, katla, koi, mrigel, pabda, rui, telapia |
| Etiqueta (`label`) | La clase como entero de 0 a 8, en el orden `boal`=0, `ilish`=1, …, `telapia`=8 |
| Desbalance | Diferencia en el número de imágenes por clase. Razón máxima/mínima ≈ 3.18 (rui 1670 vs kalibaush 525) |
| Ruido de etiqueta | Imágenes con la clase equivocada o con dos clases para la misma foto |
| Semilla (`SEED = 42`) | Número que fija los procesos aleatorios para que cualquiera obtenga el mismo resultado |
| Reproducibilidad | Que otra persona, con el mismo código, datos y semilla, obtenga los mismos resultados |

### Archivos del módulo (el "contrato")

| Archivo | Qué es |
|---|---|
| `images/` | Imágenes originales, una carpeta por clase (JPG 500×500×3) |
| `inventario.csv` | Tabla de todas las imágenes con sus huellas (`filepath`, `clase`, `label`, `leida`, `alto`, `ancho`, `canales`, MD5, dHash y dHash volteado) |
| `pares_duplicados.csv` | Pares casi idénticos ($d \le 5$): posiciones `i`, `j`, `dist`, rutas y clases |
| `splits.csv` | **Partición oficial**: `filepath`, `clase`, `label`, `group_id`, `split` |
| `excluidas.csv` | Las 100 imágenes descartadas por ruido de etiqueta: `filepath`, `clase`, `group_id` |
| `cnn_224_{split}.npz` | Caché para la CNN: `X`, `y`, `filepath` |
| `unet_256_{split}.npz` | Caché para la U-Net: `X`, `M`, `y`, `filepath` |
| `masks/`, `masks.zip` | Máscaras PNG $\{0, 255\}$ a 500×500, una por imagen |
| `mask_qc.csv` | QC de las máscaras: `filepath`, `area_frac`, `rect_frac`, `motivo`, `qc_ok` |
| `mask_params.json` | Parámetros de generación y umbrales del QC |

### Columnas de `splits.csv`

| Columna | Significado |
|---|---|
| `filepath` | Ruta de la imagen **relativa a `images/`** (por ejemplo `boal/boal_001.jpg`). Es la llave que une todos los archivos |
| `clase` | Nombre de la especie |
| `label` | Entero 0 a 8 equivalente a `clase` |
| `group_id` | Grupo de duplicados al que pertenece; todas las imágenes de un grupo comparten partición |
| `split` | `train`, `val` o `test` |

### Partición y duplicados

| Término | Definición breve |
|---|---|
| Entrenamiento / validación / prueba | 70 % para aprender, 15 % para elegir modelo e hiperparámetros, 15 % reservado para la evaluación final, que se usa **una sola vez** |
| Fuga de datos | Información de prueba que se filtra al entrenamiento e infla la métrica |
| Estratificar | Mantener la proporción de clases en cada partición |
| Pliegue (*fold*) | Cada una de las 20 porciones en que se reparten los grupos |
| Grupo (`group_id`) | Componente conexa de imágenes que son la misma foto en alguna versión |
| Exclusión | Retirar de la partición imágenes con ruido de etiqueta |
| MD5 | Huella de 128 bits de un archivo; detecta copias exactas |
| dHash | Huella perceptual de 64 bits basada en gradientes horizontales |
| Distancia de Hamming | Número de bits distintos entre dos huellas |
| Falso positivo | Par marcado como duplicado que son peces distintos |
| Componente conexa | Conjunto de nodos alcanzables entre sí en un grafo |
| Umbral $\tau$ | Distancia máxima ($d_H \le \tau$) para considerar dos imágenes la misma foto; aquí $\tau = 5$ |

### Imágenes y cachés

| Término | Definición breve |
|---|---|
| Píxel | Unidad mínima de una imagen. En color tiene 3 valores (rojo, verde, azul) |
| RGB / BGR | Orden de los canales. Las redes esperan RGB; OpenCV lee en BGR, por eso se convierte |
| `uint8` | Entero sin signo de 8 bits (0 a 255), formato habitual de una imagen |
| Forma de un arreglo | Dimensiones. Un lote de imágenes es $(N, H, W, C)$: cantidad, alto, ancho, canales |
| Interpolación | Regla para calcular píxeles al redimensionar. `INTER_AREA` promedia (imágenes); `INTER_NEAREST` copia el vecino (máscaras) |
| Caché `.npz` | Arreglos NumPy preprocesados y comprimidos |
| `X`, `y`, `M` | Entradas (imágenes), salidas esperadas (etiquetas) y máscaras |
| Época | Una pasada completa por el conjunto de entrenamiento |

### Cuaderno

| Término | Definición breve |
|---|---|
| Idempotente | Que ejecutarlo varias veces da el mismo resultado: si el producto ya existe y es consistente, se carga en lugar de recalcularse |
| `REHACER` | Diccionario de la configuración para forzar un paso aunque su resultado ya exista en Drive |
| Escritura atómica | Escribir a un archivo temporal y renombrarlo al terminar, para no dejar archivos a medias si la sesión se corta |
| Caché consistente | Caché cuyos `filepath` y etiquetas coinciden con `splits.csv` y cuyas imágenes de muestra coinciden con las recalculadas desde el JPG |

### Máscaras

| Término | Definición breve |
|---|---|
| Segmentación | Clasificar cada píxel, a diferencia de la clasificación, que da una etiqueta por imagen |
| Máscara / pseudo-máscara | Imagen pez = 255, fondo = 0. "Pseudo" porque se genera automáticamente y no es la verdad del terreno |
| GrabCut | Algoritmo de OpenCV que separa objeto y fondo con modelos de color y un corte de grafo, partiendo de un rectángulo |
| GMM | Mezcla de gaussianas: modelo de color que usa GrabCut para fondo y primer plano |
| Espacio Lab | Representación del color que separa luminosidad (L) de color (a, b) |
| Otsu | Método de umbral automático; descartado por fallar con fondos con textura |
| Cierre / apertura | Operaciones morfológicas que rellenan huecos / eliminan manchas |
| QC (control de calidad) | Reglas que marcan máscaras sospechosas: área muy chica o grande, o banda interior ocupada. Umbrales finales: `area_min = 0.02`, `area_max = 0.35`, `rect_max = 0.30` |
| `area_frac` | Fracción de la imagen que ocupa la máscara |
| `rect_frac` | Fracción de una banda de 5 px, justo dentro del rectángulo de GrabCut, que ocupa la máscara |
| Retención | Proporción de imágenes que pasan el QC |
| Recall / precisión | Fracción del pez real que la máscara cubre / fracción de la máscara que es pez real |
