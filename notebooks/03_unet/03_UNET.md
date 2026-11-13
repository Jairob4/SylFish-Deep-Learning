# Módulo 03: U-Net (segmentación pez/fondo)

*Documento del informe y guía de estudio. Un solo MD por módulo: propósito, preconceptos, decisiones, resultados, limitaciones y glosario. Última actualización: jueves 12 de noviembre de 2026.*

## 1. Propósito

Entrenar y evaluar una red que, dada una imagen, produzca una **máscara a nivel de píxel** que separe el pez del fondo. La clase del pez la aporta la CNN (módulo 02); las dos redes son independientes y solo se unen en la demo (módulo 04).

Entradas: `unet_256_{train,val,test}.npz` y `mask_qc.csv` del módulo 01, filtrados con `qc_ok`. Quedan 4933 imágenes en train, 1066 en val y 1119 en test.

Salidas: `models/unet_v1.keras` (mejor modelo según el mIoU de validación), `models/unet_v1_umbral.json` (umbral $\tau$ fijado en val), y las tablas y figuras de `reports/`.

Contrato con el módulo 04: la demo carga `unet_v1.keras` y el $\tau$ del JSON, redimensiona la imagen a $256 \times 256$, divide entre 255 y binariza la salida con $\tau$.

## 2. Preconceptos

### 2.1 Segmentación y U-Net

La **segmentación semántica** asigna una clase a cada píxel, a diferencia de la clasificación, que da una etiqueta por imagen. Aquí hay dos clases (pez y fondo), por lo que la salida es un mapa $(256, 256, 1)$ de probabilidades.

La **U-Net** es una red encoder-decoder con forma de U:

- **Encoder:** repite bloques de convolución y reduce la resolución con `MaxPool`. Gana contexto (qué hay en la imagen) y pierde precisión espacial.
- **Cuello de botella:** el nivel más profundo, a la menor resolución.
- **Decoder:** recupera la resolución con convoluciones transpuestas.
- **Skip connections:** al subir, se **concatena** la salida del encoder en la misma resolución. Devuelve al decoder el detalle fino (bordes) que el encoder había perdido.

Con entrada de $256$ px y 4 reducciones por $2$, el cuello de botella trabaja a $256 / 2^4 = 16$ px.

### 2.2 Bloque base

Cada nivel aplica dos veces: `Conv2D` $3 \times 3$ con `padding='same'` (conserva la resolución), `BatchNormalization` y `ReLU`. La **ReLU** es $\max(0, x)$.

### 2.3 Normalización por lotes (*batch normalization*)

Para los valores $x_1, \dots, x_m$ de un canal en el lote (en una convolución, $m$ cuenta lote, alto y ancho):

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i, \qquad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \qquad y_i = \gamma\,\hat{x}_i + \beta$$

$\gamma$ y $\beta$ son parámetros entrenables por canal. En entrenamiento usa las estadísticas del lote; en inferencia usa promedios móviles acumulados. Estabiliza el entrenamiento y permite tasas de aprendizaje mayores. **No es lo mismo** que escalar la entrada a $[0, 1]$: el escalado actúa una sola vez sobre la imagen; la BN actúa sobre las activaciones intermedias en cada capa.

### 2.3.1 Convolución transpuesta

`Conv2DTranspose(f, 2, strides=2)` duplica la resolución con un filtro **aprendido** y deja $f$ canales. Es distinta de `UpSampling2D`, que repite o interpola píxeles sin parámetros.

### 2.4 Salida y umbral

La última capa es una convolución $1 \times 1$ con 1 filtro y activación sigmoide:

$$\hat{y}_p = \sigma(z_p) = \frac{1}{1 + e^{-z_p}} \in (0, 1)$$

Un píxel $p$ se clasifica como pez si $\hat{y}_p > \tau$. El umbral $\tau$ se elige en validación.

### 2.5 Pérdida: BCE + Dice

El pez ocupa cerca del $15\,\%$ de los píxeles (desbalance de $\approx 1$ a $5.7$). Se combinan dos términos. Con $p_i$ la probabilidad predicha, $y_i \in \{0, 1\}$ la máscara y $N$ los píxeles de una imagen:

$$\text{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\big]$$

$$\text{Dice} = \frac{2\sum_i p_i\, y_i + \varepsilon}{\sum_i p_i + \sum_i y_i + \varepsilon}, \qquad \mathcal{L} = \text{BCE} + \left(1 - \text{Dice}\right)$$

La BCE penaliza cada píxel; el Dice mide solapamiento y no se deja engañar por el desbalance (predecir todo fondo daría $\approx 85\,\%$ de exactitud por píxel, pero Dice $\approx 0$).

### 2.6 Métricas

**IoU** (intersección sobre unión) de una imagen, con $P$ la predicción binarizada e $Y$ la máscara:

$$\text{IoU} = \frac{|P \cap Y|}{|P \cup Y|}$$

- **mIoU por imagen:** el promedio del IoU de cada imagen. Da el mismo peso a cada imagen. Es la métrica principal.
- **Dice global (píxeles):** se calcula sobre todos los píxeles del conjunto a la vez, así que pesan más las imágenes con peces grandes. **No es comparable** con el mIoU por imagen.
- **Matriz de confusión de píxeles:** conteos de verdaderos y falsos positivos y negativos sobre todos los píxeles.
- La **exactitud** por píxel no se usa: con $\approx 85\,\%$ de fondo es casi siempre alta.

### 2.7 Aumento de datos

Transformaciones aleatorias aplicadas **solo en entrenamiento** para que la red vea más variedad. Las geométricas (volteos, rotaciones de $90^\circ$) se aplican a imagen y máscara **a la vez**: se apilan en un tensor de 4 canales y reciben la misma transformación. Los ángulos múltiplos de $90^\circ$ no interpolan, así que la máscara sigue siendo exactamente $\{0, 1\}$. Brillo y contraste se aplican solo a la imagen.

### 2.8 Entrenamiento

- **Lote (*batch*):** número de imágenes por paso (aquí 16). Una época son $\lceil 4933 / 16 \rceil = 309$ pasos.
- **Adam:** optimizador adaptativo; **tasa de aprendizaje** inicial $10^{-3}$.
- **`ReduceLROnPlateau`:** reduce la tasa a la mitad si la métrica vigilada no mejora durante unas épocas.
- **`EarlyStopping`:** detiene el entrenamiento si la métrica no mejora durante `patience` épocas; con `restore_best_weights=True` devuelve los pesos de la mejor época. Con `verbose=0` (por defecto) corta **sin mensaje**.
- **`ModelCheckpoint`:** guarda en Drive el mejor modelo según la métrica de validación.
- **`tf.data`:** canalización que carga, barajea, aumenta y agrupa los datos en lotes.

### 2.9 Pseudo-máscaras

Las máscaras de entrenamiento no son verdad del terreno: las generó GrabCut (módulo 01). El IoU mide **acuerdo con GrabCut**, no con el pez real.

## 3. Decisiones y datos

| Fecha | Decisión | Motivo y evidencia |
|---|---|---|
| 2026-11-05 | U-Net binaria pez/fondo, sigmoide de 1 canal | Las máscaras de OpenCV solo separan pez de fondo; la clase la da la CNN |
| 2026-11-06 | YOLO y cajas delimitadoras descartados | Necesitan cajas (no existen en SylFishBD, habría que derivarlas de las máscaras, con el mismo error en las descartadas), cambian la tarea a detección y no entregan máscara. La caja puede sacarse de la máscara predicha con `cv2.boundingRect` |
| 2026-11-11 | Refinar máscaras descartadas dentro de sí mismas (color del soporte en Lab y Otsu) **descartado** | En 12 máscaras `area_grande`: ~5 bien, 2 parciales y 5 mal. Falla con el óxido y los reflejos. "Área en rango" no sirvió como criterio (11 de 12 pasaban y 5 estaban mal) |
| 2026-11-12 | Filtro `qc_ok` al cargar las cachés | Quedan 4933 / 1066 / 1119. Fracción media de pez 0.154 / 0.150 / 0.153: las tres particiones son comparables |
| 2026-11-12 | Encoder de 4 niveles con 16, 32, 64 y 128 filtros; cuello de botella de 256 a $16 \times 16$ | $256 \to 16$ requiere 4 reducciones. Con 3, el cuello de botella quedaría a $32 \times 32$, con menos contexto |
| 2026-11-12 | API funcional de Keras, no `Sequential` | Las skip connections necesitan ramificarse y `Concatenate` recibe dos tensores |
| 2026-11-12 | Bloque `Conv2D` → `BatchNormalization` → `ReLU` (ReLU como capa aparte), dos veces por nivel | La BN debe ir entre la convolución y la activación |
| 2026-11-12 | Decoder con `Conv2DTranspose` $2 \times 2$, `strides=2` | Como la U-Net original; la transpuesta deja los canales del nivel |
| 2026-11-12 | Entrada escalada a $[0, 1]$ en `tf.data`; la BN va además dentro de la red | Son dos normalizaciones distintas |
| 2026-11-12 | Aumento: volteo horizontal y vertical y rotaciones de $90^\circ$ (imagen y máscara apiladas); brillo $\pm 0.15$ y contraste $0.8$ a $1.2$ (solo imagen); **solo en train** | Las fotos vienen en ángulos arbitrarios. Sin ángulos libres ni zoom: obligarían a interpolar y rebinarizar la máscara |
| 2026-11-12 | Permutación única de train con semilla 42 antes de `tf.data`, más `shuffle(2048)` por época | Las cachés salen ordenadas por clase y un buffer de 2048 sobre 4933 imágenes no mezcla del todo. Se aplica **después** de la partición |
| 2026-11-12 | Pérdida BCE + Dice; métrica principal mIoU por imagen | Desbalance $\approx 1$ a $5.7$ |
| 2026-11-12 | Adam $10^{-3}$, lote 16, máximo 40 épocas | Compromiso de memoria en una T4 |
| 2026-11-12 | `ReduceLROnPlateau` (factor 0.5, paciencia 3, mínimo $10^{-5}$), `EarlyStopping` (paciencia 8, `restore_best_weights`) y `ModelCheckpoint` sobre `val_iou_por_imagen` | La validación fue inestable al inicio (cayó a 0.65 con tasa $5 \times 10^{-4}$); al bajar la tasa se calmó |
| 2026-11-12 | $\tau = 0.6$, elegido en val | mIoU de $0.8334$ frente a $0.833$ con $0.5$: diferencia despreciable y curva casi plana |
| 2026-11-12 | Test evaluado **una sola vez** con el modelo congelado y $\tau$ fijado | Regla de `CONTEXTO.md`: se elige con val, se reporta con test |
| 2026-11-12 | Autoentrenamiento (usar la U-Net para etiquetar las descartadas y reentrenar) **no se hace en esta entrega** | Circularidad (las descartadas son las escenas más difíciles) y 1.5 a 2 horas de costo sin una medida de beneficio. Si se hace, debe compararse en val **antes** de correr test |

## 4. Resultados

### 4.1 Arquitectura

- Entrada $(256, 256, 3)$; salida $(256, 256, 1)$.
- Parámetros: **1,946,993** en total, 1,944,049 entrenables y 2,944 no entrenables (los promedios móviles de la BN). Coincide con el cálculo a mano.

### 4.2 Entrenamiento

- **37 épocas** ejecutadas; el entrenamiento se detuvo por `EarlyStopping`. La mejor época es la **29**, con `val_iou_por_imagen` $= 0.8332$, y las 8 siguientes no la superaron.
- El `evaluate` sobre val con los pesos restaurados reprodujo la cifra: pérdida $0.2213$ y IoU $0.8332$.
- Tasa de aprendizaje: $10^{-3}$ hasta la época 5, luego $5 \times 10^{-4}$, $2.5 \times 10^{-4}$, $1.25 \times 10^{-4}$, ..., hasta $10^{-5}$ en la época 35.
- IoU de train al final $\approx 0.847$ frente a $\approx 0.833$ en val: brecha de $\approx 0.014$, sin sobreajuste marcado. El IoU de train se mide con aumento y como promedio móvil durante la época.
- La validación fue inestable al inicio (épocas 3 a 5, 9 y 12, con caídas a $0.65$ y $0.77$) y se estabilizó al bajar la tasa. La causa es una hipótesis (desajuste entre las estadísticas de la BN en entrenamiento e inferencia con pasos grandes) y no se comprobó.
- Tiempo: $\approx 48$ s por época en una T4; la primera tardó 162 s por la compilación.

### 4.3 Umbral

| $\tau$ | mIoU de val |
|---|---|
| 0.1 | $\approx 0.810$ |
| 0.5 | 0.833 |
| **0.6** | **0.8334** |
| 0.9 | $\approx 0.814$ |

La curva es casi plana entre $0.3$ y $0.8$: el modelo es poco sensible al umbral.

### 4.4 Validación ($\tau = 0.6$)

| Medida | Valor |
|---|---|
| mIoU por imagen | 0.8334 |
| Mediana | 0.915 |
| Cuartiles (25 / 75) | 0.796 / 0.951 |
| Percentiles 5 / 95 | 0.411 / 0.970 |
| IoU $< 0.5$ | 93 de 1066 (8.7 %) |
| IoU $< 0.7$ | 176 de 1066 (16.5 %) |

Matriz de confusión de píxeles (millones de píxeles): fondo→fondo 58.45, fondo→pez 0.92, pez→fondo 1.35, pez→pez 9.14. Por fila: 98.5 % y 1.5 % del fondo; 12.9 % y 87.1 % del pez. Precisión $\approx 0.909$, recall $\approx 0.871$, Dice global $\approx 0.890$, calculados a partir de la matriz.

IoU por clase en val:

| Clase | Imágenes | Media | Mediana | Mínimo |
|---|---|---|---|---|
| boal | 81 | 0.739 | 0.840 | 0.145 |
| ilish | 82 | 0.867 | 0.928 | 0.381 |
| kalibaush | 76 | 0.863 | 0.950 | 0.000 |
| katla | 113 | 0.819 | 0.912 | 0.000 |
| koi | 83 | 0.898 | 0.939 | 0.076 |
| mrigel | 165 | 0.814 | 0.869 | 0.207 |
| pabda | 127 | 0.855 | 0.910 | 0.000 |
| rui | 175 | 0.819 | 0.900 | 0.000 |
| telapia | 164 | 0.846 | 0.951 | 0.000 |

### 4.5 Test (una sola vez)

El modelo cargado desde `unet_v1.keras` reprodujo el mIoU de val ($0.8334$) antes de evaluar test.

| Medida | Val | Test |
|---|---|---|
| mIoU por imagen | 0.8334 | **0.8344** |
| Mediana | 0.915 | 0.910 |
| Percentil 5 | 0.411 | 0.436 |
| IoU $< 0.5$ | 93 de 1066 (8.7 %) | 75 de 1119 (6.7 %) |
| IoU $< 0.7$ | 176 de 1066 (16.5 %) | 180 de 1119 (16.1 %) |
| Precisión / recall (píxeles) | $\approx 0.909$ / $\approx 0.871$ | 0.901 / 0.877 |
| Dice global (píxeles) | $\approx 0.890$ | 0.889 |

Para val, precisión, recall y Dice global se calcularon a mano desde la matriz de confusión. La diferencia de una milésima en el mIoU entre val y test es ruido; no hay señal de sobreajuste a val.

IoU por clase en test:

| Clase | Imágenes | Media | Mediana | Mínimo |
|---|---|---|---|---|
| boal | 91 | 0.788 | 0.875 | 0.000 |
| ilish | 80 | 0.859 | 0.911 | 0.012 |
| kalibaush | 87 | 0.895 | 0.957 | 0.000 |
| katla | 137 | 0.798 | 0.900 | 0.004 |
| koi | 91 | 0.852 | 0.913 | 0.534 |
| mrigel | 135 | 0.822 | 0.922 | 0.234 |
| pabda | 110 | 0.831 | 0.878 | 0.007 |
| rui | 202 | 0.819 | 0.889 | 0.039 |
| telapia | 186 | 0.865 | 0.951 | 0.259 |

`boal` es la más baja en las dos particiones, y `katla` está entre las más bajas. El resto cambia de orden (por ejemplo `koi`, la mejor en val con $0.898$, baja a $0.852$ en test), por lo que solo esas dos diferencias son consistentes.

### 4.6 Análisis cualitativo (solo val)

- **Los 8 de menor IoU** (IoU entre 0.00 y 0.08): a ojo, en 7 la red segmenta correctamente el pez y la discrepancia se debe a que la pseudo-máscara marca otra región (suelo, borde del plato o lona). Solo en `koi_300` la red también falla (marca solo manchas pequeñas del pez borroso).
- **Los 8 de mayor IoU** (0.98) son todos `koi` sobre fondo claro y uniforme, con diferencias de uno o dos píxeles en el borde. Varios parecen versiones del mismo pez. Son el caso fácil.
- **Muestra aleatoria de 8:** 7 tienen IoU entre 0.87 y 0.97 con diferencias finas en el borde. El octavo, `telapia_690` (0.46), tiene una región grande que la etiqueta marca como pez y no lo es (a ojo).
- Las máscaras, tanto las de la etiqueta como las de la red, son contornos suaves que no recogen aletas finas: la red reproduce el suavizado de GrabCut.

**Conclusión.** La U-Net segmenta el pez con un acuerdo alto con las pseudo-máscaras en la mayoría de las imágenes de val: la mediana del IoU por imagen es $0.915$ y, en una muestra aleatoria de 8, 7 tienen IoU $\geq 0.87$. El mIoU ($\approx 0.83$) queda por debajo de la mediana por una cola de imágenes de bajo IoU (93 de 1066 con IoU $< 0.5$). En los casos de menor IoU, a ojo, el error proviene principalmente de las pseudo-máscaras y no de la red, por lo que el mIoU sería una cota inferior de la calidad real. Es una observación cualitativa: se basa en muestras pequeñas, no se contrastó con una referencia anotada a mano y no cuantifica qué proporción de la cola corresponde a etiqueta errónea y cuál a error de la red.

### 4.7 Archivos generados

- `models/unet_v1.keras`, `models/unet_v1_umbral.json`.
- `reports/tables/`: `03_unet_historial.csv`, `03_unet_confusion_pixeles.csv`, `03_unet_iou_por_clase.csv`, `03_unet_test_iou_por_clase.csv`, `03_unet_test_resumen.json`.
- `reports/figures/`: `03_unet_umbral.png`, `03_unet_confusion_pixeles.png`, `03_unet_peores_val.png`, `03_unet_mejores_val.png`, `03_unet_azar_val.png`, `03_unet_hist_iou_val.png`, `03_unet_curvas_entrenamiento.png`.

### 4.8 Pendiente

- Análisis de sensibilidad de la matriz de confusión: Dice global con y sin las imágenes de IoU $< 0.1$, reportando ambas cifras.
- Revisión manual de unas 20 a 30 imágenes de val con IoU $< 0.1$ (etiqueta mala frente a falla de la red) y, con tiempo, anotación a mano de una muestra para calcular el IoU de la red y el de GrabCut contra la misma referencia.
- Módulo 04 (demo): cargar el modelo y el umbral, mostrar el área predicha y advertir si cae fuera de $[0.02, 0.35]$.

## 5. Limitaciones

1. **Pseudo-máscaras.** El IoU y el Dice miden acuerdo con GrabCut, no con la verdad del terreno. Un falso positivo o negativo puede ser error de la red o de la etiqueta, y la matriz de confusión no los distingue.
2. **Sesgo de selección.** Las métricas son solo sobre imágenes con `qc_ok`. El $\approx 21\,\%$ con plato, balde o mesa grandes no entró al entrenamiento ni a las métricas; la red no se evaluó en ellas, y es probable que rinda peor.
3. **Errores de etiqueta que pasan el QC.** El control de calidad detecta vacías, diminutas o desbordadas, pero no máscaras de área plausible en el sitio equivocado (`mrigel_873`, `katla_197`, y los 8 peores casos de val).
4. **Contorno suavizado.** La red reproduce el suavizado de GrabCut (sin aletas finas, a ojo), por lo que el detalle del borde queda limitado por la etiqueta.
5. **Retención desigual por clase.** `boal` e `ilish` perdieron cerca del 37 % en el QC (módulo 01); su IoU por clase es más optimista que el de las demás. Con 80 a 200 imágenes por clase, una sola imagen fallida mueve la media alrededor de $0.01$, y el orden de las clases cambia entre val y test (solo `boal` y `katla` se mantienen bajas).
6. **Imágenes no independientes.** Dentro de una misma partición hay casi duplicados (mismo grupo), por lo que las imágenes no son independientes y no se da un intervalo de confianza.
7. **Métricas no comparables.** El Dice global pesa más las imágenes con peces grandes y sale más alto ($\approx 0.89$) que el mIoU por imagen ($\approx 0.83$).
8. **Una sola corrida.** No se midió la variabilidad entre semillas.
9. **Revisión visual de pocas muestras.** Las proporciones de la sección 4.6 salen de 8 imágenes por figura, contadas a ojo.
10. **Umbral.** La ganancia de $\tau = 0.6$ sobre $0.5$ es de unas cuatro diezmilésimas; no es una mejora real.

## 6. Glosario

### Red

| Término | Definición breve |
|---|---|
| U-Net | Red encoder-decoder con skip connections para segmentación |
| Encoder / decoder | Parte que reduce la resolución y extrae contexto / parte que la recupera |
| Cuello de botella (*bottleneck*) | Nivel más profundo, a $16 \times 16$ aquí |
| Skip connection | Concatenación de la salida del encoder con el decoder en la misma resolución |
| `Conv2D` | Convolución; `padding='same'` conserva la resolución |
| `MaxPooling2D` | Reduce a la mitad la resolución quedándose con el máximo de cada ventana $2 \times 2$ |
| `Conv2DTranspose` | Duplica la resolución con un filtro aprendido |
| `UpSampling2D` | Duplica la resolución repitiendo píxeles, sin parámetros |
| `BatchNormalization` | Normaliza las activaciones por canal con las estadísticas del lote, con $\gamma$ y $\beta$ aprendidos |
| ReLU | $\max(0, x)$ |
| Sigmoide | $\sigma(z) = 1 / (1 + e^{-z})$; lleva la salida a $(0, 1)$ |
| API funcional | Forma de Keras de construir modelos con ramificaciones: cada capa se aplica sobre un tensor |

### Entrenamiento

| Término | Definición breve |
|---|---|
| Lote (*batch*) | Imágenes procesadas en un paso (16) |
| Época | Una pasada completa por train (309 pasos) |
| Adam | Optimizador adaptativo |
| Tasa de aprendizaje | Tamaño del paso de actualización de los pesos |
| `ReduceLROnPlateau` | Reduce la tasa a la mitad si la métrica se estanca |
| `EarlyStopping` | Detiene el entrenamiento si la métrica no mejora en `patience` épocas |
| `ModelCheckpoint` | Guarda el mejor modelo |
| Sobreajuste | El modelo mejora en train pero no en validación |
| Aumento de datos | Transformaciones aleatorias solo en train |
| `tf.data` | Canalización de carga, barajado, aumento y lotes |

### Métricas

| Término | Definición breve |
|---|---|
| BCE | Entropía cruzada binaria, por píxel |
| Dice | Solapamiento: $2\lvert P \cap Y \rvert / (\lvert P \rvert + \lvert Y \rvert)$ |
| IoU | Intersección sobre unión: $\lvert P \cap Y \rvert / \lvert P \cup Y \rvert$ |
| mIoU por imagen | Promedio del IoU de cada imagen; métrica principal |
| Dice global | Dice sobre todos los píxeles a la vez |
| Umbral $\tau$ | Probabilidad desde la cual un píxel es pez (0.6) |
| Matriz de confusión de píxeles | Conteos de TP, FP, FN y TN |
| Precisión / recall | Fracción de lo marcado que es pez / fracción del pez real que se recupera |
| Mapa de error | Imagen donde blanco es acierto, rojo es falso positivo y azul es falso negativo |
| Cota inferior | Valor que la cifra real no debería quedar por debajo de |

### Datos

| Término | Definición breve |
|---|---|
| `qc_ok` | Máscara que pasó el control de calidad del módulo 01 |
| Pseudo-máscara | Máscara generada por GrabCut; no es verdad del terreno |
| Autoentrenamiento | Usar un modelo para etiquetar datos y reentrenar con ellas |
| Pseudo-etiqueta | Etiqueta generada por un modelo |
