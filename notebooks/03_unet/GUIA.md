# Guía 03 — U-Net de segmentación pez/fondo

**Responsable:** Jairo · **Depende de:** cuaderno 01 (`data/splits/splits.csv`, `data/masks/`, `data/masks/mask_qc.csv` congeladas el D3, y caché `data/processed/unet_256_{train,val,test}.npz`) · **Entrega a:** cuaderno 04 (`unet_best.keras`, umbral $t$, decisión de postprocesado, IoU por imagen en test), ML clásico (`unet_pred_test/` para comparar máscaras), informe y presentaciones · **Fecha objetivo:** D3 dom 8 nov (verificar la caché `.npz` U-Net que genera el cuaderno 01) → D4 lun 9 (U-Net v1 entrenada y evaluada en val) → D5 mar 10 nov (U-Net v2, umbral ajustado en val) → D7 jue 12 nov (sección final de test, una sola vez). Congelamiento: mié 11 a las 23:59.

## 1. De qué se trata

La U-Net responde "¿dónde está el pez?": produce, para cada píxel, la probabilidad de que pertenezca al pez. Es segmentación semántica binaria (pez/fondo), con salida de 1 canal sigmoide; la especie la aporta la CNN del cuaderno 02 y ambos modelos solo se juntan en la demo del cuaderno 04 (Diseño A). En la demo, la máscara se superpone a la imagen y da el área segmentada como porcentaje.

SylFishBD no trae anotaciones de segmentación. La U-Net se entrena con las pseudo-máscaras generadas con OpenCV en el cuaderno 01. Limitación que se declara en el informe y en la sustentación: la red aprende a reproducir pseudo-máscaras, no una verdad de terreno experta. Lo que aporta frente al pipeline de OpenCV es un modelo único, entrenable y sin umbrales manuales, que puede generalizar a fondos e iluminaciones donde Otsu falla. Esto se comprueba con imágenes externas en el cuaderno 04 y, si existe, con el conjunto "oro".

El cuaderno es autocontenido, como el de referencia CIFAR-10: configuración, verificación de la caché, aumento, arquitectura, pérdida y métricas, entrenamiento, ajuste del umbral en val, evaluación en val y peores casos viven en el mismo archivo. No hay módulos `.py` compartidos: las funciones auxiliares (bloque doble, pérdida, métricas, IoU por imagen) son celdas del cuaderno. La última sección mide la U-Net en test y se ejecuta una sola vez, el jueves 12, con el modelo congelado.

## 2. Entradas y salidas (contrato)

**Entradas:**

| Tipo | Ruta / objeto | Detalle |
|---|---|---|
| Entrada | `data/splits/splits.csv` | `filepath`, `clase`, `label`, `group_id`, `split` |
| Entrada | `data/masks/<clase>/<clase>_NNN.png` | PNG 1 canal, valores $\{0,255\}$, resolución original |
| Entrada | `data/masks/mask_qc.csv` | se usan solo filas con `estado = ok` |
| Caché (solo se verifica; la genera el cuaderno 01) | `data/processed/unet_256_{train,val}.npz` (Drive, bajo `MyDrive/sylfish/`) | `X`: `uint8` $(N,256,256,3)$ RGB; `M`: `uint8` $(N,256,256,1)$ en $\{0,1\}$; `y` (0–8) y `filepath` alineados con `X` |
| Caché de test (solo en la sección final) | `data/processed/unet_256_test.npz` | misma estructura; no se abre antes del jueves 12 |
| Imágenes originales (solo sección final) | `/content/sylfishbd/images/` (zip descomprimido) | para exportar `unet_pred_test` a resolución original |
| Configuración congelada (solo sección final) | `MyDrive/sylfish/models/config_inferencia.json` (lo escribe el cuaderno 04 el D6, paso 10, antes de las 23:59 del miércoles 11; aquí solo se lee) | se usa para contrastar umbral $t$, postprocesado y MD5 de `unet_best.keras` |
| Clases | `classes.txt` | orden 0–8 |

**Salidas:**

| Tipo | Ruta / objeto | Detalle |
|---|---|---|
| Salida | `MyDrive/sylfish/models/unet_best.keras` | mejor modelo según `val_loss` |
| Salida | umbral elegido en val | por defecto 0.5; valor final en `reports/tables/unet_metricas_val.json` |
| Salida | tablas en `reports/tables/` y figuras en `reports/figures/` | ver sección 8 |
| Salida (comparación con ML) | `MyDrive/sylfish/comparacion/unet_pred_test/<clase>/<clase>_NNN.png` | predicción binarizada a resolución original, PNG $\{0,255\}$, para todas las imágenes de test de `splits.csv` |

Por contrato (v1.1), cada `.npz` trae además `y` (etiqueta 0–8) y `filepath`, alineados con `X`, para desglosar métricas por clase y rastrear los peores casos sin depender del orden de `splits.csv`. Decisión sugerida (confirmar con el equipo): la U-Net también lleva `Rescaling(1/255)` dentro del modelo, como la CNN, y recibe `uint8`.

Las constantes comunes (semilla, tamaño de entrada, rutas de Drive, orden de clases) se escriben en la celda de configuración al inicio, copiadas idénticas de los demás cuadernos.

## 3. Fundamento teórico para la sustentación

**Arquitectura.** Encoder de 4 niveles (32-64-128-256 filtros) y cuello de botella de 512. Cada bloque doble es 2× (`Conv2D` $3\times3$ `padding="same"` → `BatchNormalization` → `ReLU`). Bajada con `MaxPooling2D` $2\times2$. Subida con `Conv2DTranspose` $2\times2$, `strides=2`, concatenación con el skip del mismo nivel y bloque doble. Salida: `Conv2D` $1\times1$ con 1 filtro y sigmoide. Total $\approx7.77\times10^6$ parámetros.

| Nivel | Operación | Salida |
|---|---|---|
| Entrada | `Rescaling(1/255)` | $256\times256\times3$ |
| Enc 1 | bloque doble 32 → pool | $256\times256\times32$ (skip 1) → $128\times128\times32$ |
| Enc 2 | bloque doble 64 → pool | $128\times128\times64$ (skip 2) → $64\times64\times64$ |
| Enc 3 | bloque doble 128 → pool | $64\times64\times128$ (skip 3) → $32\times32\times128$ |
| Enc 4 | bloque doble 256 → pool | $32\times32\times256$ (skip 4) → $16\times16\times256$ |
| Cuello | bloque doble 512 | $16\times16\times512$ |
| Dec 4 | ConvT 256 → concat skip 4 → bloque doble 256 | $32\times32\times256$ → $32\times32\times512$ → $32\times32\times256$ |
| Dec 3 | ConvT 128 → concat skip 3 → bloque doble 128 | $64\times64\times128$ → $64\times64\times256$ → $64\times64\times128$ |
| Dec 2 | ConvT 64 → concat skip 2 → bloque doble 64 | $128\times128\times64$ → $128\times128\times128$ → $128\times128\times64$ |
| Dec 1 | ConvT 32 → concat skip 1 → bloque doble 32 | $256\times256\times32$ → $256\times256\times64$ → $256\times256\times32$ |
| Salida | Conv $1\times1$, sigmoide | $256\times256\times1$ |

**Divisibilidad.** Hay 4 poolings, así que el lado debe ser divisible por $2^4=16$: $256/16=16$. Con un lado no divisible, por ejemplo 250, la bajada da $250\to125\to62$ (por el piso) y la subida $62\to124\neq125$: la concatenación falla por dimensiones distintas.

**Conv2DTranspose.** Con kernel $k$, paso $s$ y relleno $p$:
$$H_{out}=(H_{in}-1)\,s-2p+k$$
Con $k=2$, $s=2$, $p=0$: $H_{out}=2H_{in}$, exactamente el inverso de un pooling $2\times2$. Es un sobremuestreo con pesos aprendidos, a diferencia de `UpSampling2D`, que interpola sin parámetros.

**Skip connections.** Cada pooling gana contexto semántico ("esto es pez") pero pierde localización: en $16\times16$ cada celda resume un parche de $16\times16$ píxeles. El decoder no puede reconstruir bordes finos (aletas, cola) solo desde ese mapa grueso. La concatenación con el mapa del encoder de la misma resolución le devuelve el detalle espacial de alta frecuencia; las convoluciones siguientes combinan "qué" (decoder) con "dónde exactamente" (encoder).

**Salida sigmoide.** $p_i=\sigma(z_i)=\frac{1}{1+e^{-z_i}}\in(0,1)$ por píxel: la probabilidad de que el píxel $i$ sea pez. La máscara binaria se obtiene con un umbral $t$ (0.5 por defecto): $\hat{g}_i=\mathbb{1}[p_i\ge t]$. Se usa $t$ y no $\tau$ porque $\tau$ es el umbral de confianza de la CNN en la demo.

**Pérdida BCE + Dice.**
$$\mathcal{L}_{BCE}=-\frac{1}{n}\sum_{i=1}^{n}\big[g_i\log p_i+(1-g_i)\log(1-p_i)\big]$$
$$\mathcal{L}=\mathcal{L}_{BCE}+\mathcal{L}_{Dice},\qquad \mathcal{L}_{Dice}=1-\frac{2\sum_i p_i g_i+\varepsilon}{\sum_i p_i+\sum_i g_i+\varepsilon}$$
Intuición del Dice: con conjuntos duros, $\mathrm{Dice}=\frac{2TP}{2TP+FP+FN}$; los verdaderos negativos (fondo bien clasificado) no aparecen. Si se reemplazan los conteos por probabilidades ($TP\approx\sum_i p_ig_i$, $|A|\approx\sum_i p_i$, $|B|=\sum_i g_i$) se obtiene una versión diferenciable. El $\varepsilon$ evita la división por cero cuando no hay pez ni predicción.

Por qué ayuda con el desbalance: si el pez ocupa una fracción $a$ de la imagen (ver `area_frac` en `mask_qc.csv`), predecir "todo fondo" da exactitud por píxel $1-a$ y una BCE moderada, porque la mayoría de los términos son fondo fácil. El Dice de esa predicción es 0, porque se mide relativo al tamaño del pez. La BCE aporta gradientes estables por píxel desde el inicio; el Dice optimiza directamente el solapamiento global, que es lo que se reporta. Juntas convergen mejor que cada una sola. Decisión sugerida: calcular $\mathcal{L}_{Dice}$ por imagen y promediar en el lote, coherente con la métrica por imagen.

**IoU y Dice.**
$$\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|},\qquad \mathrm{Dice}(A,B)=\frac{2|A\cap B|}{|A|+|B|}$$
Como $|A|+|B|=|A\cup B|+|A\cap B|$, dividiendo por $|A\cup B|$:
$$\mathrm{Dice}=\frac{2\,\mathrm{IoU}}{1+\mathrm{IoU}}$$
Es creciente, así que por imagen ambas ordenan igual y $\mathrm{Dice}\ge\mathrm{IoU}$ (IoU 0.5 equivale a Dice 0.667). Los promedios sobre el dataset no cumplen esta relación exactamente, porque la función no es lineal; por eso se reportan los dos. Aquí $A$ es la máscara predicha binarizada y $B$ la pseudo-máscara de referencia; se promedia por imagen (cada pez cuenta igual), no por píxel acumulado.

**Por qué test se usa una sola vez.** Cada métrica medida en test es una variable aleatoria: depende de qué imágenes cayeron en test. Si se evalúan varios modelos, umbrales o variantes de postprocesado en test y se reporta el mejor, el número está sesgado hacia arriba, porque el máximo de estimaciones ruidosas sobreestima el máximo de los valores verdaderos:
$$\mathbb{E}\Big[\max_m \widehat{\mathrm{IoU}}_m\Big]\ \ge\ \max_m \mathbb{E}\Big[\widehat{\mathrm{IoU}}_m\Big].$$
Elegir en test lo convierte en un segundo val y deja al proyecto sin una estimación honesta. Por eso el umbral $t$, el postprocesado y la versión del modelo se eligen en val y test se consulta una vez, con todo congelado.

**Incertidumbre de la cifra de test.** Con $n_{ok}$ imágenes evaluadas, desviación estándar muestral $s$ del IoU por imagen y media $\overline{\mathrm{IoU}}$, el intervalo aproximado al 95 % para la media es
$$\overline{\mathrm{IoU}}\ \pm\ 1.96\,\frac{s}{\sqrt{n_{ok}}}.$$
Para una clase pequeña como kalibaush (aprox. 79 imágenes en test antes del filtro `ok`) el intervalo es bastante más ancho que el global. Diferencias de uno o dos puntos de IoU entre variantes no son concluyentes, y este intervalo mide la variabilidad por la muestra de test, no la debida al entrenamiento (semilla, orden de lotes).

## 4. Pipeline paso a paso

Formato del notebook (estilo del equipo): docstring de encabezado (Autor, Código, Materia, Tema); celda única de imports con comentario en línea por import; markdown corto en primera persona plural ("Ahora vamos a ..."); una celda de código por paso, con comentarios en línea. Los pasos 1 a 16 son el cuaderno de trabajo (D3–D5) y solo tocan train y val. Los pasos 17 a 23 son la sección final de test.

1. **Configuración y entorno.** Docstring de encabezado, celda única de imports (incluir `matplotlib.pyplot`: es el error (d) del cuaderno CIFAR-10) y celda de configuración con las constantes comunes: `SEED = 42` en `random`, NumPy y `keras.utils.set_random_seed`, lado de entrada 256, rutas de Drive bajo `MyDrive/sylfish/`, orden de clases de `classes.txt`. GPU T4 y Drive montado. Descomprimir el zip de datos en `/content` (nunca leer miles de archivos directo desde Drive). Verificar: GPU visible, rutas existentes, versión de TensorFlow/Keras impresa.
2. **Verificar la caché U-Net (D3).** Este cuaderno no genera los `.npz`: comprobar que `unet_256_train.npz` y `unet_256_val.npz` existen en `data/processed/`; si faltan, se generan en el cuaderno 01 (sección de caché) y no aquí, para que el preprocesado de 256 que genera el caché viva en un solo lugar (la única excepción es la exportación de `unet_pred_test/` del paso 23, que cubre imágenes fuera del caché). Verificar: claves `X`, `M`, `y`, `filepath`; formas, `dtype` `uint8`, `np.unique(M)` igual a $\{0,1\}$; conteo por split y por clase igual al de las filas `ok` de `mask_qc.csv`; y cuántas descartadas o "revisar" quedaron fuera (tabla, que se guarda como `unet_excluidas.csv`). De `unet_256_test.npz` se comprueba solo que el archivo exista (por ejemplo con `os.path.exists`); no se abre antes del paso 17.
3. **Carga y verificación visual.** Abrir los `.npz` de train y val (copiados a `/content`). Grilla de 12 pares imagen/overlay de máscara de train. Verificar: la máscara cae sobre el pez (si está desplazada o espejada, el orden de `X` y `M` se rompió).
4. **Aumento sincronizado.** Pipeline `tf.data` solo para train: convertir imagen a `float32` y máscara a `float32`, concatenarlas en un tensor de 4 canales y aplicar un `keras.Sequential` geométrico (`RandomFlip("horizontal_and_vertical")`, `RandomRotation`, `RandomZoom`). Como es un único tensor, la imagen y la máscara reciben exactamente la misma transformación. Luego separar los canales, re-binarizar la máscara ($\ge0.5\to1$; la interpolación bilineal crea valores intermedios en el borde) y aplicar el aumento de color (`RandomContrast` o `RandomBrightness`) SOLO a los 3 canales de imagen. Val sin aumento. `batch(16)`, `prefetch`. Verificar: grilla de 8 aumentos con overlay alineado y `np.unique` de la máscara aumentada igual a $\{0,1\}$.
5. **Modelo.** Construir la U-Net de la tabla de la sección 3 con la API funcional, con una celda previa que define el bloque doble. Verificar: `model.summary()` con salida $(None,256,256,1)$ y $\approx7.77$ M parámetros.
6. **Pérdida y métricas.** Pérdida propia BCE + Dice (BCE con `keras.losses.BinaryCrossentropy` más el término Dice por imagen). Métricas: `keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)` y un Dice propio. Registrar las funciones propias con `keras.saving.register_keras_serializable` para poder recargar el `.keras`. Esta es también la celda que define el IoU y el Dice por imagen con NumPy que se usan en los pasos 9 y 19.
7. **Compilación y callbacks.** `Adam` con $\eta=10^{-3}$. Callbacks en `fit`: `EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)`, `ModelCheckpoint` a `unet_best.keras` con `save_best_only=True`, un segundo `ModelCheckpoint` por época que sobrescribe `unet_last.keras`, `ReduceLROnPlateau` y `CSVLogger`. Verificar que los callbacks se pasan a `fit` (error (b) del cuaderno CIFAR-10).
8. **Entrenamiento v1 (D4).** `fit` con el dataset de train y val, de 30 a 60 épocas. Verificar: curvas de pérdida, IoU y Dice en train/val; `val_loss` decrece y se estabiliza.
9. **Predicción en val y métricas por imagen.** `model.predict` sobre todo `X_val`, umbral $t=0.5$. Para cada imagen calcular IoU y Dice con NumPy contra `M`. Convención: si predicción y referencia están ambas vacías, $\mathrm{IoU}=1$; si solo la predicción está vacía, $\mathrm{IoU}=0$. Con el filtro `ok` no debería ocurrir. Verificar: un valor por imagen, valores en $[0,1]$; media y mediana globales.
10. **Ajuste de umbral (D5).** Barrer $t\in\{0.30,0.35,\dots,0.70\}$ sobre las probabilidades de val y elegir el que maximiza el IoU medio por imagen. Si la mejora frente a 0.5 es menor que 0.005, se mantiene 0.5. Verificar: tabla y gráfico IoU medio vs. $t$. Este paso y el 13 son las únicas decisiones sobre el umbral y el postprocesado; ninguna usa test.
11. **Evaluación por clase de pez en val.** Agrupar IoU y Dice por `y`: media, mediana, desviación y $N$ por clase. Histograma de IoU global. Verificar: 9 filas, $N$ total igual al número de imágenes de val `ok`, y media global igual al promedio ponderado de las medias por clase.
12. **Peores casos en val.** Las 10 imágenes de menor IoU en una figura de 10 filas × 4 columnas: imagen, máscara de referencia, predicción, overlay. Para cada una, anotar si falla la predicción o la pseudo-máscara de referencia (la referencia también puede estar mal). Categorías sugeridas, reutilizables en test: sombra pegada al pez, aletas o cola recortadas, huecos por reflejos plateados, marca de agua incluida, pez que toca el borde, o referencia equivocada (la pseudo-máscara está peor que la predicción).
13. **Postprocesado opcional.** Quedarse con la componente conexa mayor (`cv2.connectedComponentsWithStats`) y rellenar huecos. Evaluar en val con y sin postprocesado; se adopta solo si sube el IoU medio. La decisión se pasa al cuaderno 04.
14. **U-Net v2 (D5).** Una sola variable por experimento (intensidad del aumento, $\eta$, peso relativo de Dice). Comparación en val con la misma partición y semilla, registrada en `unet_experimentos.csv`. Elegir la final, recargar `unet_best.keras` y repetir los pasos 9–11 para confirmar las cifras.
15. **Conjunto oro (si existe).** IoU de la U-Net contra las máscaras manuales y de la pseudo-máscara contra las manuales. Si la U-Net se acerca o supera a la pseudo-máscara, es evidencia de que generaliza más allá del ruido de OpenCV.
16. **Congelar la U-Net (mié 11, antes de las 23:59).** Guardar `reports/tables/unet_metricas_val.json` con IoU y Dice medios de val, umbral $t$ elegido y si se usa postprocesado, y calcular el MD5 de `unet_best.keras` con `hashlib` para que el cuaderno 04 lo registre en `config_inferencia.json`. A partir de aquí no hay reentrenamiento ni cambio de $t$.

### Sección final del cuaderno — Evaluación en test (ejecutar una sola vez el jueves 12)

Esta sección va separada del resto, con un título markdown que diga "ejecutar una sola vez el jueves 12" y con modelo, umbral y postprocesado ya congelados. Aquí solo se mide, no se decide nada.

17. **Verificar el congelamiento y abrir test.** Leer `unet_metricas_val.json` (umbral $t$ y postprocesado), contrastarlo con `config_inferencia.json` (que el cuaderno 04 escribió el D6, así que existe) y recalcular el MD5 de `unet_best.keras` con `hashlib` para compararlo con el registrado allí. Verificar: coinciden; si no, detenerse y aclararlo con el equipo antes de seguir. Solo entonces abrir `unet_256_test.npz` con `np.load`. Cargar el modelo con `keras.models.load_model` y `compile=False` (la pérdida y las métricas propias no hacen falta para inferir). Verificar: `model.input_shape` es $(None,256,256,3)$ y la salida tiene 1 canal.
18. **Tabla maestra de test y alineación.** Leer `splits.csv` con Pandas, filtrar `split == "test"` y unir con `mask_qc.csv` por `filepath` para traer `estado`. Verificar: número de filas del `.npz` igual al número de imágenes de test con `estado == "ok"` en la tabla (por contrato, `unet_256_test.npz` trae solo máscaras `ok`; el total de test, aprox. 1362, sale de `splits.csv`); `X.dtype == uint8`; forma $(n_U,256,256,3)$; `np.unique(M)` igual a $\{0,1\}$; y que, al unir por la clave `filepath` del `.npz`, el vector `y` coincide elemento a elemento con la columna `label` de la tabla (evita el error (c) del cuaderno CIFAR-10, predicciones desalineadas). Reportar $n_{ok}$ y cuántas imágenes de test quedaron fuera por no ser `ok`.
19. **Pasada única de predicción y métricas por imagen.** `model.predict` sobre `X` con `batch_size` 16, sin aumento. Binarizar con el $t$ congelado y aplicar la componente conexa mayor solo si así quedó congelado. Calcular IoU y Dice por imagen contra `M` con la misma celda del paso 6, únicamente en filas con `estado == "ok"`, con la misma convención de imágenes vacías. Esta celda se ejecuta una sola vez y su salida queda guardada en el cuaderno. Verificar: valores en $[0,1]$ y número de imágenes evaluadas igual a $n_{ok}$.
20. **Métricas globales y por clase.** Media, mediana y desviación de IoU y Dice; intervalo aproximado de la media con la fórmula de la sección 3; desglose por clase de pez con `groupby` sobre la tabla maestra (media, mediana, $N$); histograma de IoU global y diagrama de cajas de IoU por clase. Verificar: 9 filas, $N$ total igual a $n_{ok}$ y media global igual al promedio ponderado de las medias por clase.
21. **Peores casos en test.** Los 10 casos de menor IoU en cuatro columnas (imagen, referencia, predicción, overlay). Clasificar cada caso en una de las categorías del paso 12 y contar casos por categoría. Es descripción: no se usa para cambiar nada.
22. **Exportar tablas y figuras.** Guardar las tablas y figuras de test con los nombres de la sección 8 (`plt.savefig` con `dpi=200` y `bbox_inches="tight"`). La tabla por imagen (`filepath`, `clase`, `label`, `estado`, `iou`, `dice`) es la que usa el cuaderno 04 para calcular la métrica del sistema completo, así que debe conservar el orden y la clave `filepath`.
23. **Exportar `unet_pred_test/` para la comparación con ML.** Para todas las imágenes de test de `splits.csv` (no solo las `ok`, para que ML pueda comparar con sus máscaras propias): leer el JPG desde `/content/sylfishbd/images/` con `cv2.imread`, convertir con `cv2.cvtColor` y `cv2.COLOR_BGR2RGB`, redimensionar a 256 con la misma interpolación que usa el cuaderno 01 para generar la caché (`cv2.INTER_AREA`), predecir, binarizar con $t$, aplicar el postprocesado si se congeló, reescalar a la forma original $(H,W)$ con `cv2.INTER_NEAREST` y guardar con `cv2.imwrite` como PNG multiplicando por 255. Esta celda es la única de este cuaderno que preprocesa imágenes, porque las no `ok` no están en el caché; es una excepción declarada y por eso exige la prueba de paridad. Antes de exportar, comprobar la paridad del preprocesado: para unas pocas imágenes de test, el arreglo redimensionado en esta celda debe ser idéntico al `X` del `.npz` (misma `filepath`). Es inferencia, no reevaluación: no cambia ninguna métrica de este cuaderno. Verificar: número de PNG igual al de test en `splits.csv`; en una muestra releída, valores únicos $\{0,255\}$ y misma forma que el JPG.

## 5. Decisiones de diseño y alternativas

- **Binaria y no multiclase por especie.** La especie la resuelve la CNN; una U-Net de 10 clases duplicaría la tarea de clasificación con pseudo-máscaras ruidosas. Descartada.
- **`Conv2DTranspose` en lugar de `UpSampling2D`.** Sobremuestreo aprendido. `UpSampling2D` + `Conv2D` es la alternativa si aparecen artefactos en tablero de ajedrez.
- **BCE + Dice en lugar de solo BCE.** Por el desbalance pez/fondo explicado arriba. Solo Dice es inestable al inicio y en imágenes con poco pez.
- **Solo máscaras `ok`.** Entrenar con máscaras dudosas enseña errores; perder algunas imágenes es más barato. En test se evalúa también solo sobre `ok`: medir contra referencias dudosas mide el error de la referencia, no el del modelo; se reporta $n_{ok}$ y cuántas quedaron fuera.
- **Aumento geométrico compartido en 4 canales.** Garantiza la alineación por construcción. La alternativa de aplicar dos capas con la misma semilla es frágil.
- **Evaluación a $256\times256$** contra la máscara redimensionada con vecino más cercano, tanto en val como en test, para que las cifras sean comparables. Decisión sugerida (confirmar con el equipo): las métricas oficiales se miden a 256; reescalar a la resolución original solo se hace en la demo y en la exportación de `unet_pred_test/`. El IoU a resolución original puede reportarse aparte si sobra tiempo, pero mezcla el error del reescalado.
- **Métrica por imagen.** `BinaryIoU` de Keras acumula la matriz de confusión de todos los píxeles del conjunto (IoU agregado), que pondera más a los peces grandes, y con su configuración por defecto (`target_class_ids=(0, 1)`) promedia IoU de fondo y de pez, lo que infla la cifra. El contrato pide IoU por imagen promediado, calculado con NumPy después de predecir.
- **Umbral y postprocesado fijos en test.** Si se quiere mostrar el efecto del postprocesado en test, se reporta como ablación secundaria, marcada como tal, sin elegir con ella.
- **Intervalo con fórmula normal, no bootstrap.** Para una media de IoU por imagen basta el error estándar $s/\sqrt{n_{ok}}$; el bootstrap por `group_id` queda para el F1 macro de la CNN en el cuaderno 02. Decisión sugerida (confirmar con el equipo).

## 6. Errores comunes y trampas

- Redimensionar la máscara con interpolación bilineal o de área: aparecen valores intermedios y bordes falsos. Siempre `cv2.INTER_NEAREST` y binarizar.
- Máscaras en $\{0,255\}$ dentro de la pérdida: la BCE espera $\{0,1\}$.
- Aplicar flips o rotaciones a la imagen y a la máscara por separado: quedan desalineadas sin dar error.
- Aplicar contraste o brillo a la máscara.
- Olvidar re-binarizar la máscara tras rotar o hacer zoom.
- Imagen en BGR en entrenamiento y RGB en la demo (o al revés); en la exportación de `unet_pred_test/` también: `cv2.imread` entrega BGR.
- Ajustar el umbral o el postprocesado mirando test, o abrir `unet_256_test.npz` antes de la sección final.
- Correr la celda de predicción de test dos veces "porque algo falló" y quedarse con el mejor resultado: eso es selección en test. Si hay un error técnico (por ejemplo, desalineación), se corrige, se documenta en el cuaderno y se reporta el resultado corregido, sin tocar modelo ni umbral.
- Predicciones desalineadas con etiquetas por haber filtrado o reordenado una tabla y no la otra.
- Dividir entre 255 antes de `predict`: la U-Net ya trae `Rescaling(1/255)` dentro; normalizar dos veces deja la entrada en $[0,1/255]$.
- Calcular IoU con la salida sigmoide sin binarizar.
- Cargar `unet_best.keras` en la demo sin registrar la pérdida propia: falla `load_model`. Alternativa: cargar con `compile=False`, porque para inferir no se necesita la pérdida.
- Reportar el IoU agregado de `BinaryIoU` como si fuera el IoU medio por imagen, o usar `BinaryIoU` con su configuración por defecto (mezcla fondo y pez).
- Interpretar un IoU bajo como fallo de la red sin mirar la pseudo-máscara de referencia; presentar el IoU como "precisión de segmentación real" cuando es concordancia con pseudo-máscaras de OpenCV.
- Interpretar el IoU de una clase pequeña como exacto con tan pocas imágenes.
- Quedarse sin RAM al convertir todo a `float32`: la conversión se hace por lote dentro de `tf.data`.

## 7. Criterio de aceptación

- [ ] Caché U-Net de train y val verificada con formas y valores del contrato; `np.unique(M)` igual a $\{0,1\}$.
- [ ] Solo imágenes con `estado = ok` en train y val; tabla de excluidas.
- [ ] Grilla de aumento sincronizado con overlays alineados.
- [ ] `model.summary()` con salida $256\times256\times1$.
- [ ] Callbacks pasados a `fit`; historial en CSV.
- [ ] IoU y Dice por imagen en val: media global y tabla por clase de pez.
- [ ] Umbral elegido en val con su tabla; test no se tocó antes del jueves 12.
- [ ] Figura de los 10 peores casos de val con diagnóstico breve.
- [ ] `unet_best.keras` recargado (con `compile=False` o con objetos registrados) reproduce las métricas de val.
- [ ] `unet_metricas_val.json` y MD5 de `unet_best.keras` entregados antes del congelamiento.
- [ ] Sección de test ejecutada una sola vez, con su salida guardada en el cuaderno y la alineación `y` ↔ `label` verificada en una celda explícita.
- [ ] IoU y Dice en test: media, mediana y por clase, con $n_{ok}$ reportado.
- [ ] Peores 10 casos de test clasificados por categoría.
- [ ] `unet_pred_test/` exportado con el formato del contrato y paridad de preprocesado comprobada.
- [ ] Limitación de las pseudo-máscaras redactada para el informe.

## 8. Artefactos que debe dejar

- Cuaderno: `notebooks/03_unet/03_unet.ipynb` (salidas limpiadas por `nbstripout` en el commit; las cifras viven en las tablas).
- (La caché `data/processed/unet_256_{split}.npz` la deja el cuaderno 01; aquí solo se verifica.)
- `MyDrive/sylfish/models/unet_best.keras` y `MyDrive/sylfish/models/unet_last.keras`.
- Val: `reports/tables/unet_historial_<experimento>.csv`, `reports/tables/unet_experimentos.csv`, `reports/tables/unet_metricas_val.json` (IoU y Dice medios, umbral elegido, si se usa postprocesado), `reports/tables/unet_por_clase_val.csv`, `reports/tables/unet_umbral_val.csv`, `reports/tables/unet_excluidas.csv`.
- Val, figuras: `reports/figures/unet_curvas.png`, `reports/figures/unet_aumento_sincronizado.png`, `reports/figures/unet_hist_iou_val.png`, `reports/figures/unet_peores10_val.png`, `reports/figures/unet_umbral_val.png`, `reports/figures/unet_ejemplos_val.png`.
- Test: `reports/tables/unet_iou_test.csv` (una fila por imagen con `filepath`, `clase`, `label`, `estado`, `iou`, `dice`), `reports/tables/unet_metricas_test.json` (IoU y Dice medios, medianas, intervalo, $n$ y $n_{ok}$, umbral y postprocesado usados), `reports/tables/unet_por_clase_test.csv`, `reports/tables/unet_errores_segmentacion_test.csv` (los 10 peores casos con su categoría).
- Test, figuras: `reports/figures/unet_hist_iou_test.png`, `reports/figures/unet_iou_por_clase_test.png`, `reports/figures/unet_peores10_test.png`.
- Comparación con ML: `MyDrive/sylfish/comparacion/unet_pred_test/<clase>/<clase>_NNN.png`.

## 9. Preguntas probables de sustentación

1. **¿Por qué U-Net y no una CNN de clasificación?** La clasificación da una etiqueta por imagen; la segmentación necesita una etiqueta por píxel. La U-Net recupera la resolución completa con el decoder y las skip connections.
2. **¿Qué hacen las skip connections?** Llevan mapas de alta resolución del encoder al decoder. El decoder aporta el "qué" y el skip el "dónde exactamente", así se recuperan bordes finos que el pooling había perdido.
3. **¿Por qué la entrada es de 256?** Hay 4 poolings de factor 2; el lado debe ser múltiplo de $2^4=16$ para que al subir las dimensiones coincidan con los skips. 256 cumple y es manejable en una T4 con lote 16.
4. **¿Por qué BCE + Dice?** El fondo domina los píxeles; la BCE sola puede conformarse con predecir fondo. El Dice mide solapamiento relativo al tamaño del pez e ignora los verdaderos negativos. La BCE da gradientes estables.
5. **¿Qué relación hay entre IoU y Dice?** $\mathrm{Dice}=\frac{2\,\mathrm{IoU}}{1+\mathrm{IoU}}$ por imagen: ordenan igual y Dice siempre es mayor o igual; es la misma información en otra escala.
6. **Si no había máscaras, ¿contra qué evalúan y qué tan seguros están de la cifra?** Contra pseudo-máscaras de OpenCV con control de calidad, solo en estado `ok`. La U-Net aprende a imitarlas; es una limitación declarada, y varios "errores" son fallos de la referencia. El conjunto oro manual, si existe, da una medida contra verdad humana. La incertidumbre de la media se resume con $1.96\,s/\sqrt{n_{ok}}$ y es mayor en las clases pequeñas.
7. **¿Cómo aumentan datos sin desalinear la máscara?** Concatenamos imagen y máscara en 4 canales, aplicamos la misma transformación geométrica, separamos y re-binarizamos la máscara. El color solo se altera en la imagen.
8. **¿Cómo eligieron el umbral y por qué test se usó una sola vez?** Barrido en val maximizando el IoU medio por imagen. Elegir o repetir en test sesga la cifra hacia arriba (el máximo de estimaciones ruidosas sobreestima), así que test se midió una vez con modelo, umbral y postprocesado congelados.
9. **¿En qué clases segmenta peor y por qué?** Se responde con la tabla por clase y los peores casos de val y test (por ejemplo, peces de color parecido al fondo, sombras pegadas al pez o aletas translúcidas que la pseudo-máscara recorta).
10. **¿Para qué sirve el postprocesado de componente mayor?** Elimina manchas falsas aisladas (reflejos, marca de agua), porque en estas fotos hay un solo pez. Se adoptó solo si mejoró el IoU en val.
