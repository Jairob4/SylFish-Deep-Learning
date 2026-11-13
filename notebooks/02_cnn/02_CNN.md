# Módulo 02 — Clasificación con CNN (de cero y transfer learning)

**Autor del módulo:** compañero de equipo (CNN) · **Documento:** Jairo (revisión y debate)
**Materia:** Inteligencia Artificial 2026-2 · **Entrada:** `cnn_224_{train,val,test}.npz` (módulo 01) · **Salida:** `models/cnn_best.keras`, `comparacion/pred_val_cnn.csv`

> **Verificado el 2026-11-13:** `cnn_best.keras` pesa 29.1 MB y sus capas son `aumento`, `efficientnetb0`, `gap_tl`, `dropout_tl`, `salida_tl` (sin `rescalado`): es el modelo de transfer learning (`tl_fase2`). **Pendiente (⚠):** reproducibilidad de la línea base (ver 4.5); cifras derivadas a mano (conteo de errores) y el contenido de `config_inferencia.json`.

---

## 1. Propósito

Dada una imagen de pez, devolver **la especie (9 clases)** y una **confianza** (máximo del softmax). Es la parte de clasificación de la demo (módulo 04); la máscara la aporta el U-Net (módulo 03).

El módulo compara dos familias bajo la misma partición (`splits.csv`), la misma semilla (42) y el mismo presupuesto de épocas (60):

1. Una CNN propia entrenada desde cero (línea base) y tres variantes de un solo cambio.
2. Transfer learning con EfficientNetB0 preentrenada en ImageNet, en dos fases.

Se elige por **F1 macro en val**; **test se evalúa una sola vez** con el modelo congelado.

## 2. Preconceptos

**Partición por grupos.** Las imágenes casi duplicadas comparten grupo y viven en el mismo conjunto (`StratifiedGroupKFold`, módulo 01). Así val/test no filtran información de train.

**Desbalance de especies.** rui (1670) vs kalibaush (525): razón ≈ 3.2. Se compensa con pesos por clase

$$w_c = \frac{N}{K\,N_c},$$

con $N$ el total de imágenes de train, $K=9$ y $N_c$ las de la clase $c$. La pérdida pasa a ser $\sum_i w_{y_i}\,\ell(\hat y_i, y_i)$.

**Métricas.** Accuracy $=\frac{1}{N}\sum_i \mathbf 1[\hat y_i=y_i]$ puede ocultar fallos en clases pequeñas. **F1 macro** promedia el F1 de cada clase con el mismo peso:

$$F1_{\text{macro}}=\frac{1}{K}\sum_{c=1}^{K}\frac{2\,P_c\,R_c}{P_c+R_c}.$$

**Batch Normalization.** En entrenamiento normaliza con media y varianza del lote; en inferencia usa estadísticos acumulados (media móvil). Si esos estadísticos se desajustan respecto de la distribución de val, la accuracy de val puede oscilar muy por debajo de la de train (ver 4.3).

**Transfer learning en dos fases.** Fase 1: backbone congelado, se entrena solo la cabeza. Fase 2 (fine-tuning): se descongelan las últimas capas con $\eta$ muy pequeña y las BatchNorm **siguen congeladas** (modo inferencia), para no destruir lo aprendido con pocas imágenes.

**Confianza.** $\max_k \hat y_k$ del softmax. Las redes con entropía cruzada tienden a ser sobreconfiadas: es una preferencia relativa, no una probabilidad calibrada.

## 3. Decisiones y datos

| Decisión | Valor | Por qué |
|---|---|---|
| Datos | train 6246 / val 1359 / test 1370 | Caché 224×224 RGB, verificado fila a fila contra `splits.csv` (rutas, orden, conteos por clase) |
| Redimensionado | 224×224, `cv2.INTER_AREA`, RGB | Promedia píxeles al reducir; es el contrato de entrada de la demo |
| Desbalance | `class_weight` (fórmula de 2) | Razón 3.2 entre clases |
| Aumento (dentro del modelo) | flip H+V, rotación 0.1, zoom 0.1, contraste 0.1 | Solo activo en entrenamiento |
| Línea base | 5 bloques Conv→BN→ReLU→MaxPool (32/64/128/256/256), SpatialDropout 0.3, GAP, Dense 256+BN+ReLU, Dropout 0.2, Dense 9 | 1,050,569 parámetros |
| Normalización de entrada | línea base: `Rescaling(1/255)`; EfficientNetB0: **ninguna externa** (normaliza por dentro, espera 0–255) | Cambia respecto del plan inicial (Xception con `preprocess_input` en [-1,1]) |
| Optimización | Adam, $\eta=10^{-3}$, lote 32, 60 épocas máx. | |
| Callbacks | EarlyStopping (val_loss, paciencia 8, restaura mejores pesos), ReduceLROnPlateau (×0.5, paciencia 3), 2 ModelCheckpoint, CSVLogger | |
| TL fase 1 | EfficientNetB0 congelada + GAP + Dropout 0.3 + Dense 9; $\eta=10^{-3}$ | 11,529 parámetros entrenables de 4,061,100 |
| TL fase 2 | últimas 30 capas del backbone entrenables (sin BN); $\eta=10^{-5}$, máx. 15 épocas | Backbone llamado con `training=False` |
| Selección | mayor F1 macro en val; desempate por menor val_loss y menor tamaño | |

**Experimentos de la línea base (un cambio por experimento):** `l2` (L2 $10^{-4}$ en convs y Dense), `aumento` (rotación 0.25 + RandomErasing 0.1), `sin_cw` (sin pesos de clase).

## 4. Resultados

### 4.1 Comparación en val

| Experimento | Accuracy val | F1 macro val | Mejor época |
|---|---|---|---|
| `tl_fase2` | **0.9875** | **0.9892** | 13 |
| `tl_fase1` | 0.9772 | 0.9810 | 35 |
| `linea_base` | 0.9691 | 0.9693 | 33 |
| `l2` | 0.9610 | 0.9610 | 35 |
| `aumento` | 0.7035 | 0.7354 | 4 |
| `sin_cw` | 0.4982 | 0.4525 | 2 |

Ganador: **`tl_fase2`**. El transfer learning mejora a la CNN de cero en ≈ 2 puntos de accuracy y ≈ 2 de F1 macro. ⚠ Con 1359 imágenes, 1 punto de accuracy ≈ 14 imágenes; además es **una sola semilla**, así que la diferencia entre `tl_fase1`, `linea_base` y `l2` es pequeña frente al ruido esperable.

### 4.2 Test (una sola vez, modelo congelado `tl_fase2`)

| Accuracy test | F1 macro test |
|---|---|
| **0.9942** | **0.9955** |

Es mayor que en val (0.9875 / 0.9892). La diferencia equivale a ≈ 9 imágenes ⚠ (cálculo a mano: $0.9942\cdot1370\approx1362$ aciertos, ≈ 8 errores, frente a ≈ 17 en val) y no indica por sí sola ningún problema; test no se usó para elegir nada. Conviene reportar **ambas** cifras, sin presentar test como "mejor modelo".

### 4.3 Experimentos `aumento` y `sin_cw`: no concluyentes

En ambos, la accuracy de train llega a ≈ 0.95 mientras la de val oscila entre ≈ 0.23 y ≈ 0.71, con val_loss entre 1 y 9; el EarlyStopping restauró pesos de la época 2–5. Un desbalance de 3.2× no puede explicar una brecha de ese tamaño: es **inestabilidad de entrenamiento** (BatchNorm, $\eta=10^{-3}$, una semilla), no un efecto medido del balanceo ni del aumento. Redacción recomendada para el informe: "no concluyente". Para medir de verdad el efecto del balanceo habría que comparar el **recall por clase** (rui, kalibaush) con varias semillas.

### 4.4 Chequeo de atajos (fondo neutralizado)

Sobre las 1066 imágenes de val con `qc_ok`, se deja el pez intacto y el fondo en gris (128) usando la máscara GrabCut.

| Versión | Accuracy | F1 macro |
|---|---|---|
| Original | 0.9906 | 0.9913 |
| Fondo neutralizado | 0.8180 | 0.8263 |

Caída de ≈ 17 puntos. Por clase (F1): rui 0.977→0.643, mrigel 0.988→0.773, katla 0.991→0.792, ilish 0.988→0.794; koi casi no cambia (1.000→0.994).

**Lectura prudente:** el modelo **depende del contexto** (fondo/escena). No es prueba pura de "atajo", porque el gris plano y los bordes imperfectos de la máscara son también un cambio de distribución que el modelo nunca vio. Aun así, 0.818 está muy por encima del azar (≈ 0.11), así que sí usa rasgos del pez.

### 4.5 Confianza y archivos

- Confianza media en aciertos: **0.986**; en errores: **0.722** → sirve como señal relativa de duda.
- Verificación de archivos (2026-11-13). En la celda de selección, `linea_base` apareció con tamaño 29.07 MB y val_loss 1.5985, frente a 0.0993 en la verificación previa. En Drive, `cnn_best.keras` y `cnn_best_tl_fase2.keras` pesan igual (29.07 MB), de modo que `cnn_best.keras` es el modelo de transfer learning que usa la demo (tamaño 29.1 MB y capas `efficientnetb0` comprobados). Pero `cnn_historial_linea_base.csv` (1692 bytes, creado el 13 de noviembre a las 14:53 UTC) y `cnn_best_linea_base.keras` (12.7 MB, modificado a las 15:01 UTC) son **posteriores** a la tabla de experimentos (02:39 UTC), y el CSV es mucho más corto que el de `l2` (4550 bytes, 44 épocas). Es probable que la línea base se haya vuelto a entrenar o a guardar después, con una corrida más corta. **Las cifras de la línea base del informe (0.9691 / 0.9693) provienen de la corrida original y puede que ya no se reproduzcan con los archivos actuales; hay que confirmarlo con el compañero antes de citarlas.** Las cifras del modelo final (`tl_fase2`) no se ven afectadas.

## 5. Limitaciones

1. **Una sola semilla** por experimento; no hay intervalos de confianza.
2. **Selección sobre val:** las cifras de val son optimistas; por eso se reporta test una vez.
3. **Experimentos `aumento` y `sin_cw` inestables** (4.3): no permiten concluir sobre aumento fuerte ni sobre balanceo.
4. **Dependencia del fondo** (4.4): la exactitud en distribución (≈ 99 %) probablemente sobreestima el rendimiento con fotos de otros fondos, superficies o cámaras. Entrenar con fondo neutralizado o recortado queda como trabajo futuro.
5. **Fase 2 limitada por presupuesto:** la mejor época fue la 13 de 15 máximas; podría mejorar con más épocas.
6. **Confianza no calibrada:** media 0.722 en errores; no leerla como probabilidad real.
7. **Misma distribución:** test proviene del mismo dataset y la misma partición por grupos; no mide generalización a imágenes externas.
8. **Comparación TL vs. cero no es de una sola variable** (arquitectura, preentrenamiento y tamaño cambian a la vez).
9. **Contrato de preprocesamiento frágil:** CNN espera RGB 224×224 en 0–255 (sin `/255`); el U-Net espera 256×256 con `/255`. Mezclarlos no da error, da predicciones erróneas.

## 6. Glosario

- **Backbone:** red preentrenada sin la capa final de clasificación (aquí EfficientNetB0, salida 7×7×1280).
- **BatchNorm en modo inferencia:** usa medias/varianzas acumuladas, no las del lote.
- **Class weight:** peso $w_c$ por clase en la pérdida.
- **Fine-tuning:** reentrenar parte del backbone con $\eta$ pequeña.
- **F1 macro:** promedio simple del F1 por clase.
- **GAP:** Global Average Pooling, promedia cada mapa de características a un valor.
- **Línea base:** punto de comparación del que parten los experimentos.
- **Neutralizar fondo:** reemplazar el fondo por gris constante usando la máscara.
- **Softmax / confianza:** distribución sobre 9 clases; la confianza es su máximo.
- **Transfer learning:** reutilizar pesos preentrenados en otra tarea.
