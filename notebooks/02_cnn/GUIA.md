# Guía 02 — CNN de clasificación de especies

**Responsable:** Compañero DL · **Depende de:** cuaderno `01_preparacion_datos` (`data/splits/splits.csv`, `data/processed/cnn_224_{split}.npz`, y las máscaras y `mask_qc.csv` solo para el chequeo de atajos) y, para la evaluación en test, de `config_inferencia.json`, que escribe el cuaderno `04_demo` el D6 (paso 10, antes de las 23:59 del miércoles 11) y que este cuaderno solo lee en la sección de test · **Entrega a:** `04_demo`, informe, presentaciones y comparación DL vs ML · **Fecha objetivo:** D1 vie 6 nov (arranque con datos de prueba) → D2 sáb 7 (línea base entrenando) → D3 dom 8 (línea base evaluada) → D4 lun 9 (mejoras) → D5 mar 10 nov (modelo final elegido en val y chequeo de atajos) → D7 jue 12 nov (evaluación en test, una sola vez). Congelamiento: mié 11 a las 23:59.

## 1. De qué se trata

Este cuaderno entrena la red convolucional que responde "¿qué especie es?" entre las 9 clases de SylFishBD y mide su desempeño final. En el Diseño A la CNN es independiente de la U-Net: se entrena sola, con sus propios datos en caché, y solo se une a la segmentación en la demo, donde aporta la clase predicha y la confianza top-3.

El punto de partida es el notebook CIFAR-10 del curso. No se copia: se conserva su esqueleto (bloques Conv–BN–ReLU–Pool, SpatialDropout, GAP, cabeza densa con BN y Dropout, Adam, aumento con capas de Keras) y se corrigen sus cuatro errores conocidos, además de adaptarlo a imágenes de $224\times224$, a 9 clases desbalanceadas y a la partición compartida del equipo. La sustentación debe poder explicar cada cambio frente a la referencia.

El cuaderno es autocontenido, como el de referencia: configuración, carga del caché `.npz`, arquitectura, entrenamiento, mejoras, matriz de confusión, chequeo de atajos y exportación viven en el mismo archivo. No hay módulos `.py` compartidos: las funciones que se reutilizan (evaluación alineada, neutralización del fondo) son celdas del cuaderno. La evaluación en test es la última sección, separada y marcada "ejecutar una sola vez el jueves 12": todo lo anterior se decide en val y esa sección solo mide.

## 2. Entradas y salidas (contrato)

| Tipo | Ruta / objeto | Detalle |
|---|---|---|
| Entrada | `data/processed/cnn_224_{train,val}.npz` (Drive, bajo `MyDrive/sylfish/`; los genera el cuaderno `01_preparacion_datos`) | claves `X` (`uint8`, forma $(N,224,224,3)$, RGB), `y` (enteros 0–8, orden de `classes.txt`) y `filepath` |
| Entrada (solo sección de test) | `data/processed/cnn_224_test.npz` | mismas claves; no se copia ni se abre hasta la última sección |
| Entrada | `data/splits/splits.csv` | trazabilidad (`filepath`, `clase`, `label`, `group_id`, `split`); `group_id` se usa en el bootstrap |
| Entrada (chequeo) | `data/masks/<clase>/<clase>_NNN.png` y `data/masks/mask_qc.csv` | solo imágenes con `estado = ok` |
| Entrada (solo sección de test) | `MyDrive/sylfish/models/config_inferencia.json` | MD5 registrado de `cnn_best.keras`, umbral de confianza $\tau$ y temperatura $T$ si se usó (lo escribe `04_demo`; este cuaderno no lo modifica) |
| Salida | `MyDrive/sylfish/models/cnn_best.keras` | mejor modelo según `val_loss` |
| Salida | checkpoints por época con `ModelCheckpoint` | ver sección 8 |
| Salida | tablas en `reports/tables/` y figuras en `reports/figures/` | ver sección 8 |
| Salida | `MyDrive/sylfish/comparacion/pred_val_cnn.csv` | `filepath`, `label`, `pred`, `conf` del modelo final en val (contrato de comparación) |
| Salida | `MyDrive/sylfish/comparacion/pred_test_cnn.csv` | `filepath`, `label`, `pred`, `conf` del modelo congelado en test; se escribe una sola vez el jueves 12 |

Regla: el modelo recibe `uint8` RGB $224\times224$ en rango $[0,255]$ y normaliza adentro (capa `Rescaling(1/255)`). La demo no debe normalizar por fuera.

## 3. Fundamento teórico para la sustentación

**Convolución discreta.** Una `Conv2D` con kernel $k_h\times k_w$, $C_{in}$ canales de entrada y $C_{out}$ filtros calcula (técnicamente es una correlación cruzada):
$$Y[i,j,c']=b_{c'}+\sum_{c=1}^{C_{in}}\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1}K[u,v,c,c']\,X[i+u,\,j+v,\,c]$$
El mismo kernel se desliza por toda la imagen (pesos compartidos), lo que da equivariancia a traslaciones y muy pocos parámetros. Tamaño espacial de salida: $H_{out}=\left\lfloor\frac{H+2p-k}{s}\right\rfloor+1$; con `padding="same"` y $s=1$, $H_{out}=H$.

**Parámetros de una Conv2D.** $(k_h k_w C_{in}+1)\,C_{out}$. Ejemplos de nuestra red: primer bloque $(3\cdot3\cdot3+1)\cdot32=896$; quinto bloque $(3\cdot3\cdot256+1)\cdot256=590\,080$. Una `BatchNormalization` sobre $C$ canales suma $4C$ parámetros ($2C$ entrenables $\gamma,\beta$ y $2C$ no entrenables: media y varianza móviles). La red de la sección 4 tiene $\approx 1.05\times10^6$ parámetros; debe coincidir con `model.summary()`.

**Campo receptivo.** Es la región de la imagen de entrada que influye en una neurona. Con capas de kernel $k_l$ y paso $s_l$:
$$r_l=r_{l-1}+(k_l-1)\prod_{i<l}s_i,\qquad r_0=1$$
Con 3 bloques (Conv $3\times3$ + Pool $2\times2$) el campo receptivo final es de 22 píxeles: suficiente en $32\times32$ (CIFAR) pero diminuto en $224\times224$, donde el pez ocupa más de 100 píxeles. Con 5 bloques llega a 94 píxeles y el mapa final es de $7\times7$. Esta es la justificación cuantitativa de "más bloques".

**Batch Normalization.** Para cada canal, sobre el mini-lote $\mathcal{B}$ (y todas las posiciones espaciales):
$$\mu_\mathcal{B}=\frac{1}{m}\sum_{i=1}^{m}x_i,\qquad \sigma_\mathcal{B}^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_\mathcal{B})^2,\qquad \hat{x}_i=\frac{x_i-\mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\varepsilon}},\qquad y_i=\gamma\hat{x}_i+\beta$$
Estabiliza la escala de las activaciones, permite tasas de aprendizaje mayores y regulariza levemente. En inferencia usa las medias y varianzas móviles acumuladas en entrenamiento, no las del lote.

**Dropout y SpatialDropout.** Dropout apaga cada activación con probabilidad $p$ durante el entrenamiento y reescala las sobrevivientes: $\tilde{h}=\frac{m\odot h}{1-p}$, con $m_i\sim\mathrm{Bernoulli}(1-p)$. En mapas convolucionales los píxeles vecinos están muy correlacionados, así que apagar píxeles sueltos casi no regulariza; `SpatialDropout2D` apaga mapas de características completos (canales enteros). En inferencia ambos son la identidad.

**Global Average Pooling.** $g_c=\frac{1}{HW}\sum_{i,j}F[i,j,c]$. Convierte el mapa $7\times7\times256$ en un vector de 256. Un `Flatten` + `Dense(256)` necesitaría $12\,544\cdot256\approx3.2\times10^6$ pesos; con GAP la capa densa tiene $(256+1)\cdot256=65\,792$. Menos parámetros, menos sobreajuste.

**Softmax y entropía cruzada.**
$$\mathrm{softmax}(z)_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}},\qquad \mathcal{L}_{CE}=-\sum_{k=1}^{K} y_k\log \hat{y}_k$$
Con etiquetas enteras (no one-hot) se usa `sparse_categorical_crossentropy`, que es lo mismo: $\mathcal{L}_{CE}=-\log\hat{y}_{y}$. Con `class_weight`, la pérdida de cada muestra se multiplica por $w_{y}$, con $w_c=\dfrac{N}{K\,N_c}$ calculado sobre train.

**Adam (conceptual).** Combina momento (promedio móvil del gradiente, $m_t$) y escalado adaptativo por parámetro (promedio móvil del gradiente al cuadrado, $v_t$):
$$\theta_t=\theta_{t-1}-\eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}$$
donde $\hat{m}_t,\hat{v}_t$ son las versiones con corrección de sesgo. Cada parámetro tiene en la práctica su propio paso efectivo; por eso converge rápido sin mucho ajuste de $\eta$.

**Por qué F1 macro.** Por clase, $P_c=\frac{TP_c}{TP_c+FP_c}$, $R_c=\frac{TP_c}{TP_c+FN_c}$, $F1_c=\frac{2P_cR_c}{P_c+R_c}$, y $F1_{macro}=\frac{1}{K}\sum_c F1_c$. La accuracy pondera por tamaño de clase: un modelo que nunca acierta kalibaush ($525/9075\approx5.8\,\%$ de los datos) pierde menos de 6 puntos de accuracy, pero su F1 macro cae del orden de $1/9\approx11$ puntos. Con razón máx/mín $\approx3.18$, la F1 macro es la métrica de selección.

**Matriz de confusión normalizada por filas.** Con $C_{ij}$ el número de imágenes de la clase real $i$ predichas como $j$, la matriz normalizada es $\tilde C_{ij}=C_{ij}/\sum_{j}C_{ij}$. Cada fila suma 1, su diagonal es el recall de cada clase y cada celda fuera de la diagonal es la fracción de esa especie confundida con otra.

**Por qué test se usa una sola vez.** Cada métrica medida en test es una variable aleatoria: depende de qué imágenes cayeron en test. Si se evalúan varios modelos (o varios umbrales) en test y se reporta el mejor, el número reportado está sesgado hacia arriba, porque el máximo de estimaciones ruidosas sobreestima el máximo de los valores verdaderos:
$$\mathbb{E}\Big[\max_m \widehat{\mathrm{Acc}}_m\Big]\ \ge\ \max_m \mathbb{E}\Big[\widehat{\mathrm{Acc}}_m\Big].$$
Elegir en test convierte test en un segundo val y deja al proyecto sin ninguna estimación honesta del desempeño en datos nuevos. Por eso la selección se hace en val y test se consulta una vez, con el modelo congelado.

**Incertidumbre de las métricas.** Con $n$ imágenes, el error estándar de una accuracy $\hat p$ es $\sqrt{\hat p(1-\hat p)/n}$. Con $n\approx 1362$ y $\hat p=0.90$, el intervalo aproximado al 95 % es $\pm 1.96\sqrt{0.09/1362}\approx\pm 0.016$. Para el recall de kalibaush ($n\approx 79$ en test, $\hat p=0.85$) es $\approx\pm 0.079$. Conclusión: diferencias de 1–2 puntos entre modelos no son concluyentes, y el desempeño por clase pequeña es muy incierto.

**Bootstrap percentil del F1 macro.** En cada remuestreo $b=1,\dots,B$ se toma con reemplazo un conjunto de grupos (`group_id`) del mismo tamaño que el original, se calcula $F1^{*b}_{macro}$ con las predicciones ya hechas y el intervalo al 95 % es $\big[q_{0.025},\,q_{0.975}\big]$ de los $B$ valores. Se remuestrea por grupo para respetar los duplicados agrupados. Mide la variabilidad debida a la muestra de test, no la debida al entrenamiento (semilla, orden de lotes).

## 4. Pipeline paso a paso

Formato del cuaderno (estilo del equipo): docstring de encabezado (Autor, Código, Materia, Tema); una sola celda de imports con comentario en línea por import; markdown corto en primera persona plural ("Ahora vamos a ..."); una celda de código por paso, con comentarios en línea. Los pasos 1 a 19 son la parte de desarrollo (solo train y val); los pasos 20 a 28 son la sección final de test.

### Parte A — Desarrollo con train y val

1. **Encabezado e imports.** Docstring (Autor, Código, Materia, Tema) y una sola celda de imports que incluya `matplotlib.pyplot` (así se evita el error (d) de la referencia), `numpy`, `pandas`, `sklearn.metrics`, `sklearn.utils.class_weight`, `cv2` y `keras`. Verificar: la celda corre sin errores tras "Reiniciar sesión".
2. **Configuración y entorno.** Celda de configuración con las constantes comunes del proyecto, idénticas en los cuatro cuadernos: `SEED = 42`, tamaño de entrada $224$, número de clases 9, ruta base `MyDrive/sylfish/`, rutas de `data/processed/`, `models/` y `comparacion/`, y la lista de clases leída de `classes.txt` en su orden (índices 0–8). Verificar GPU T4 (`tf.config.list_physical_devices`), montar Drive y fijar la semilla en `random`, NumPy y `keras.utils.set_random_seed`. Verificar: la GPU aparece y la lista de clases tiene 9 nombres en el orden del contrato.
3. **Carga del caché.** Copiar `cnn_224_train.npz` y `cnn_224_val.npz` de Drive a `/content` y abrirlos con `np.load` (el de test no se copia ni se abre hasta la sección final). Si no existen, se generan en el cuaderno `01_preparacion_datos`, no aquí; este cuaderno no preprocesa imágenes. Verificar: claves `X`, `y`, `filepath`; `X.dtype == uint8`, formas $(N,224,224,3)$, rango $[0,255]$; $N_{train}$ y $N_{val}$ y el conteo por clase iguales a los de `splits.csv`; `filepath` en el mismo orden que las filas de su split. En el D1 se trabaja con datos de prueba; desde el D2, con los oficiales.
4. **Inspección rápida.** Grilla de 9 imágenes de train con su nombre de clase, para confirmar que el color es RGB (peces sin tono azulado) y que etiqueta y nombre coinciden.
5. **Pesos de clase.** `compute_class_weight` de `sklearn.utils.class_weight` con `"balanced"`, `classes` = `np.arange(9)` e `y` = `y_train`; convertir a diccionario. Verificar: kalibaush tiene el peso mayor ($\approx1.9$) y rui el menor ($\approx0.6$).
6. **Bloque de aumento.** Un `keras.Sequential` con `RandomFlip("horizontal_and_vertical")`, `RandomRotation` (factor 0.1 a 0.25, `fill_mode="reflect"`), `RandomZoom(0.1)` y `RandomContrast(0.1)`. Justificación: las fotos son cenitales sobre superficie lisa, así que la orientación del pez es arbitraria (flips y rotación no cambian la especie), la distancia de la cámara varía (zoom) y la iluminación cambia entre tomas (contraste). No se usa aumento de tono/saturación: el color es una pista legítima de especie. Verificar: grilla de 8 versiones aumentadas de una misma imagen.
7. **Modelo.** API funcional: `Input((224,224,3))` → aumento → `Rescaling(1/255)` → 5 bloques [`Conv2D` $3\times3$ `padding="same"` → `BatchNormalization` → `ReLU` → `MaxPooling2D` $2\times2$] con 32, 64, 128, 256, 256 filtros → `SpatialDropout2D(0.3)` → `GlobalAveragePooling2D` → `Dense(256)` → `BatchNormalization` → `ReLU` → `Dropout(0.2)` → `Dense(9, activation="softmax")`. Las capas de aumento solo actúan con `training=True`, así que el modelo guardado sirve tal cual para inferencia. Verificar: `model.summary()` con salida $(None,9)$, mapa previo al GAP de $7\times7\times256$ y $\approx1.05$ M parámetros.
8. **Compilación.** `keras.optimizers.Adam` con $\eta=10^{-3}$, pérdida `sparse_categorical_crossentropy`, métrica `accuracy`.
9. **Callbacks.** `EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)`, `ModelCheckpoint` hacia `cnn_best.keras` con `save_best_only=True`, un segundo `ModelCheckpoint` por época, `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)` y `CSVLogger` para el historial. Todos en la lista `callbacks=[...]` de `fit` (error (b) de la referencia). Verificar: al imprimir la lista aparecen los cinco objetos.
10. **Entrenamiento de la línea base.** `model.fit` con `X_train`, `y_train`, `validation_data=(X_val, y_val)` (nunca test: error (a)), `class_weight`, `batch_size=32`, `epochs=60`. Verificar: el entrenamiento se detiene antes de 60 épocas o termina con LR reducida, y el `.csv` del historial existe en Drive.
11. **Curvas.** Pérdida y accuracy de train y val por época desde el `CSVLogger`. Verificar: la brecha train–val y la época del mínimo de `val_loss`.
12. **Evaluación alineada en val.** Celda que define la rutina de evaluación (entradas: modelo, `X`, `y`; salidas: probabilidades, predicciones y métricas) y que se reutiliza en los pasos 16, 18 y 23. `model.predict` sobre TODO `X_val` (sin `shuffle`, sin slices), `argmax` por fila y comparar con `y_val` en el mismo orden (error (c)). Calcular `accuracy_score`, `f1_score(average="macro")`, `classification_report` con `target_names` de `classes.txt` y `confusion_matrix(normalize="true")`, graficada con `ConfusionMatrixDisplay`. Verificar: `len(y_pred) == len(y_val)` y la diagonal de la matriz normalizada está en $[0,1]$ con filas que suman 1.
13. **Registro del experimento.** Una fila en la tabla de experimentos con identificador, cambios respecto al anterior, accuracy y F1 macro en val, época del mejor modelo.
14. **Fase de mejoras (D4).** Probar UNA variable por experimento, con la misma partición, semilla y presupuesto de épocas: (a) intensidad del aumento (rotación mayor, `RandomErasing` de la referencia con baja probabilidad); (b) regularización L2 con `kernel_regularizer=keras.regularizers.L2(1e-4)` en las `Conv2D` y la `Dense`, que suma $\lambda\sum_l\lVert W_l\rVert_2^2$ a la pérdida; (c) con y sin `class_weight` para medir su efecto en F1 macro.
15. **Transfer learning opcional.** Solo si la línea base queda por debajo de lo esperado o sobra tiempo el D4. `keras.applications.MobileNetV2` o `EfficientNetB0` con `include_top=False` y pesos `imagenet`. Fase 1: base congelada (`trainable=False`, llamada con `training=False` para que sus BN queden en modo inferencia), cabeza GAP + Dropout + `Dense(9)`, $\eta=10^{-3}$. Fase 2 (fine-tuning): descongelar las últimas decenas de capas, recompilar con $\eta\approx10^{-5}$ y pocas épocas. El preprocesado va dentro del modelo y es el de cada backbone: `keras.applications.mobilenet_v2.preprocess_input` lleva a $[-1,1]$; `EfficientNetB0` de Keras ya normaliza internamente y espera $[0,255]$ (en ese caso se quita `Rescaling(1/255)`). La entrada sigue siendo `uint8` $224\times224$.
16. **Selección del modelo final (D5).** El de mayor F1 macro en val (desempate: menor `val_loss`, luego menor tamaño). Copiarlo como `cnn_best.keras`, recargarlo con `keras.models.load_model` y repetir el paso 12 para confirmar las mismas cifras. Guardar accuracy, F1 macro y F1 por clase en `cnn_metricas_val.json`.
17. **Exportar predicciones de val para la comparación.** Con el modelo final recargado, escribir `MyDrive/sylfish/comparacion/pred_val_cnn.csv` con una fila por imagen de val: `filepath` (tomado del `.npz`), `label` (`y_val`), `pred` (`argmax`) y `conf` ($\max_k\hat{y}_k$). Verificar: tantas filas como val en `splits.csv` y `label` igual a la columna `label` de `splits.csv` al unir por `filepath`. Mismo formato que `pred_val_ml.csv` de ML clásico.
18. **Chequeo de atajos (D5).** Una celda define la neutralización del fondo; para las imágenes de val con máscara `ok`, construir la versión con fondo neutralizado: $X'=M\odot X+(1-M)\odot128$ (pez intacto sobre gris uniforme), redimensionando la máscara a 224 con `cv2.INTER_NEAREST` (las imágenes salen del caché; la máscara se lee de `/content/masks/`, descomprimida de `data/masks.zip`, porque el caché de 224 no la trae: es el único preprocesado propio de este cuaderno y se declara). Evaluar el modelo final en original y neutralizado sobre el mismo subconjunto y comparar accuracy y F1 macro. Revisar además si los errores se concentran en imágenes con marca de agua. Grad-CAM es opcional. Verificar: la caída de F1 macro está reportada en una tabla.
19. **Confianza.** La "confianza" de la demo es $\max_k\hat{y}_k$, no calibrada: las redes con entropía cruzada suelen ser sobreconfiadas. Opcional: temperature scaling, $\hat{y}=\mathrm{softmax}(z/T)$, con un único $T>0$ ajustado en val minimizando $\mathcal{L}_{CE}$. No hace falta reconstruir el modelo para obtener los logits: $\log\hat{y}_k=z_k-\log\sum_j e^{z_j}$ difiere de $z_k$ en una constante por muestra, y $\mathrm{softmax}(\log\hat{y}/T)=\mathrm{softmax}(z/T)$. No cambia el `argmax`, así que accuracy y F1 no cambian; solo las probabilidades. Si se usa, $T$ se guarda y la demo lo aplica.

### Parte B — Evaluación final en test (ejecutar una sola vez el jueves 12)

Esta sección va al final del cuaderno, separada por un título de markdown que dice "Evaluación final en test: ejecutar una sola vez el jueves 12", y se corre con el modelo congelado (miércoles 11, 23:59). Aquí no se decide nada: arquitectura, aumento, pesos, $T$ y $\tau$ ya se fijaron en val.

20. **Verificar el congelamiento y el aviso de una sola ejecución.** Leer `config_inferencia.json`, recalcular el MD5 de `cnn_best.keras` con `hashlib` y compararlo con el registrado. Comprobar además que `comparacion/pred_test_cnn.csv` NO existe: si existe, la evaluación ya se corrió y no se repite. Verificar: el MD5 coincide; si no, detenerse y aclarar en el equipo antes de seguir.
21. **Cargar test y verificar alineación.** Copiar `cnn_224_test.npz` a `/content` y abrirlo con `np.load`. Leer `splits.csv` con Pandas y filtrar `split == "test"`. Verificar en una celda explícita: número de filas $\approx 1362$ (el valor exacto sale de `splits.csv`); proporción por clase similar a la de train; `X.dtype == uint8`, forma $(n,224,224,3)$; y que, al unir por la clave `filepath` del `.npz`, el vector `y` coincide elemento a elemento con la columna `label` de la tabla (evita el error (c) de CIFAR-10).
22. **Pasada única del modelo.** `keras.models.load_model` con `compile=False` y comprobar `model.input_shape == (None,224,224,3)` y salida de 9 clases. `model.predict` sobre `X` de test con `batch_size` 64, sin dividir entre 255 (el modelo ya trae `Rescaling`) y sin aumento (las capas `Random*` están inactivas en inferencia). Guardar las probabilidades completas. Verificar: forma $(n,9)$, cada fila suma $\approx 1$, no hay NaN.
23. **Métricas en test.** Reutilizar la rutina del paso 12: `accuracy_score`, `f1_score` con `average="macro"`, `classification_report` con `target_names` de `classes.txt`, `digits=3` y `output_dict=True` para exportarlo, y `confusion_matrix` con `normalize="true"` graficada con `ConfusionMatrixDisplay`. Verificar: la suma de `support` es $n$; cada fila de la matriz suma 1.
24. **Confianza y regla de rechazo.** Histograma de $\max_k \hat y_k$ separado en aciertos y errores. Con el $\tau$ congelado en `config_inferencia.json` (y las probabilidades tras $T$, si se usó), reportar cobertura (fracción de test con $\max_k\hat y_k\ge\tau$) y accuracy dentro de esa fracción. Esto respalda lo que muestra la demo.
25. **Análisis de errores de clasificación.** Ordenar los pares fuera de la diagonal de la matriz sin normalizar y listar los 5 más frecuentes. Para rui, katla y mrigel, armar una grilla de imágenes mal clasificadas con clase real, clase predicha y confianza. Describir en 3 a 5 líneas qué se ve: forma del cuerpo, cabeza, aletas, color, iluminación, tamaño del pez en la foto. Revisar si los errores se concentran en imágenes con marca de agua o en las de baja confianza.
26. **Chequeo de atajos en test (opcional).** Repetir solo la predicción sobre las imágenes de test con máscara `ok`, con fondo neutralizado como en el paso 18. Reportar la caída de accuracy y F1 macro como medición secundaria; nunca usarla para cambiar nada.
27. **Bootstrap del F1 macro (opcional).** Con $B=1000$ remuestreos a nivel de `group_id` (unir `group_id` desde `splits.csv` por `filepath`), calcular el intervalo percentil al 95 %. Verificar: el F1 macro puntual cae dentro del intervalo y $B$ y el nivel quedan declarados en la tabla.
28. **Exportar.** Escribir `MyDrive/sylfish/comparacion/pred_test_cnn.csv` desde las probabilidades del paso 22: una fila por imagen de test con `filepath` (del `.npz`), `label`, `pred` (`argmax`) y `conf` ($\max_k\hat y_k$). Guardar tablas y figuras con los nombres de la sección 8 (`plt.savefig` con `dpi=200` y `bbox_inches="tight"`) y la fila DL de la tabla comparativa con las mismas columnas que usará ML clásico. Verificar: el número de filas de `pred_test_cnn.csv` es el de test en `splits.csv` y `label` coincide con el de `splits.csv` al unir por `filepath`.

## 5. Decisiones de diseño y alternativas

| Aspecto | CIFAR-10 de referencia | CNN SylFish | Por qué |
|---|---|---|---|
| Entrada | 32×32×3 | 224×224×3 `uint8` RGB | fotos de 500×500 con detalle de escamas y aletas; 224 es estándar y compatible con backbones |
| Normalización | fuera del modelo | `Rescaling(1/255)` dentro | un solo preprocesado para entrenamiento y demo; arrays en `uint8` ocupan ¼ de RAM ($\approx0.96$ GB train vs. $\approx3.8$ GB en `float32`) |
| Datos | `keras.datasets.cifar10` | `.npz` del contrato | misma partición para los cinco integrantes |
| Bloques conv | 3 (32/64/128) | 5 (32/64/128/256/256) | campo receptivo de 22 → 94 px; mapa final $7\times7$ |
| Regularización conv | `SpatialDropout2D(0.3)` | se conserva | — |
| Cabeza | GAP + Dense(256)+BN+Dropout(0.2) + softmax(10) | igual, softmax(9) | — |
| Aumento | Flip, Rotation(0.1), Zoom(0.1), RandomErasing | Flip H+V, Rotation, Zoom, Contrast; Erasing opcional | orientación arbitraria, distancia e iluminación variables |
| Validación en `fit` | conjunto de test | split val | test se usa una vez (D7) |
| Callbacks | `EarlyStopping` definido y no pasado | ES + 2× Checkpoint + ReduceLROnPlateau + CSVLogger en `callbacks` | corrige error (b); resiste desconexiones de Colab |
| Desbalance | no aplica (balanceado) | `class_weight` balanceado | razón máx/mín $\approx3.18$ |
| Optimizador | Adam, $\eta=0.002$ | Adam, $\eta=10^{-3}$ + reducción en meseta | más bloques y lotes de imágenes grandes; se reduce si se estanca |
| Evaluación | accuracy con slice desalineado | predicción completa alineada; accuracy, F1 macro, report, matriz normalizada | corrige error (c); métrica adecuada al desbalance |
| Imports | `plt` sin importar tras reinicio | celda única de imports | corrige error (d) |
| Estructura | un notebook, sin partición compartida | un cuaderno autocontenido; constantes en una celda de configuración, rutinas como celdas | sin módulos compartidos; la configuración se copia idéntica en cada cuaderno del repo |

Alternativas descartadas: `Flatten` en lugar de GAP (demasiados parámetros); sobremuestreo de clases minoritarias (duplica imágenes y favorece el sobreajuste; `class_weight` logra el efecto sin copiar datos); aumento de color fuerte (destruye una pista real de especie); seleccionar por accuracy (sesgada hacia rui, mrigel y telapia).

Decisiones de la evaluación en test:

- **Bootstrap percentil sobre test.** Remuestreo por `group_id`, $B=1000$, intervalo al 95 %. Hay que decirlo con precisión: mide la variabilidad debida a la muestra de test, no la debida al entrenamiento. Medir esta última exigiría reentrenar con varias semillas, lo que el cronograma no permite; se declara como limitación.
- **Todo fijo en test.** Modelo, $T$ (si se usó) y $\tau$ vienen de val. Cualquier ablación adicional en test se reporta como secundaria y no se usa para elegir.
- **Tamaño de test y desempeño por clase pequeña.** Con $\approx79$ imágenes de kalibaush, su F1 es muy incierto; se reporta con su soporte y no como valor exacto.

Decisión sugerida (confirmar con el equipo): el `ModelCheckpoint` por época sobrescribe un único archivo `cnn_last.keras` (para reanudar tras una desconexión) en lugar de guardar un archivo por época, para no llenar Drive.

Decisión sugerida (confirmar con el equipo): la salvaguarda de una sola ejecución es comprobar que `pred_test_cnn.csv` no exista antes de predecir en test; si hay un error técnico (por ejemplo, desalineación) se corrige, se documenta en el cuaderno y se borra el archivo de forma consciente, sin tocar modelo ni umbrales.

## 6. Errores comunes y trampas

- Pasar `(X_test, y_test)` a `validation_data`: invalida el test del D7.
- Abrir `cnn_224_test.npz` o medir en test antes del jueves 12 "para ver cómo va": es selección en test.
- Correr la celda de predicción en test dos veces "porque algo falló" y quedarse con el mejor resultado. Si hay un error técnico, se corrige el error, se documenta y se reporta el resultado corregido, sin tocar modelo ni umbrales.
- Definir callbacks y no incluirlos en `callbacks=[...]`.
- Barajar o recortar val (o test) antes de predecir y comparar con `y` en otro orden; filtrar o reordenar una tabla y no la otra.
- Normalizar dos veces: dividir por 255 fuera y además tener `Rescaling` dentro (el modelo ve imágenes casi negras y la accuracy cae a nivel de azar). Con transfer learning, usar `Rescaling(1/255)` en vez del `preprocess_input` del backbone.
- Calcular `class_weight` con todo el dataset o con val: se calcula solo con `y_train`.
- Leer imágenes con OpenCV en el chequeo de atajos sin pasar de BGR a RGB.
- Comparar experimentos con particiones, semillas o presupuestos de épocas distintos.
- Concluir "el modelo es robusto" sin el chequeo de atajos: el fondo claro uniforme y la marca de agua son atajos posibles.
- Interpretar el F1 de kalibaush ($\approx79$ imágenes en test) como exacto.
- Ajustar la temperatura o $\tau$, o elegir el modelo, mirando test.
- Reentrenar después del miércoles 11 a las 23:59.

## 7. Criterio de aceptación

- [ ] El cuaderno corre de principio a fin con "Reiniciar y ejecutar todo" sin errores (la sección de test se ejecuta solo el jueves 12).
- [ ] No hay referencias a módulos `.py` externos: las constantes están en la celda de configuración y las rutinas son celdas.
- [ ] `validation_data` usa val; test solo aparece en la última sección, separada y rotulada.
- [ ] Los callbacks están en la llamada a `fit` y el historial quedó en CSV.
- [ ] `len(y_pred) == len(y_val)` y las métricas se calculan con todo val.
- [ ] Reportadas en val: accuracy, F1 macro, `classification_report` y matriz normalizada por filas.
- [ ] Tabla de experimentos con al menos línea base y dos mejoras.
- [ ] `cnn_best.keras` recargado reproduce las métricas de val.
- [ ] Chequeo de atajos con fondo neutralizado reportado.
- [ ] `comparacion/pred_val_cnn.csv` exportado con las columnas del contrato.
- [ ] Entrada del modelo: `uint8` RGB $224\times224$, sin normalización externa.
- [ ] MD5 de `cnn_best.keras` igual al registrado al congelar.
- [ ] La celda de predicción en test se ejecutó una sola vez y su salida quedó guardada.
- [ ] Tamaño de test y alineación `y` ↔ `label` verificados en una celda explícita.
- [ ] Test: accuracy, F1 macro, `classification_report` y matriz normalizada exportados; cobertura y accuracy con $\tau$ reportadas.
- [ ] Top-5 confusiones listadas y grilla de errores en carpas comentada.
- [ ] `comparacion/pred_test_cnn.csv` exportado con las columnas del contrato.
- [ ] (Opcional) Intervalo bootstrap del F1 macro con $B$ y nivel declarados.

## 8. Artefactos que debe dejar

Parte A (val):

- `MyDrive/sylfish/models/cnn_best.keras` y `MyDrive/sylfish/models/cnn_last.keras`.
- `reports/tables/cnn_historial_<experimento>.csv` (salida de `CSVLogger`).
- `reports/tables/cnn_experimentos.csv` (id, cambio, accuracy val, F1 macro val, época del mejor).
- `reports/tables/cnn_metricas_val.json` del modelo final (accuracy, F1 macro, F1 por clase; y $T$ si se usa temperature scaling).
- `reports/tables/cnn_atajos_val.csv` (original vs. fondo neutralizado).
- `MyDrive/sylfish/comparacion/pred_val_cnn.csv` (paso 17).
- `reports/figures/cnn_curvas.png`, `reports/figures/cnn_matriz_val.png`, `reports/figures/cnn_aumento.png`, `reports/figures/cnn_atajos_ejemplos.png`.

Parte B (test, jueves 12):

- `MyDrive/sylfish/comparacion/pred_test_cnn.csv` (paso 28).
- `reports/tables/cnn_predicciones_test.csv`: una fila por imagen con `filepath`, `clase`, `label`, `pred`, `conf_top1`, top-3.
- `reports/tables/cnn_report_test.csv`: precisión, recall, F1 y soporte por clase, más macro y ponderado.
- `reports/tables/cnn_resumen_test.csv`: accuracy, F1 macro (con intervalo si se calculó), cobertura y accuracy con $\tau$, y $n$; es la fila DL de la tabla comparativa.
- `reports/tables/cnn_atajos_test.csv` (opcional).
- `reports/figures/cnn_matriz_test.png`, `reports/figures/cnn_errores_carpas_test.png`, `reports/figures/cnn_confianza_test.png`.
- (Opcional) `reports/figures/cnn_bootstrap_f1_test.png`.

## 9. Preguntas probables de sustentación

1. **¿Qué tomaron del notebook CIFAR-10 y qué cambiaron?** Conservamos el esqueleto Conv–BN–ReLU–Pool, SpatialDropout, GAP y la cabeza densa. Cambiamos resolución, número de bloques, normalización interna, datos, validación con val, callbacks efectivos, evaluación alineada, `class_weight` y métricas.
2. **¿Por qué 5 bloques y cuántos parámetros tiene la primera convolución?** A $224\times224$, con 3 bloques cada neurona final "ve" solo 22 píxeles; con 5 ve 94 y el mapa final es $7\times7$, adecuado para el GAP. La primera convolución tiene $(3\cdot3\cdot3+1)\cdot32=896$ parámetros; en general $(k_hk_wC_{in}+1)C_{out}$.
3. **¿Para qué sirve BatchNormalization?** Normaliza cada canal con la media y varianza del lote y luego reescala con $\gamma,\beta$ aprendidos. Estabiliza y acelera el entrenamiento. En inferencia usa estadísticas móviles.
4. **¿Por qué F1 macro y no solo accuracy, y qué hace `class_weight`?** Las clases van de 525 a 1670 imágenes; la accuracy puede ser alta ignorando una clase pequeña y la F1 macro promedia por clase y lo penaliza. `class_weight` multiplica la pérdida de cada muestra por $w_c=\frac{N}{KN_c}$: los errores en clases pequeñas pesan más.
5. **¿Por qué no validaron con test y por qué lo evaluaron una sola vez?** Val se usa para early stopping y selección de modelo; si fuera test, la cifra final estaría optimista. Además, elegir entre varias evaluaciones en test sesga hacia arriba el resultado, porque el máximo de estimaciones ruidosas sobreestima el desempeño real. Test se usó una sola vez, el jueves 12, con el modelo congelado.
6. **¿El porcentaje de la demo es una probabilidad real?** Es el máximo del softmax, no calibrado. Indica preferencia relativa; las CNN suelen ser sobreconfiadas. Temperature scaling lo corrige sin cambiar la clase predicha.
7. **¿Cómo saben que la red mira el pez y no el fondo?** Evaluamos con el fondo reemplazado por gris usando las máscaras. Si el desempeño casi no cae, la decisión depende del pez; si cae mucho, hay dependencia del fondo y lo declaramos como limitación.
8. **¿Por qué no aumentaron el color?** El color es una característica real de varias especies; alterarlo destruye información útil.
9. **¿Cómo se lee la matriz normalizada y por qué se confunden rui, katla y mrigel?** Cada fila es una clase real y suma 1; la diagonal es el recall y cada celda fuera de ella es la fracción de esa especie confundida con otra. Las tres son carpas de la misma familia, con cuerpo, escamas y color parecidos; las diferencias están en proporciones de cabeza, boca y aletas, que se pierden a $224\times224$ o con el pez pequeño en la foto.
10. **¿Qué tan seguros están de las cifras de test?** Con $n\approx1362$ y una accuracy cercana a 0.90, el margen aproximado es de $\pm1.6$ puntos; para una clase pequeña como kalibaush, cerca de $\pm8$ (recalcular con las cifras reales). El intervalo bootstrap cuantifica la variación por muestra de test, no por reentrenamiento.
