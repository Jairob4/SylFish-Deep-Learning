# Guía de reportes: figuras y tablas

**Responsable:** el dueño de cada cuaderno produce sus figuras y tablas; Jairo consolida la lista el D8 · **Depende de:** los cuatro cuadernos de `notebooks/` (`01_preparacion_datos`, `02_cnn`, `03_unet`, `04_demo`) · **Entrega a:** informe, presentaciones y banco de preguntas de sustentación · **Fecha objetivo:** cada figura el día de su notebook; revisión completa el D8, viernes 13 de noviembre.

## De qué se trata

Todo número o imagen que aparezca en el informe o en la presentación sale de un archivo de esta carpeta, generado por un notebook. Nada se copia a mano desde una salida de celda ni se rehace en PowerPoint: si el modelo cambia, se vuelve a ejecutar la celda y la figura se actualiza. Las salidas de los notebooks se borran con `nbstripout` antes del commit, así que esta carpeta es la evidencia versionada de los resultados.

## Estructura

- `reports/figures/`: PNG.
- `reports/tables/`: TODAS las tablas de resultados (contrato): CSV para el informe, registros de entrenamiento y métricas en val de `02_cnn` y `03_unet` (historiales de `CSVLogger`, registros de experimentos, JSON de métricas). La carpeta existe en el repo con un `.gitkeep`. Nada se guarda suelto en la raíz de `reports/`.

Prefijos: el prefijo del nombre identifica el cuaderno que produce el archivo, y esa es la única regla de nombrado. Los nombres de archivo son el contrato con el informe y no cambian. Sufijo `_val` o `_test` siempre que la figura o la tabla dependa de un split.

| Prefijo | Cuaderno que lo produce | Dueño |
|---|---|---|
| `01_` | `01_preparacion_datos` (exploración, partición, máscaras) | Jairo |
| `cnn_` | `02_cnn` (desarrollo en val y celda de test) | Compañero DL |
| `unet_` | `03_unet` (desarrollo en val y celda de test) | Jairo |
| `04_` | `04_demo` (demo y evaluación del sistema completo) | Jairo, con apoyo del Compañero DL |

Los cuadernos 02 y 03 usan `cnn_` y `unet_` en lugar de `02_` y `03_`, de modo que el nombre diga qué modelo produjo el archivo. Un archivo sin uno de estos cuatro prefijos no pertenece a esta lista.

## Formato de las figuras

1. Guardar con `plt.savefig` usando `dpi=200` y `bbox_inches="tight"`, y cerrar con `plt.close` (si no, las figuras se acumulan en memoria en notebooks largos).
2. Título, ejes con nombre y unidades, y leyenda si hay más de una serie, todo en español.
3. Nombres de clases como en `classes.txt` y en ese orden en todos los ejes y leyendas.
4. Un color fijo por split en todas las figuras (por ejemplo train, val y test siempre con los mismos tres colores) y un único mapa de color para matrices de confusión.
5. Matrices de confusión normalizadas por filas, con valores con 2 decimales en cada celda.
6. Grillas de imágenes: imágenes en RGB (una figura con colores azulados delata que se mostró BGR), con la clase en el título de cada panel.
7. Ninguna figura ni tabla `_test` se genera antes del D7.

## Lista de figuras

### `01_` — Preparación de datos (`01_preparacion_datos`, Jairo)

Exploración y duplicados (D1):

- `figures/01_conteos_por_clase.png`: barras de imágenes por clase, con el número sobre cada barra.
- `figures/01_grilla_muestras.png`: grilla de muestras aleatorias por clase (`SEED = 42`).
- `figures/01_fondo_por_clase.png`: diagramas de caja de brillo y saturación del fondo por clase (riesgo de que el fondo prediga la clase).
- `figures/01_marca_agua_ejemplos.png`: ejemplos de imágenes con marca de agua.
- `figures/01_hist_dhash.png`: histograma de distancias de Hamming entre pares cercanos.
- `figures/01_pares_duplicados.png`: pares duplicados o casi duplicados lado a lado.

Partición (D1):

- `figures/01_proporciones_por_split.png`: proporción de cada clase en train, val y test.
- `figures/01_tamanos_de_grupo.png`: distribución del tamaño de los grupos de duplicados.

Máscaras OpenCV (D2–D3):

- `figures/01_comparacion_canales.png`: canales gris, $S$ de HSV, $L^*$, $a^*$, $b^*$ y croma para una muestra por clase, con la tabla de $\eta$.
- `figures/01_pipeline_etapas.png`: una imagen a través de cada etapa (canal, desenfoque, Otsu, apertura, cierre, componente mayor, relleno, superposición).
- `figures/01_hist_area_frac.png`: histograma de `area_frac` con las líneas de alerta en 0.05 y 0.80.
- `figures/01_grilla_qc_<clase>.png`: una grilla de QC por clase (9 archivos).
- `figures/01_casos_descartados.png`: máscaras descartadas con el motivo.

### `cnn_` — CNN (`02_cnn`, Compañero DL)

Desarrollo en val (D2–D5):

- `figures/cnn_aumento.png`: una imagen con varias versiones aumentadas.
- `figures/cnn_curvas.png`: pérdida y accuracy de train y val por época, del modelo final (y de la línea base si cabe en la misma figura).
- `figures/cnn_matriz_val.png`: matriz de confusión normalizada en val del modelo final.
- `figures/cnn_atajos_ejemplos.png`: imágenes originales y con fondo neutralizado, con sus predicciones.

Celda de test (D7):

- `figures/cnn_matriz_test.png`: matriz de confusión normalizada por filas en test.
- `figures/cnn_errores_carpas_test.png`: errores entre rui, katla y mrigel con clase real, predicha y confianza.
- `figures/cnn_confianza_test.png`: histograma de $\max_k\hat p_k$ separado en aciertos y errores.
- `figures/cnn_bootstrap_f1_test.png` (opcional): distribución bootstrap del F1 macro.

### `unet_` — U-Net (`03_unet`, Jairo)

Desarrollo en val (D4–D5):

- `figures/unet_aumento_sincronizado.png`: pares imagen–máscara aumentados, mostrando que la geometría coincide.
- `figures/unet_curvas.png`: pérdida, IoU y Dice de train y val por época.
- `figures/unet_umbral_val.png`: IoU medio en val en función del umbral.
- `figures/unet_hist_iou_val.png`: histograma de IoU por imagen en val.
- `figures/unet_peores10_val.png`: los 10 peores casos de val (imagen, referencia, predicción, overlay).
- `figures/unet_ejemplos_val.png`: casos típicos bien segmentados.

Celda de test (D7):

- `figures/unet_hist_iou_test.png`: histograma de IoU por imagen en test.
- `figures/unet_iou_por_clase_test.png`: diagramas de caja de IoU por clase de pez.
- `figures/unet_peores10_test.png`: 10 peores casos de test.

### `04_` — Demo y sistema completo (`04_demo`, Jairo con apoyo del Compañero DL)

Demo (D6 y D8):

- `figures/04_paridad_rgb_bgr.png`: misma imagen predicha con RGB y con BGR (respuesta lista para la sustentación).
- `figures/04_cobertura_vs_accuracy_val.png`: curva para elegir $\tau$ en val.
- `figures/04_demo_captura.png`: captura de la interfaz para el informe.
- `figures/04_demo_respaldo_grilla.png`: grilla con los 14 casos del ensayo, como respaldo si falla la conexión.

Evaluación del sistema completo (D7):

- `figures/04_sistema_tabla2x2_test.png`: tabla 2×2 clase correcta/incorrecta por $\mathrm{IoU}\ge0.5$ o $<0.5$ (o la tabla directamente en el informe).
- `figures/04_cnn_errores_carpas_test.png` y `figures/04_cnn_confianza_aciertos_errores_test.png`: los mismos análisis de la CNN rehechos en el cuaderno 04 con la función de la demo; deben coincidir con `cnn_errores_carpas_test.png` y `cnn_confianza_test.png`.
- `figures/04_unet_peores10_test.png`: 10 peores casos de segmentación, en la evaluación del sistema.
- `figures/04_bootstrap_f1_test.png` (opcional): distribución bootstrap del F1 macro.

## Lista de tablas

Todas en `reports/tables/`. Cada archivo lo produce una sola celda de un solo cuaderno, el que indica su prefijo.

### `01_` — Cifras de datos (`01_preparacion_datos`)

- `tables/01_resoluciones.csv`: tamaños y modos de imagen únicos con su conteo.
- `tables/01_marca_agua_por_clase.csv`: imágenes con marca de agua por clase y método usado.
- `tables/01_duplicados_resumen.csv`: pares y grupos por MD5 y por dHash, y grupos que mezclan clases.
- `tables/01_conteos_por_split.csv`: conteo y proporción por clase y split, número de grupos y exclusiones.
- `tables/01_eta_canales.csv`: separabilidad $\eta$ de cada canal de color por clase.
- `tables/01_estados_por_clase.csv`: máscaras ok, revisar y descartada por clase y por split.
- `tables/01_auditoria_ok.csv`: fallos visibles y tasa $\hat p$ de falsos ok por clase.
- `tables/01_iou_oro.csv` (opcional): IoU y Dice de las pseudo-máscaras contra el conjunto oro, global y por clase.

### Desarrollo en val (`02_cnn` y `03_unet`)

- `tables/cnn_historial_<experimento>.csv` y `tables/unet_historial_<experimento>.csv`: salida de `CSVLogger` por experimento.
- `tables/cnn_experimentos.csv` y `tables/unet_experimentos.csv`: una fila por experimento (ver [CONVENCIONES.md](../docs/CONVENCIONES.md), sección 11).
- `tables/cnn_metricas_val.json`: accuracy, F1 macro y F1 por clase del modelo final en val; $T$ si se usó temperature scaling.
- `tables/cnn_atajos_val.csv`: accuracy y F1 macro en val, original frente a fondo neutralizado.
- `tables/unet_metricas_val.json`: IoU y Dice medios en val, umbral $t$ elegido y si se usa postprocesado.
- `tables/unet_por_clase_val.csv`, `tables/unet_umbral_val.csv`, `tables/unet_excluidas.csv`.

### Evaluación en test, D7

Celda de test de `02_cnn`:

- `tables/cnn_predicciones_test.csv`: una fila por imagen de test con `filepath`, `clase`, `label`, `pred`, `conf_top1` y top-3.
- `tables/cnn_report_test.csv`: precisión, recall, F1 y soporte por clase, más promedios macro y ponderado.
- `tables/cnn_resumen_test.csv`: accuracy, F1 macro (con intervalo si se calculó), cobertura y accuracy con $\tau$, y $n$; es la fila de la CNN en la tabla comparativa.
- `tables/cnn_atajos_test.csv` (opcional).

Celda de test de `03_unet`:

- `tables/unet_iou_test.csv`: una fila por imagen con `filepath`, `clase`, `label`, `estado`, `iou`, `dice`.
- `tables/unet_metricas_test.json`: IoU y Dice medios y medianas, intervalo, $n$ y $n_{ok}$, umbral y postprocesado usados.
- `tables/unet_por_clase_test.csv`: IoU y Dice medios, mediana y $n$ por clase.
- `tables/unet_errores_segmentacion_test.csv`: 10 peores casos con su categoría de error.

Evaluación del sistema completo en `04_demo`:

- `tables/04_predicciones_sistema_test.csv`: una fila por imagen de test con `filepath`, `clase`, `label`, `pred`, `conf_top1`, `estado`, `iou`, `dice`, `sistema_ok`.
- `tables/04_resumen_final_test.csv`: la tabla principal del informe (accuracy, F1 macro, IoU medio, Dice medio, $S$, cobertura y accuracy con $\tau$, $n$ y $n_{ok}$). Su fila DL usa las mismas columnas que la del equipo ML clásico para la tabla comparativa.
- `tables/04_errores_segmentacion.csv`: 10 peores casos con su categoría de error, vistos desde el sistema completo.
- `tables/04_ensayo_demo.csv`: las 14 imágenes del ensayo (10 de test, 3 externas, 1 que no es pez) con origen, clase real, top-1, confianza, área, mensaje de rechazo, acierto y explicación preparada (D8).

## Qué va al informe y qué a la presentación

| Pieza | Informe | Presentación |
|---|---|---|
| Datos y desbalance | `01_conteos_por_clase`, `01_grilla_muestras`, `01_proporciones_por_split`, `01_conteos_por_split.csv` | `01_grilla_muestras` |
| Máscaras | `01_pipeline_etapas`, `01_hist_area_frac`, `01_estados_por_clase.csv` | `01_pipeline_etapas` |
| CNN | `cnn_curvas`, `cnn_matriz_val`, `cnn_atajos_val.csv`, `cnn_matriz_test`, `cnn_resumen_test.csv` | `cnn_matriz_test` |
| U-Net | `unet_curvas`, `unet_umbral_val`, `unet_iou_por_clase_test`, `unet_peores10_test`, `unet_metricas_test.json` | `unet_peores10_test` |
| Sistema | `04_resumen_final_test.csv`, `04_sistema_tabla2x2_test` | `04_resumen_final_test.csv` |
| Demo | `04_demo_captura`, `04_paridad_rgb_bgr` | demo en vivo y `04_demo_respaldo_grilla` |

Para la CNN y la U-Net sueltas se citan las figuras `cnn_` y `unet_`; las `04_` repetidas (errores en carpas, confianza, peores 10) solo se citan si el texto habla del sistema completo.

## Errores comunes y trampas

- Figuras de val presentadas como resultados finales, o figuras de test generadas antes del D7.
- Matriz de confusión sin normalizar con clases desbalanceadas: rui domina visualmente.
- Reportar el IoU agregado de `BinaryIoU` en lugar del IoU medio por imagen.
- Archivos regenerados en `/content/sylfish-dl/reports/` que no llegaron a commit antes de que Colab desconectara.
- Figuras con nombres distintos a esta lista: el informe enlaza nombres exactos.

## Criterio de aceptación

- [ ] Todas las figuras obligatorias de esta lista existen en `reports/figures/` en `main` al cierre del D8.
- [ ] `tables/04_resumen_final_test.csv` coincide con la tabla del informe, y `tables/cnn_resumen_test.csv` y `tables/unet_metricas_test.json` coinciden con sus filas.
- [ ] Ninguna cifra del informe ni de la presentación está escrita a mano sin un archivo de origen aquí.
- [ ] Todas las figuras están en RGB, con títulos y ejes en español y el orden de clases del contrato.
