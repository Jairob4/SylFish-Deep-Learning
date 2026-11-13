# Guía 04 — Demo con Gradio y evaluación del sistema completo

**Responsable:** Jairo (líder) y Compañero DL · **Depende de:** 01 (`splits.csv`, `mask_qc.csv`, `cnn_224_{val,test}.npz`, `unet_256_{val,test}.npz`), 02 (`cnn_best.keras`, `reports/tables/cnn_metricas_val.json` con $T$ si se usó, y, solo para el paso 17, `pred_test_cnn.csv`), 03 (`unet_best.keras`, `reports/tables/unet_metricas_val.json` con el umbral de máscara $t$ y la decisión de postprocesado; para el paso 17, `reports/tables/unet_iou_test.csv`). Orden del D7: celda de test de 02, celda de test de 03 y, solo después, los pasos 14 a 22 de este cuaderno · **Entrega a:** sustentación y presentaciones, informe (capturas y tabla final), guía de comparación DL vs ML (`SylFish_Guia_Equipo/04_comparacion/`), banco de preguntas · **Fecha objetivo:** D6, miércoles 11 de noviembre de 2026 (demo y `config_inferencia.json`, antes del congelamiento de las 23:59); D7, jueves 12 (evaluación del sistema completo en test); D8, viernes 13 (ensayo)

## 1. De qué se trata

Este cuaderno es el único punto donde la CNN y la U-Net trabajan juntas (diseño A: modelos independientes que se unen en la inferencia). Hace dos cosas con el mismo par de modelos congelados. Primero, la demo: recibe una imagen cualquiera y devuelve la especie predicha con su top-3 de confianzas, la máscara del pez superpuesta a la imagen y el área segmentada como porcentaje de la imagen. Corre en Colab con Gradio y se comparte con `share=True`. Segundo, la evaluación del sistema completo sobre test: se mide cuántas imágenes quedan bien resueltas tal como las vería el usuario de la demo, es decir, con la especie correcta y una máscara con $\mathrm{IoU}\ge 0.5$ a la vez, y se arma la tabla final para el informe.

Lo que se evalúa en la sustentación no es que la demo "se vea bien", sino que el equipo sepa explicar cada transformación que sufre la imagen, por qué la confianza no es una probabilidad calibrada y qué hace el sistema cuando la imagen no es un pez. Esta guía también define el plan de contingencia y el ensayo con 14 imágenes. Todo lo que se decidió (arquitecturas, umbral de máscara, postprocesado, $\tau$) se decide en val y queda congelado en `config_inferencia.json`; el jueves 12 solo se mide.

No existe ningún módulo `.py` compartido: la función de predicción y el preprocesado viven en celdas de este cuaderno. Como el preprocesado de 224 y de 256 se escribió originalmente en el cuaderno 01 (para generar el caché) y aquí se reescribe, la prueba de paridad contra el caché `.npz` es obligatoria (paso 9).

## 2. Entradas y salidas (contrato)

**Entradas:**

| Elemento | Ruta / formato |
|---|---|
| Modelos | `MyDrive/sylfish/models/cnn_best.keras`, `MyDrive/sylfish/models/unet_best.keras` |
| Clases | `classes.txt` (orden 0–8) |
| Partición y calidad de máscaras | `data/splits/splits.csv` y `data/masks/mask_qc.csv` (columna `estado`) |
| Caché de val y test | `data/processed/cnn_224_{val,test}.npz` (`X` `uint8` 224×224×3 RGB, `y`, `filepath`) y `data/processed/unet_256_{val,test}.npz` (`X` `uint8` 256×256×3 RGB, `M` en $\{0,1\}$, `y`, `filepath`), generados por el cuaderno 01 |
| Predicciones de la CNN en test | `MyDrive/sylfish/comparacion/pred_test_cnn.csv` (`filepath`, `label`, `pred`, `conf`), generado por el cuaderno 02; solo para la verificación cruzada del paso 17 |
| Ejemplos | imágenes del split test listadas en `data/splits/splits.csv`, leídas desde `/content` (zip descomprimido) |
| Configuración | `MyDrive/sylfish/models/config_inferencia.json` (contrato v1.1): JSON con umbral de máscara $t$ (el de 03), $\tau$, temperatura $T$ (1 si no se usa), área mínima (1 %), postprocesado sí/no, interpolación usada en cada `.npz` y MD5 de cada modelo. **Lo escribe este cuaderno al congelar** (paso 10, D6) y lo leen la demo, el paso 14 antes de evaluar y las secciones de test de 02 y 03 |
| Entrada del usuario | imagen de `gr.Image` con `type="numpy"` e `image_mode="RGB"`: arreglo `uint8` $(H,W,3)$ **en RGB** |

**Salidas de la función de predicción:** un diccionario `{clase: probabilidad}` para `gr.Label` con `num_top_classes=3`; una imagen RGB `uint8` $(H,W,3)$ con el overlay a la resolución original; un texto con el área % y, si aplica, el mensaje de rechazo.

**Salidas de la evaluación del sistema:** las tablas de `reports/tables/` y las figuras de `reports/figures/` de la sección 8, incluida la fila DL de la tabla final para el informe y la comparación con ML clásico.

## 3. Fundamento teórico para la sustentación

**Confianza softmax.** La CNN produce logits $z\in\mathbb{R}^9$ y
$$\mathrm{softmax}(z)_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}.$$
El "porcentaje" mostrado es $\max_k \hat p_k$. No está calibrado: una red entrenada con entropía cruzada suele ser sobreconfiada, es decir, cuando dice 0.95 acierta menos del 95 % de las veces. Una corrección opcional es temperature scaling, $\hat p_k(T)=e^{z_k/T}/\sum_j e^{z_j/T}$ con $T$ ajustado en val; no cambia la clase predicha, solo suaviza la confianza.

**Mundo cerrado.** Las nueve probabilidades suman 1 por construcción, así que softmax siempre reparte toda la masa entre las 9 especies conocidas. Una foto de un zapato recibe una "especie" y, con frecuencia, una confianza alta, porque la red nunca vio ejemplos de "no es un pez" y no existe una salida para eso. Las reglas de rechazo son heurísticas que mitigan el problema, no una solución.

**Reglas de rechazo.**
$$\max_k \hat p_k<\tau\ \Rightarrow\ \text{"baja confianza"},\qquad a=\frac{100}{HW}\sum_{u,v} m(u,v)<1\ \Rightarrow\ \text{"no se detecta un pez"}.$$
$\tau$ (por ejemplo 0.5) se elige en val mirando la curva cobertura–accuracy: fracción de imágenes con $\max_k\hat p_k\ge\tau$ contra accuracy dentro de esa fracción. Alternativa de puntaje: entropía $H(\hat p)=-\sum_k \hat p_k\log\hat p_k$ (alta = indecisión).

**Reescalado de la máscara con vecino más cercano.** Para llevar la máscara $m$ de $256\times256$ a $H\times W$, cada píxel destino copia el valor del píxel fuente más cercano, $m'(u,v)=m\big(\lfloor u\cdot 256/H\rfloor,\lfloor v\cdot 256/W\rfloor\big)$, por lo que la salida sigue siendo binaria. La interpolación bilineal promedia vecinos y produce valores intermedios en el borde que ya no son "pez" ni "fondo".

**Overlay.** Con color $c$ y opacidad $\alpha$ (por ejemplo 0.4):
$$O(u,v)=\begin{cases}(1-\alpha)\,I(u,v)+\alpha\,c, & m'(u,v)=1\\ I(u,v), & m'(u,v)=0\end{cases}$$

**Por qué test se usa una sola vez.** Cada métrica medida en test es una variable aleatoria: depende de qué imágenes cayeron en test. Si se evalúan varios modelos (o varios umbrales) en test y se reporta el mejor, el número reportado está sesgado hacia arriba, porque el máximo de estimaciones ruidosas sobreestima el máximo de los valores verdaderos:
$$\mathbb{E}\Big[\max_m \widehat{\mathrm{Acc}}_m\Big]\ \ge\ \max_m \mathbb{E}\Big[\widehat{\mathrm{Acc}}_m\Big].$$
Elegir en test convierte test en un segundo val y deja al proyecto sin una estimación honesta del desempeño en datos nuevos. Por eso $\tau$, $t$ y el postprocesado se fijan en val, y esta evaluación solo mide.

**Métricas por imagen.** Con $A$ la máscara predicha binarizada y $B$ la pseudo-máscara de referencia:
$$\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|},\qquad \mathrm{Dice}(A,B)=\frac{2|A\cap B|}{|A|+|B|},\qquad \mathrm{Dice}=\frac{2\,\mathrm{IoU}}{1+\mathrm{IoU}}.$$
La última identidad explica por qué Dice siempre sale más alto que IoU: es una transformación monótona, no información distinta. Se promedia por imagen (cada pez cuenta igual), no por píxel acumulado. Para la clasificación, con $TP_k$, $FP_k$, $FN_k$ por clase:
$$P_k=\frac{TP_k}{TP_k+FP_k},\qquad R_k=\frac{TP_k}{TP_k+FN_k},\qquad F1_k=\frac{2P_kR_k}{P_k+R_k},\qquad F1_{\text{macro}}=\frac{1}{K}\sum_{k=1}^{K}F1_k.$$

**Métrica del sistema completo.** Sobre el conjunto $\mathcal{T}_{ok}$ de imágenes de test con máscara de referencia en estado ok:
$$S=\frac{1}{|\mathcal{T}_{ok}|}\sum_{i\in\mathcal{T}_{ok}}\mathbb{1}\big[\hat y_i=y_i\big]\cdot\mathbb{1}\big[\mathrm{IoU}_i\ge 0.5\big].$$
Si los errores de ambos modelos fueran independientes, $S\approx \mathrm{Acc}\cdot P(\mathrm{IoU}\ge 0.5)$. Si $S$ es mayor que ese producto, los dos modelos fallan en las mismas imágenes difíciles (errores correlacionados).

**Incertidumbre.** Con $n$ imágenes, el error estándar de una accuracy $\hat p$ es $\sqrt{\hat p(1-\hat p)/n}$. Con $n\approx 1362$ y $\hat p=0.90$, el intervalo aproximado al 95 % es $\pm 1.96\sqrt{0.09/1362}\approx\pm 0.016$. Para el recall de kalibaush ($n\approx 79$, $\hat p=0.85$) es $\approx\pm 0.079$. Diferencias de 1–2 puntos entre modelos no son concluyentes y el desempeño por clase pequeña es muy incierto.

## 4. Pipeline paso a paso

Los pasos 1 a 13 se ejecutan el D6, con imágenes de val; el paso 10 escribe `config_inferencia.json` antes de las 23:59 del miércoles 11, es decir, antes de que 02 y 03 lo lean para sus celdas de test. Los pasos 14 a 22 (evaluación del sistema completo) se ejecutan **una sola vez** el D7, con los modelos congelados, en celdas separadas al final del cuaderno y después de que 02 y 03 hayan corrido sus celdas de test (el paso 17 las contrasta). El paso 23 (ensayo y respaldo) se completa el D8, porque usa los errores reales que salen del análisis del D7.

1. **Encabezado, imports y configuración.** Docstring con Autor, Código, Materia, Tema; celda única de imports con comentario en línea (Gradio, Keras, NumPy, OpenCV, Pandas, scikit-learn, Matplotlib, `json`, `hashlib`; incluir `matplotlib.pyplot` aunque parezca obvio: error (d) del cuaderno CIFAR-10). Instalar Gradio en una celda previa si el entorno no lo trae. Celda de configuración con las constantes comunes copiadas idénticas de los demás cuadernos: `SEED = 42` (fijada con `random`, NumPy y `keras.utils.set_random_seed`), tamaños 224 y 256, rutas del contrato y orden de clases. Verificar: versiones impresas y anotadas en el README; versión de TensorFlow/Keras igual a la de entrenamiento.
2. **Preparar datos y modelos.** Montar Drive, descomprimir el zip en `/content`, cargar ambos modelos **una sola vez, fuera de la función** con `keras.models.load_model` y `compile=False` (la U-Net tiene pérdida y métricas propias y fallaría al deserializarlas). Leer `classes.txt` y, si ya existe, la configuración. Verificar: formas de entrada $(None,224,224,3)$ y $(None,256,256,3)$; salidas de 9 y de 1 canal; 9 nombres de clase en el orden del contrato.
3. **Normalizar la entrada.** Primer bloque de la función de predicción: si la imagen es `None`, devolver mensaje; si tiene 4 canales (PNG con transparencia), quitar alfa; si es de un canal, replicarlo a 3; forzar `uint8`. Guardar $H$ y $W$ originales. **Trampa RGB/BGR:** Gradio entrega RGB; OpenCV lee BGR. Dentro de la demo **no** se aplica `cv2.COLOR_BGR2RGB` a la entrada de Gradio (ya es RGB); solo se convierte si una imagen se lee con `cv2.imread` (por ejemplo, los ejemplos o la prueba de paridad).
4. **Rama CNN (celda de preprocesado y predicción).** `cv2.resize` a 224×224 con la **misma** interpolación usada al crear `cnn_224_*.npz` (`cv2.INTER_AREA`, la que fijó el cuaderno 01; anotarla en la celda de configuración; ojo, `dsize` va como ancho, alto); mantener `uint8`, **sin** dividir entre 255 (el modelo trae `Rescaling(1/255)`); agregar dimensión de lote con `np.expand_dims`; `predict`. Si el modelo final es de transfer learning, su preprocesado propio debe estar dentro del modelo o replicarse idéntico. Verificar: vector de 9 que suma $\approx1$.
5. **Top-3 y confianza.** Construir el diccionario de las 9 clases con sus probabilidades; `gr.Label` con `num_top_classes=3` muestra las tres mayores. Aplicar la regla de $\tau$.
6. **Rama U-Net (celda de preprocesado y predicción).** Redimensionar a 256×256 con la interpolación de `unet_256_*.npz`; aplicar la misma normalización que en el entrenamiento del cuaderno 03 (si la U-Net no trae `Rescaling` dentro, dividir entre 255 aquí, exactamente como allá); `predict` → mapa $(1,256,256,1)$; binarizar con el umbral congelado; componente conexa más grande (`cv2.connectedComponentsWithStats`) solo si así se congeló.
7. **Reescalar la máscara.** `cv2.resize` de la máscara binaria a $(W,H)$ con `cv2.INTER_NEAREST`. Verificar: forma $(H,W)$ y valores únicos $\{0,1\}$.
8. **Overlay y área.** Mezcla semitransparente según la fórmula de la sección 3 (por ejemplo con `cv2.addWeighted` aplicado sobre la región de la máscara) y, opcional, contorno con `cv2.findContours` y `cv2.drawContours`. Área $a$ en porcentaje; regla del 1 %. Componer el texto: área, y los mensajes de rechazo que apliquen (si no se detecta pez, se muestra igual el top-3, pero con la advertencia primero).
9. **Prueba de paridad contra el caché (obligatoria).** El preprocesado de este cuaderno es una reescritura del que generó los `.npz`, así que hay que demostrar que son equivalentes. Para 5 imágenes de val leídas con `cv2.imread` y convertidas a RGB: (a) comparar el arreglo de 224×224 y el de 256×256 producidos aquí con las filas correspondientes de `cnn_224_val.npz` y `unet_256_val.npz` (buscar la fila por `filepath`); (b) comparar las probabilidades de la función de la demo con las obtenidas desde `cnn_224_val.npz`; (c) comparar la máscara binaria a 256 de la demo con la obtenida desde `unet_256_val.npz` (IoU entre ambas cercano a 1). Verificar: misma clase y diferencia máxima pequeña en las probabilidades (por ejemplo $<0.01$; diferencias mínimas pueden venir del decodificador JPEG). Repetir intencionalmente sin convertir a RGB: las predicciones deben cambiar (koi y telapia son las más sensibles, por su color naranja/rojizo). Guardar esa comparación: es una respuesta lista para la sustentación. Si la paridad falla, se corrige este cuaderno, no el 01.
10. **Elegir $\tau$ en val y congelar.** Requiere que el paso 16 del cuaderno 03 (congelar la U-Net) y el paso 16 del 02 (modelo final de la CNN) ya estén hechos. Curva cobertura–accuracy sobre val (usar las probabilidades de la CNN sobre `cnn_224_val.npz`, tras $T$ si el 02 lo usó); elegir $\tau$ y escribir `config_inferencia.json` con umbral de máscara $t$ y decisión de postprocesado (copiados de `unet_metricas_val.json`, que fijó el cuaderno 03), $\tau$, $T$ (de `cnn_metricas_val.json`, o 1), área mínima, postprocesado sí/no, interpolaciones, MD5 de los modelos (calculado con `hashlib`) y fecha. Verificar: archivo escrito antes de las 23:59 del miércoles 11 y MD5 coincidente con los `.keras` del Drive.
11. **Interfaz.** `gr.Interface` con `fn` = la función de predicción, `inputs` = `gr.Image(type="numpy", image_mode="RGB")`, `outputs` = `gr.Label(num_top_classes=3)`, `gr.Image` y `gr.Textbox`; título y descripción breve que diga que la confianza no está calibrada. `gr.Blocks` solo si sobra tiempo (sección 5).
12. **Ejemplos precargados.** Una imagen por clase (9), elegidas **al azar con `SEED = 42`** dentro del split test, pasadas en el argumento `examples`. Ver sección 5 sobre cuándo cargarlas.
13. **Lanzar.** Hacer una predicción de calentamiento antes de lanzar (la primera llamada es lenta por el trazado del grafo). `launch` con `share=True` y `debug=True` para ver errores en la celda. Verificar: el enlace público abre desde un celular con datos móviles.
14. **Verificar el congelamiento (inicio de la evaluación, D7).** Celda separada. Leer `config_inferencia.json`, recalcular el MD5 de `cnn_best.keras` y `unet_best.keras` con `hashlib` y compararlo con el registrado. Verificar: coinciden; si no, detenerse y aclarar en el equipo antes de seguir. No se ejecuta nada más sobre test hasta que esta celda pase.
15. **Construir la tabla maestra de test.** Leer `splits.csv` con Pandas, filtrar `split == "test"`, unir con `mask_qc.csv` por `filepath` para traer `estado`. Cargar ambos `.npz` de test con `np.load`. Verificar: número de filas $\approx 1362$ (el valor exacto de la partición del cuaderno 01); proporción por clase similar a la de train; `X.dtype == uint8`; formas $(n,224,224,3)$ y $(n_U,256,256,3)$; `np.unique(M)` es $\{0,1\}$; y que, al unir por la clave `filepath` de cada `.npz`, el vector `y` coincide elemento a elemento con la columna `label` de la tabla (esto evita el error (c) de CIFAR-10, predicciones desalineadas).
16. **Pasada única del sistema sobre test.** CNN: `model.predict` sobre `X` de `cnn_224_test.npz` con `batch_size` 64, sin aumento y sin dividir entre 255; guardar las probabilidades completas (forma $(n,9)$, filas que suman $\approx 1$, sin NaN). U-Net: `model.predict` sobre `X` de `unet_256_test.npz` con `batch_size` 16; binarizar con $t$ de `config_inferencia.json` y aplicar la componente conexa más grande solo si así quedó congelado. Calcular IoU y Dice por imagen con NumPy contra `M`, únicamente en filas con `estado == "ok"`. Convención: si predicción y referencia están ambas vacías, IoU $=1$; si solo la predicción está vacía, IoU $=0$. Verificar: IoU y Dice en $[0,1]$; $n_{ok}$ reportado junto con cuántas imágenes de test quedaron fuera.
17. **Verificación cruzada con los cuadernos 02 y 03.** Las clases predichas del paso 16 deben coincidir fila a fila (por `filepath`) con `pred_test_cnn.csv`, y el IoU medio debe coincidir con el de `unet_metricas_test.json` (cuaderno 03). Verificar: coincidencia total. Si difieren, hay una desalineación o un preprocesado distinto: se corrige el error técnico, se documenta en el cuaderno y se reporta el resultado corregido, sin tocar modelos ni umbrales.
18. **Métrica $S$ y tabla 2×2.** Calcular $S$ sobre $\mathcal{T}_{ok}$ y la tabla 2×2 (clase correcta / incorrecta × $\mathrm{IoU}\ge0.5$ / $<0.5$); comparar $S$ con $\mathrm{Acc}\cdot P(\mathrm{IoU}\ge 0.5)$. Verificar: las cuatro celdas suman $n_{ok}$.
19. **Confianza y regla de rechazo en test.** Histograma de $\max_k \hat p_k$ separado en aciertos y errores; con el $\tau$ congelado, reportar cobertura (fracción de test con $\max_k\hat p_k\ge\tau$) y accuracy dentro de esa fracción. Esto respalda lo que muestra la demo. No se vuelve a elegir $\tau$.
20. **Análisis de errores de clasificación del sistema.** Ordenar los pares fuera de la diagonal de la matriz de confusión sin normalizar y listar los 5 más frecuentes. Para rui, katla y mrigel (las carpas), armar una grilla de imágenes mal clasificadas con clase real, predicha y confianza. Describir en 3–5 líneas qué se ve: forma del cuerpo, cabeza, aletas, color, iluminación, tamaño del pez en la foto.
21. **Análisis de errores de segmentación y de combinación.** Visualizar los 10 casos de menor IoU en cuatro columnas (imagen, referencia, predicción, overlay). Clasificar cada caso en una categoría: sombra pegada al pez, aletas o cola recortadas, huecos por reflejos plateados, marca de agua incluida, pez que toca el borde, o **referencia equivocada** (la pseudo-máscara está peor que la predicción). Contar casos por categoría. Con la tabla 2×2, describir las imágenes en que fallan los dos modelos y comentar si los errores están correlacionados (comparación de $S$ con el producto del paso 18).
22. **Tabla final para el informe y exportación.** Armar la fila DL con las mismas columnas que usará ML clásico: accuracy, F1 macro (con `classification_report`, `digits=3`, `output_dict=True`), IoU medio, Dice medio, $S$, cobertura y accuracy con $\tau$, $n$ y $n_{ok}$. Opcional: intervalo bootstrap percentil del F1 macro con $B=1000$ remuestreos a nivel de `group_id` (sección 5). Guardar tablas y figuras con los nombres de la sección 8 (`plt.savefig` con `dpi=200` y `bbox_inches="tight"`).
23. **Ensayo y respaldo (D8).** Ejecutar el guion de la sección 6.1 y producir el material de contingencia de la sección 6.2.

## 5. Decisiones de diseño y alternativas

- **Una sola función de predicción, definida en una celda de este cuaderno.** El preprocesado de 224 y 256 existe en dos cuadernos: el 01 (para generar el caché) y este (para la demo); el 02 y el 03 leen el caché y no preprocesan, con dos excepciones menores y declaradas: el 02 redimensiona la máscara a 224 para el chequeo de atajos y el 03 redimensiona la imagen a 256 al exportar `unet_pred_test/` (comprueba su paridad con el caché). Tener varias copias obliga a la prueba de paridad del paso 9, que es lo que garantiza que no divergen. Alternativas descartadas: un módulo `.py` compartido (la estructura del repo es de cuadernos autocontenidos) y copiar el preprocesado en cada cuaderno.
- **`gr.Interface` como mínimo viable.** Se arma en minutos y cubre las salidas del contrato. `gr.Blocks` permite filas, columnas, botones y un panel de Markdown con la explicación (útil para presentar), pero cuesta más tiempo; se deja como mejora del D6 por la tarde si todo lo demás funciona.
- **Vecino más cercano sobre la máscara binaria.** Es lo que fija el contrato y conserva valores $\{0,1\}$. Alternativa opcional: reescalar el mapa de probabilidad con bilineal y umbralizar después (bordes más suaves); si se usa, declararlo.
- **Imágenes no cuadradas.** El dataset es 500×500 y se redimensionó directo. **Decisión sugerida (confirmar con el equipo):** para fotos externas no cuadradas, rellenar a cuadrado con `cv2.copyMakeBorder` y `cv2.BORDER_REPLICATE` antes de redimensionar, y recortar ese relleno de la máscara antes de reescalarla; si no hay tiempo, redimensionar directo y declarar la deformación.
- **Ejemplos de test y congelamiento.** **Decisión sugerida:** durante el desarrollo del D6 se prueba con imágenes de val; las imágenes de test se cargan como ejemplos después de las 23:59, cuando ya nada puede cambiar. Mirar predicciones de test antes del congelamiento invita a "arreglar" el modelo con información de test.
- **Ejemplos al azar, no escogidos.** Elegir solo los aciertos sería una demo engañosa; la selección aleatoria con semilla se puede justificar ante el profesor.
- **IoU a 256×256 contra `M` del `.npz`.** Es la misma resolución de val, así las cifras son comparables con las del cuaderno 03. Alternativa descartada como métrica principal: IoU a resolución original (requiere reescalar y mezcla el error del reescalado); puede reportarse aparte si sobra tiempo.
- **Evaluar el sistema solo en máscaras ok.** Medir contra referencias marcadas como dudosas mide el error de la referencia, no el del modelo. Se reporta $n_{ok}$ y cuántas imágenes de test quedaron fuera.
- **$S$ sin reglas de rechazo.** La métrica del sistema mide a los dos modelos tal cual, sin el filtro de $\tau$ ni el del área mínima; el efecto de $\tau$ se reporta aparte (paso 19) como cobertura y accuracy. Mezclarlos haría que $S$ dependiera de un umbral y ya no midiera solo a los modelos.
- **Promedio por imagen, no IoU acumulado por píxeles.** Con el acumulado, los peces grandes dominan. Además, `keras.metrics.BinaryIoU` por defecto promedia IoU de fondo y de pez (`target_class_ids=(0, 1)`), lo que infla la cifra; por eso el cálculo final se hace con NumPy.
- **Intervalos opcionales: bootstrap percentil sobre test.** En cada remuestreo $b$ se toma con reemplazo un conjunto de grupos del mismo tamaño, se calcula $F1^{*b}_{\text{macro}}$ y el intervalo es $\big[q_{0.025},\,q_{0.975}\big]$ de los $B$ valores. Se remuestrea por `group_id` para respetar los duplicados agrupados. Mide la variabilidad debida a la muestra de test, **no** la debida al entrenamiento (semilla, orden de lotes). Medir esta última exigiría reentrenar con varias semillas, lo que el cronograma no permite; se declara como limitación.
- **Postprocesado y umbral fijos.** Si se quiere mostrar el efecto del postprocesado, se reporta como ablación secundaria, marcada como tal, sin elegir con ella.

## 6. Errores comunes y trampas

- Aplicar `COLOR_BGR2RGB` a la imagen de Gradio (que ya es RGB) o no aplicarlo a una imagen leída con OpenCV: invierte rojo y azul y cambia las predicciones.
- Dividir entre 255 antes de la CNN (doble normalización): el modelo ya trae `Rescaling(1/255)`; la entrada queda en $[0, 1/255]$ y la accuracy cae a nivel de azar.
- Pasar `dsize` a `cv2.resize` como (alto, ancho): con fotos no cuadradas la máscara queda transpuesta o deformada.
- Reescalar la máscara con bilineal y luego contar el área sin umbralizar.
- Cargar los modelos dentro de la función: cada predicción tardaría varios segundos.
- `load_model` sin `compile=False` para la U-Net con pérdida propia: error de deserialización.
- Fotos de celular con orientación EXIF: verificar que la imagen llega derecha a la función.
- Confiar en el enlace `share=True` como recurso estable: caduca (Gradio 4 documenta 72 horas; verificar con la versión instalada) y muere al cerrarse o reiniciarse la sesión de Colab, que además se desconecta por inactividad.
- Presentar la confianza como "probabilidad de que sea esa especie".
- Saltarse la prueba de paridad porque "el código es el mismo": no lo es, se reescribió, y una interpolación distinta basta para degradar la CNN sin dar error.
- Correr la celda de evaluación en test dos veces "porque algo falló" y quedarse con el mejor resultado: eso es selección en test. Si hay un error técnico (por ejemplo, desalineación), se corrige el error, se documenta en el cuaderno y se reporta el resultado corregido, sin tocar modelos, umbrales ni $\tau$.
- Predicciones desalineadas con etiquetas por haber filtrado o reordenado una tabla y no la otra.
- Usar `BinaryIoU` con su configuración por defecto (mezcla fondo y pez), o calcular IoU con la salida sigmoide sin binarizar.
- Interpretar el F1 de kalibaush como exacto con $\approx 79$ imágenes en test.
- Presentar el IoU como "precisión de segmentación real": es concordancia con pseudo-máscaras de OpenCV, no con una verdad experta.

### 6.1 Guion de ensayo (14 imágenes)

Llenar una tabla `reports/tables/04_ensayo_demo.csv` con: imagen, origen, clase real, top-1, confianza, área %, mensaje de rechazo, ¿correcto?, explicación preparada.

1. **10 de test:** al menos una de cada carpa (rui, katla, mrigel), kalibaush (la clase más pequeña), y al menos un error real tomado del análisis de errores de los pasos 20 y 21 para explicarlo con honestidad.
2. **3 externas:** fotos propias de celular o de internet con licencia libre, con fondos e iluminación distintos (mano, mesa de madera, mercado). La tilapia (telapia) es fácil de conseguir en mercados colombianos y es la mejor candidata para una foto propia. Esperar peor desempeño: es cambio de dominio, y hay que decirlo.
3. **1 que no sea pez:** por ejemplo, un objeto oscuro sobre fondo claro. Preparar la explicación de mundo cerrado y mostrar si se activan las reglas de rechazo. Si la U-Net segmenta el objeto, explicar que aprendió "objeto contrastado sobre fondo claro", porque así eran las pseudo-máscaras.

### 6.2 Plan de contingencia el día de la sustentación

1. **Sin GPU:** la inferencia de una imagen funciona en CPU en pocos segundos; probar el D8 la demo completa con entorno de ejecución CPU.
2. **Sin internet o sin Colab:** **Decisión sugerida:** un portátil con un entorno local (Gradio y TensorFlow instalados antes, modelos descargados) que corra la misma función con `share=False` en el navegador local.
3. **Último recurso:** figuras pregrabadas de los 14 casos del ensayo (`reports/figures/04_demo_respaldo_grilla.png`) y un video corto de 60–90 s grabando la demo en vivo (`demo_respaldo.mp4` en `MyDrive/sylfish/`, no en GitHub por tamaño), enlazados desde la presentación.
4. La noche anterior: relanzar la demo, comprobar el enlace desde otro dispositivo, y tener los archivos de respaldo descargados localmente.

## 7. Criterio de aceptación

- [ ] Ambos modelos se cargan una vez con `compile=False` y las formas de entrada coinciden con el contrato.
- [ ] Prueba de paridad contra `cnn_224_val.npz` y `unet_256_val.npz` superada en 5 imágenes de val (misma clase, diferencia pequeña, máscara casi idéntica).
- [ ] Prueba RGB/BGR documentada con al menos una imagen de koi o telapia.
- [ ] Máscara reescalada con `cv2.INTER_NEAREST`, valores $\{0,1\}$, forma $(H,W)$.
- [ ] Top-3 visible con `gr.Label`, overlay a resolución original y área % en texto.
- [ ] Reglas de rechazo activas con $\tau$ elegido en val y área mínima del 1 %.
- [ ] `config_inferencia.json` escrito antes de las 23:59 del miércoles 11.
- [ ] 9 ejemplos de test elegidos con `SEED = 42`, cargados después del congelamiento.
- [ ] Enlace `share=True` probado desde otro dispositivo.
- [ ] MD5 de ambos modelos coincide con el registrado en `config_inferencia.json` antes de evaluar.
- [ ] Tamaño de test y alineación `y` ↔ `label` verificados en una celda explícita; predicciones coincidentes con `pred_test_cnn.csv`.
- [ ] La celda de evaluación en test se ejecutó una sola vez y su salida quedó guardada en el cuaderno.
- [ ] Métrica $S$ y tabla 2×2 calculadas, con $n_{ok}$ reportado.
- [ ] Top-5 confusiones listadas, grilla de errores en carpas comentada y peores 10 casos de segmentación clasificados por categoría.
- [ ] Tabla final exportada con las mismas columnas que la de ML clásico.
- [ ] Ensayo de 14 imágenes registrado y material de respaldo generado.
- [ ] (Opcional) Intervalo bootstrap del F1 macro con $B$ y nivel declarados.

## 8. Artefactos que debe dejar

- `notebooks/04_demo/04_demo.ipynb` (salidas limpiadas por `nbstripout`; las cifras viven en las tablas). La función de predicción y el preprocesado viven en sus celdas.
- `MyDrive/sylfish/models/config_inferencia.json`.
- `reports/figures/04_paridad_rgb_bgr.png`: misma imagen, predicción con RGB y con BGR.
- `reports/figures/04_cobertura_vs_accuracy_val.png`: curva para elegir $\tau$.
- `reports/figures/04_demo_captura.png`: captura de la interfaz para el informe.
- `reports/figures/04_demo_respaldo_grilla.png` y `MyDrive/sylfish/demo_respaldo.mp4`.
- `reports/tables/04_ensayo_demo.csv`.
- `reports/tables/04_predicciones_sistema_test.csv`: una fila por imagen de test con `filepath`, `clase`, `label`, `pred`, `conf_top1`, `estado`, `iou`, `dice`, `sistema_ok`.
- `reports/tables/04_resumen_final_test.csv`: accuracy, F1 macro (con intervalo si se calculó), IoU medio, Dice medio, $S$, cobertura y accuracy con $\tau$, $n$ y $n_{ok}$; es la fila DL de la tabla final del informe y de la comparación con ML clásico.
- `reports/tables/04_errores_segmentacion.csv`: los 10 peores casos con su categoría.
- `reports/figures/04_sistema_tabla2x2_test.png` (o la tabla en el informe).
- `reports/figures/04_cnn_errores_carpas_test.png`
- `reports/figures/04_cnn_confianza_aciertos_errores_test.png`
- `reports/figures/04_unet_peores10_test.png`
- (Opcional) `reports/figures/04_bootstrap_f1_test.png`

## 9. Preguntas probables de sustentación

1. **¿Qué significa el porcentaje que muestra la demo?** Es el máximo de softmax de la CNN. Indica qué clase prefiere el modelo, pero no está calibrado: no equivale a la probabilidad real de acierto.
2. **¿Por qué una foto que no es un pez también recibe una especie?** Porque softmax reparte la probabilidad solo entre 9 clases (mundo cerrado); no existe la salida "otra cosa". Lo mitigamos con el umbral $\tau$ y el área mínima, pero no lo resolvemos.
3. **¿Cómo garantizan que la demo preprocesa igual que el entrenamiento?** El preprocesado de la demo se reescribió en este cuaderno, así que lo comprobamos con una prueba de paridad: los arreglos, las predicciones y las máscaras de la demo coinciden con los del caché `.npz` en imágenes de val.
4. **¿Qué pasa si se confunde RGB con BGR?** Se intercambian los canales rojo y azul; un koi naranja se ve azul para la red y la predicción cambia. Gradio entrega RGB y OpenCV lee BGR; por eso se convierte solo lo que lee OpenCV.
5. **¿Por qué vecino más cercano para la máscara?** Porque conserva valores binarios; bilineal crea valores intermedios en el borde que no son ni pez ni fondo.
6. **¿Cómo se unen CNN y U-Net?** No comparten pesos ni se entrenan juntas; la misma imagen pasa por ambas en la función de predicción y se combinan las salidas: clase de la CNN y máscara de la U-Net.
7. **¿Cómo eligieron $\tau$?** En val, con la curva cobertura–accuracy; nunca en test.
8. **¿Por qué falla más con fotos de celular?** Cambio de dominio: el entrenamiento fue con peces sobre superficie clara y lisa, encuadre similar y 500×500; otros fondos, luz y escala están fuera de esa distribución.
9. **¿Qué es el área %?** La fracción de píxeles marcados como pez sobre el total de la imagen. Es un dato descriptivo, no un tamaño físico: depende de la distancia de la cámara.
10. **¿Y si el día de la sustentación no hay internet?** Tenemos un entorno local en un portátil y, como último recurso, figuras y un video pregrabados de la demo funcionando.
11. **¿Por qué evaluaron en test una sola vez?** Porque elegir entre varias evaluaciones en test sesga hacia arriba el resultado: el máximo de estimaciones ruidosas sobreestima el desempeño real. Toda elección se hizo en val; test es la estimación honesta.
12. **¿Qué significa la métrica del sistema completo?** El porcentaje de imágenes donde el usuario de la demo recibe la especie correcta y una máscara aceptable ($\mathrm{IoU}\ge0.5$) a la vez. Es la cifra que mejor representa la demo; se calcula solo sobre máscaras de referencia en estado ok, y se compara con el producto $\mathrm{Acc}\cdot P(\mathrm{IoU}\ge0.5)$ para ver si los errores de ambos modelos están correlacionados.
13. **¿Contra qué miden el IoU y qué tan seguros están de los números?** Contra pseudo-máscaras de OpenCV en estado ok: mide qué tan bien la U-Net reproduce ese procedimiento, no una verdad experta. Con $n\approx1362$ y una accuracy cercana a 0.90 el margen es de $\pm1.6$ puntos aproximadamente; para una clase pequeña como kalibaush, cerca de $\pm8$ (recalcular con las cifras reales). El bootstrap cuantifica la variación por muestra de test, no por reentrenamiento.
