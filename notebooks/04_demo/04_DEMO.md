# Módulo 04 — Demo (clasificación + segmentación)

**Autor:** Jairo de Jesús Blanco Pérez · **Materia:** Inteligencia Artificial 2026-2
**Entrada:** una imagen de pez (cualquier tamaño) · **Salida:** imagen con contorno y leyenda `especie  confianza%`, más un resumen en texto
**Modelos usados (finales):** `models/cnn_best.keras` (módulo 02), `models/unet_v2.keras` y `models/unet_v2_umbral.json` (módulo 05, U-Net entrenada con máscaras de SAM 2)

> **Estado (19 de noviembre de 2026).** El modelo final de segmentación, tras evaluar la opción SAM 2, es `unet_v2`. La demo se escribió y se probó con `unet_v1` (`unet_v1.keras`, `unet_v1_umbral.json`); el cambio consiste en apuntar las dos rutas a los archivos de la v2 (mismo $\tau = 0.6$ y mismo preprocesamiento). **Ese cambio y el reensayo con las cuatro fotos externas aún no se han hecho**: la tabla de la sección 4 corresponde a la v1 y debe rehacerse con la v2 antes de citarla.

---

## 1. Propósito

Reunir en una sola interfaz los dos modelos entrenados por separado:

- La **CNN** (EfficientNetB0, transfer learning) dice qué especie es y con qué confianza.
- La **U-Net** (binaria pez/fondo; versión final `unet_v2`) dibuja el contorno del pez.

La demo es **cruda a propósito**: usa los modelos ya entrenados sin umbrales nuevos de confianza y sin reentrenar. La interfaz es de Gradio y corre sobre Colab.

## 2. Preconceptos

**Dos modelos independientes.** Cada uno recibe la misma foto con su propio preprocesamiento. No se pasan información entre sí: el contorno no depende de la especie predicha.

**Clasificador de conjunto cerrado.** La salida es un softmax sobre 9 clases,

$$\hat y_k=\frac{e^{z_k}}{\sum_{j=1}^{9}e^{z_j}},\qquad \sum_k \hat y_k = 1,$$

así que siempre reparte el 100 % entre las 9 especies conocidas. Si la foto es de otra especie, el modelo igual elige una. La **confianza** es $\max_k \hat y_k$; no es una probabilidad calibrada.

**Máscara.** El U-Net entrega para cada píxel una probabilidad $p_{ij}\in[0,1]$. Se binariza con $\tau$:

$$m_{ij}=\mathbf 1\!\left[p_{ij}>\tau\right],\qquad \tau=0.6.$$

**Área del pez** (fracción de la imagen):

$$a=\frac{1}{HW}\sum_{i,j} m_{ij}.$$

**Contorno.** Se obtiene de la máscara con `cv2.findContours` (borde exterior); se dibuja el de mayor área.

## 3. Decisiones y datos

| Decisión | Valor | Por qué |
|---|---|---|
| Interfaz | Gradio (`gr.Interface`), `share=True` | Se usa desde el navegador o el celular mientras el Colab esté activo |
| Preprocesamiento CNN | RGB, 224×224, `cv2.INTER_AREA`, valores 0–255 sin dividir | EfficientNetB0 normaliza por dentro; igual que la caché del módulo 01 |
| Preprocesamiento U-Net | RGB, 256×256, `cv2.INTER_AREA`, dividido entre 255 | Igual que el entrenamiento de las v1 y v2 (módulos 03 y 05) |
| Modelo de segmentación | `unet_v2.keras` | Modelo final seleccionado tras la opción SAM 2 (`05_SAM2.md`); la v1 queda de respaldo |
| Umbral de máscara | $\tau=0.6$, leído de `unet_v2_umbral.json` (clave `tau`) | Elegido en val en el módulo 05; coincide con el de la v1 |
| Ampliación de la máscara | se amplían las **probabilidades** a tamaño original con interpolación bilineal y después se aplica $\tau$ | Contorno más suave que ampliar la máscara binaria con vecino más cercano |
| Aviso de fiabilidad | contorno naranja y aviso si $a\notin[0.02,\,0.35]$ | Es el rango del control de calidad de GrabCut del módulo 01, que coincide con `AREA_MIN` y `AREA_MAX` del control de SAM 2; no se ajustó con datos nuevos |
| Umbral de confianza | **ninguno** | Decisión: demo cruda; la especie y el porcentaje se muestran siempre |
| Leyenda y trazo | proporcionales al ancho de la imagen | Se ven bien en fotos pequeñas y grandes |

**Contratos que no se pueden mezclar:** CNN = 224 en 0–255; U-Net = 256 en $[0,1]$. Si se intercambian no hay error: las predicciones salen mal sin avisar.

**Orden de clases:** boal, ilish, kalibaush, katla, koi, mrigel, pabda, rui, telapia (el mismo del módulo 02).

## 4. Resultados

No se calculan métricas en la demo; las métricas de los modelos están en los módulos 02 (test: accuracy 0.9942, F1 macro 0.9955) y 05 (U-Net final `unet_v2`, test: mIoU 0.8944 frente a pseudo-máscaras de SAM 2; no comparable con el 0.8344 de la v1, módulo 03). Aquí solo hay una prueba **cualitativa** con cuatro fotos de la web, sin conjunto de evaluación, que ilustra el comportamiento y **no** es evidencia de generalización.

**Resultados con `unet_v1` (13 de noviembre; pendientes de repetir con `unet_v2`):**

| Foto | Predicción | Área | Contorno | Lectura |
|---|---|---|---|---|
| Rui, fondo blanco | rui, 94.9 % | 21.9 % | Muy bueno (lomo, aletas, cola) | Acierto con fondo distinto al del dataset |
| Rui, fondo claro, pez en diagonal | rui, 69.0 % | 24.7 % | Bueno en lomo y cola; algo justo en el vientre (a ojo) | Acierto con confianza menor |
| Pez de acuario con plantas | katla, 68.6 % | 18.9 % | Malo: cubre solo el torso e incluye plantas | Especie probablemente fuera de las 9 (no verificada); escena de acuario |
| Pez tipo pangasio, fondo blanco | kalibaush, 72.6 % | 23.7 % | Muy bueno | Especie probablemente fuera de las 9 (no verificada) |

**Resultados con `unet_v2`:** no hay todavía. Cuando se repita la prueba con las mismas cuatro fotos, completar una tabla igual y comparar a ojo con la anterior (la clasificación no cambia, porque la CNN es la misma).

Lo que sugiere la prueba con la v1 (con tan pocos casos, solo como hipótesis): la segmentación funciona bien con fondos limpios aun con especies no vistas, porque el U-Net aprende "pez frente a fondo"; la clasificación siempre elige una de las 9 clases y su porcentaje no indica si la especie es conocida.

## 5. Limitaciones

1. **Conjunto cerrado:** no existe la clase "otro"; un pez de una especie desconocida se clasifica como una de las 9, con porcentaje que no significa "probabilidad de ser esa especie".
2. **Fondos recargados:** con acuario, plantas o escenas con plato/balde, la máscara puede fallar. La v1 no vio el 21 % de escenas de ese tipo (descartadas por el control de calidad de GrabCut); la v2 se entrenó con 97.0 % de las imágenes, pero con ruido de etiqueta residual (`05_SAM2.md`). El módulo 02 mostró además que la CNN depende en parte del fondo.
3. **El aviso de área solo detecta tamaños absurdos:** una máscara con forma equivocada y área plausible (caso del acuario con la v1, 18.9 %) no activa el aviso.
4. **Pseudo-máscaras:** el U-Net aprendió de máscaras generadas automáticamente (SAM 2 en la v2, GrabCut en la v1); el contorno hereda sus errores.
5. **Un solo pez por imagen:** se dibuja el contorno de mayor área; con varios peces solo se marca uno.
6. **Dependencia de Colab:** el enlace `share=True` solo funciona mientras el cuaderno esté ejecutándose y caduca.
7. **Prueba cualitativa mínima:** cuatro imágenes; sin métricas sobre fotos externas, y con la v2 todavía sin repetir.

## 6. Glosario

- **Conjunto cerrado:** clasificador que solo conoce las clases con las que se entrenó.
- **Contorno:** curva que rodea la región blanca de la máscara.
- **Fuera de distribución:** datos distintos a los de entrenamiento (otra especie, otro fondo, otra cámara).
- **Gradio:** librería que crea una interfaz web alrededor de una función de Python.
- **INTER_AREA:** interpolación que promedia píxeles al reducir una imagen.
- **Pseudo-máscara:** máscara generada automáticamente (GrabCut o SAM 2), no dibujada a mano.
- **Softmax / confianza:** distribución sobre las 9 clases; la confianza es su máximo.
- **$\tau$:** umbral que convierte la probabilidad de píxel en máscara binaria (0.6).
- **`unet_v2`:** U-Net entrenada con máscaras de SAM 2; modelo final de segmentación.
