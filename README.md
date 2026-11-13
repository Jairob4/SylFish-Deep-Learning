# sylfish-dl

Clasificación y segmentación de peces con deep learning, sobre la base de datos SylFishBD. Es el proyecto final de Inteligencia Artificial (Universidad del Magdalena, 2026-2). Entrega: sábado 14 de noviembre de 2026.

El sistema recibe la fotografía de un pez y responde dos cosas: de qué especie es, con qué confianza, y dónde está. La demo es una interfaz Gradio ejecutada en Colab (`share=True`, enlace público temporal): se sube una imagen y se obtiene la imagen con el contorno del pez y una leyenda con la especie y el porcentaje de confianza, más un resumen con el área del pez.

Este repositorio contiene la parte de deep learning. La parte de ML clásico, que desarrolla otro subgrupo del equipo, vive fuera de este repositorio y usa el mismo `splits.csv`.

## Cómo funciona

```
              ┌─► CNN (EfficientNetB0)  ─► especie + confianza ─┐
foto del pez ─┤                                                  ├─► imagen con contorno y leyenda
              └─► U-Net (binaria)       ─► máscara pez/fondo  ───┘
```

Son dos modelos independientes que solo se unen en la demo. Se entrenan, validan y guardan por separado, de modo que el fallo de uno no contamina al otro y cada uno tiene sus propias métricas.

## Datos

SylFishBD tiene 9075 fotografías JPG a color de 500×500 píxeles. La única anotación es la clase, dada por la carpeta; no hay máscaras de segmentación. Las clases siguen este orden:

| boal | ilish | kalibaush | katla | koi | mrigel | pabda | rui | telapia | Total |
|---|---|---|---|---|---|---|---|---|---|
| 885 | 789 | 525 | 1133 | 592 | 1293 | 862 | 1670 | 1326 | 9075 |

La razón entre la clase mayor y la menor es $1670/525\approx3.18$. Hay copias exactas y casi duplicadas, así que la partición se hace **por grupos de duplicados** (MD5 y dHash con distancia de Hamming $\le 5$), 70/15/15 y con semilla 42: una foto y sus copias nunca quedan en conjuntos distintos. Quedan 8975 imágenes (train 6246, val 1359, test 1370); 100 se excluyeron por ruido de etiqueta. Las imágenes, las máscaras, los `.npz` y los modelos no están en este repositorio: viven en Drive ([enlace a la carpeta de Drive]). Referencia del dataset: [completar referencia del dataset].

## Solución

**Clasificador (CNN).** EfficientNetB0 preentrenada en ImageNet con una cabeza nueva de 9 clases (`GlobalAveragePooling2D`, `Dropout` 0.3, `Dense` 9 con softmax). Se entrena en dos fases: primero solo la cabeza ($\eta=10^{-3}$) y después un ajuste fino de las últimas 30 capas del backbone ($\eta=10^{-5}$) con las capas `BatchNormalization` en modo inferencia. Entrada: RGB 224×224 con valores 0–255, porque EfficientNetB0 normaliza por dentro. El aumento (volteos, rotación, zoom, contraste) va dentro del modelo y solo actúa en entrenamiento. Como línea base se entrenó una CNN propia de 5 bloques Conv-BN-ReLU-MaxPool (≈ 1.05 M de parámetros). Para compensar el desbalance se usan pesos de clase $w_c=N/(K\,N_c)$ y se elige el modelo por F1 macro en validación.

**Segmentación (U-Net).** Red codificador-decodificador hecha desde cero en Keras (API funcional), con 4 niveles de 16, 32, 64 y 128 filtros, un cuello de botella de 256 filtros a 16×16 y `Conv2DTranspose` 2×2 para subir. Cada nivel usa dos veces Conv 3×3, `BatchNormalization` y ReLU, y las conexiones de salto se concatenan. Tiene 1,946,993 parámetros. Entrada 256×256 dividida entre 255; salida de 1 canal con sigmoide. La pérdida suma entropía cruzada binaria y Dice, que contrarresta que el pez ocupe solo ≈ 15 % de los píxeles:

$$\mathcal{L}=\mathcal{L}_{BCE}+\mathcal{L}_{Dice},\qquad \mathcal{L}_{Dice}=1-\frac{2\sum_i p_i g_i+\varepsilon}{\sum_i p_i+\sum_i g_i+\varepsilon}$$

donde $p_i$ es la probabilidad predicha y $g_i\in\{0,1\}$ la máscara de referencia. La métrica principal es el mIoU por imagen, con $\mathrm{IoU}(A,B)=|A\cap B|/|A\cup B|$ y la máscara binarizada con $\tau=0.6$ (elegido en validación).

**Máscaras.** Como el dataset no trae máscaras, se generan con GrabCut (OpenCV) y un control de calidad por área y ocupación del borde. La U-Net se entrena y se evalúa solo con las máscaras que pasan el control (7118 de 8975; train 4933, val 1066, test 1119). Es decir, aprende de **pseudo-máscaras**, no de una verdad de terreno dibujada a mano.

## Evaluación

Todas las decisiones (arquitectura, umbral $\tau$, selección de modelo) se toman en validación. El test se evalúa una sola vez, con el modelo congelado.

La CNN se evalúa con accuracy, F1 macro, `classification_report` y matriz de confusión normalizada por filas; la accuracy sola engaña con un desbalance de 3.18. La U-Net se evalúa con IoU por imagen (media y mediana), Dice global y matriz de confusión de píxeles.

## Resultados

| Modelo | Métrica | Validación | Prueba |
|---|---|---|---|
| CNN (EfficientNetB0, `tl_fase2`) | Accuracy | 0.9875 | 0.9942 |
| CNN (EfficientNetB0, `tl_fase2`) | F1 macro | 0.9892 | 0.9955 |
| U-Net | mIoU por imagen | 0.8334 | 0.8344 |

La CNN desde cero obtuvo, con el mismo protocolo, 0.9691 de accuracy y 0.9693 de F1 macro en validación.

Cómo leer estas cifras:

- El IoU de la U-Net mide acuerdo con las pseudo-máscaras de GrabCut, no con una verdad de terreno. En los peores casos de validación que se revisaron a mano, la red suele dibujar mejor que su máscara de referencia, así que el mIoU sería una cota inferior de la calidad real; es una observación cualitativa sobre pocas muestras.
- Las cifras de la CNN son en distribución. Al neutralizar el fondo de las imágenes de validación (con máscara ok) la accuracy baja de 0.99 a 0.82, lo que indica que la red también usa el contexto y no solo el pez. Con fotos de otros fondos puede rendir peor.
- La confianza es el máximo del softmax y no está calibrada: la media fue 0.986 en aciertos y 0.722 en errores en validación.

## Limitaciones

- Clasificador de conjunto cerrado: no existe la clase "otro", así que una especie desconocida se asigna a una de las 9.
- La U-Net no vio el ≈ 21 % de imágenes con plato, balde o mesa grandes (descartadas por el control de calidad); con fondos muy recargados, como un acuario con plantas, el contorno puede fallar.
- Cada experimento se corrió con una sola semilla y los experimentos `aumento` y `sin_cw` de la CNN resultaron inestables, por lo que no son concluyentes.
- La prueba de la demo con fotos externas es cualitativa y de cuatro imágenes.

El detalle de cada módulo, con sus decisiones y limitaciones, está en su carpeta de `notebooks/`.

## Estructura del repositorio

```text
sylfish-dl/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_preparacion_datos/         cuaderno, GUIA.md y 01_PREPARACION_DATOS.md
│   ├── 02_cnn/                       cuaderno, GUIA.md y 02_CNN.md
│   ├── 03_unet/                      cuaderno, GUIA.md y 03_UNET.md
│   └── 04_demo/                      cuaderno, GUIA.md y 04_DEMO.md
└── reports/
    ├── figures/                      figuras de los informes
    └── tables/                       tablas de resultados (.csv, .json)
```

El repositorio solo contiene los cuadernos, sus guías y los resultados. Los datos y los modelos están en la carpeta de Drive: [enlace a la carpeta de Drive].

Cada módulo tiene su `GUIA.md` (plan de trabajo) y un `.md` con propósito, preconceptos, decisiones, resultados, limitaciones y glosario. Cada cuaderno es autocontenido: carga sus datos, define el modelo, entrena y evalúa en el mismo archivo, y no hay módulos `.py` compartidos. Las constantes comunes (semilla, tamaños, rutas, orden de clases) se copian idénticas en una celda de configuración al inicio de cada uno.

## Cómo ejecutar

Se copia la carpeta de Drive ([enlace a la carpeta de Drive]) a `MyDrive/sylfish/` y se monta en Colab. Luego se abren los cuadernos en este orden: `01_preparacion_datos` (deja `splits.csv`, las máscaras y los `.npz` en Drive), `02_cnn` y `03_unet` (leen el caché del 01 y pueden correr en paralelo) y `04_demo` (carga `cnn_best.keras`, `unet_v1.keras` y `unet_v1_umbral.json`, y abre la interfaz Gradio). Todo usa una GPU T4, salvo el 01, que corre en CPU casi por completo.

## Equipo

Jairo de Jesús Blanco Pérez: preparación de datos, máscaras, U-Net y demo. Compañero DL: CNN de clasificación. La parte de ML clásico la desarrolla otro subgrupo fuera de este repositorio.
