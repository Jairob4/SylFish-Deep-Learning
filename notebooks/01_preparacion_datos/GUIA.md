# Guía 01 — Preparación de datos: exploración, duplicados, partición, máscaras y caché

**Responsable:** Jairo (cuaderno y máscaras); la partición (bloque C) la revisan Jairo y Compañero DL · **Depende de:** `docs/00_SETUP_REPO_COLAB.md` (Colab, Drive y repo listos; zip del dataset en `MyDrive/sylfish/data/raw/`) · **Entrega a:** los cinco integrantes (`splits.csv`), `02_cnn` (`cnn_224_*.npz`, desbalance, riesgo de atajos, máscaras para el chequeo con fondo neutralizado), `03_unet` (`unet_256_*.npz`), `04_demo` (parámetros de preprocesado y máscaras) e Integrantes ML 1–3 (`splits.csv` y máscaras) · **Fecha objetivo:** bloques A–C el D1 (viernes 6 de noviembre de 2026, `splits.csv` publicado esa noche); bloque D del D2 al D3 (sábado 7 y domingo 8); bloque E: caché CNN el D1 por la noche y caché U-Net la noche del D3

---

## 1. De qué se trata

Este cuaderno único fija, para todo el proyecto, los hechos y los artefactos de datos de los que dependen los demás módulos: qué contiene SylFishBD, qué imágenes están duplicadas, en qué split cae cada imagen, cuál es la máscara pez/fondo de cada una y cómo quedan almacenadas, ya preprocesadas, para entrenar. No entrena ningún modelo. Todo lo que 02, 03 y 04 asumen sobre los datos (tamaño de entrada, desbalance, partición, máscaras, normalización) se decide y se mide aquí, una sola vez.

El cuaderno tiene cinco bloques consecutivos. Los tres primeros son la ruta crítica de la noche del D1; el cuarto es la pieza de mayor riesgo del cronograma (si el D3 las máscaras no están listas, la U-Net, la evaluación y la demo se retrasan en cadena); el quinto convierte todo lo anterior en arreglos NumPy que los demás cuadernos leen sin volver a procesar imágenes.

| Bloque | Contenido | Cuándo |
|---|---|---|
| A | Configuración y exploración | D1 |
| B | Inventario y duplicados | D1 |
| C | Partición estratificada por grupos | D1, `splits.csv` publicado por la noche |
| D | Máscaras OpenCV y control de calidad | D2 (v1 y revisión cruzada sobre muestra) y D3 (v2 completa, congelamiento) |
| E | Preprocesado y caché `.npz` | CNN 224: D1 por la noche, justo después de C; U-Net 256: noche del D3, tras congelar las máscaras |

Lo verificado hasta hoy es parcial: 9 clases, 9075 imágenes, muestras en JPG RGB de $500\times 500$ con el pez sobre una superficie clara y lisa, y al menos una foto con la marca de agua "HUAWEI nova 3i DUAL CAMERA" abajo a la izquierda. La uniformidad de resolución y la existencia de duplicados NO están verificadas; los bloques A y B deben cerrarlas antes de partir.

## 2. Entradas y salidas (contrato)

**Entradas**

- Zip del dataset en `MyDrive/sylfish/data/raw/sylfishbd.zip`, descomprimido en `/content/` al iniciar cada sesión: `/content/sylfishbd/classes.txt` y `/content/sylfishbd/images/<clase>/<clase>_NNN.jpg`.
- Orden de clases: el de `classes.txt` (índices 0–8: boal, ilish, kalibaush, katla, koi, mrigel, pabda, rui, telapia).

**Salidas** (rutas relativas a `MyDrive/sylfish/`)

| Archivo | Bloque | Contenido |
|---|---|---|
| `data/splits/inventario.csv` | B | Una fila por imagen: `filepath` (relativa a `sylfishbd/images/`), `clase`, `label`, `alto`, `ancho`, `canales`, `legible`, `md5`, `dhash` (64 bits en 16 caracteres hexadecimales), `L_borde`, `S_borde`, `marca_agua` ∈ {si, no, sin_revisar} |
| `data/splits/pares_duplicados.csv` | B | `filepath_a`, `filepath_b`, `tipo` ∈ {md5, dhash}, `distancia` |
| `data/splits/splits.csv` | C | `filepath`, `clase`, `label`, `group_id`, `split` ∈ {train, val, test}, en ese orden |
| `data/masks/<clase>/<clase>_NNN.png` | D | PNG de 1 canal, valores $\{0,255\}$ (255 = pez), resolución original, mismo nombre base que el JPG |
| `data/masks.zip` | D | Las máscaras comprimidas (adentro, la carpeta `masks/`); se descomprime en `/content/masks/` al iniciar la sesión |
| `data/masks/mask_qc.csv` | D | `filepath`, `area_frac`, `n_componentes`, `toca_borde`, `estado` ∈ {ok, revisar, descartada} |
| `data/masks/mask_params.json` | D | Versión y parámetros congelados: canal, núcleo de desenfoque, núcleos morfológicos, uso de GrabCut |
| `data/masks_oro/<clase>/<clase>_NNN.png` | D | Opcional: conjunto oro corregido a mano, mismo formato |
| `data/processed/cnn_224_{split}.npz` | E | `X` $(n,224,224,3)$ `uint8` RGB, `y` $(n,)$ entera, `filepath` alineado con `X` |
| `data/processed/unet_256_{split}.npz` | E | `X` $(n,256,256,3)$ `uint8` RGB, `M` $(n,256,256,1)$ `uint8` en $\{0,1\}$, `y`, `filepath`; solo máscaras con `estado = ok` |

Las máscaras se generan en `/content/masks/` y se guardan en Drive como zip, porque escribir y leer 9075 PNG sueltos en Drive es lento. Este cuaderno es el único que produce estos archivos; 02, 03 y 04 solo los leen y verifican. `filepath` en los `.npz` permite unir predicciones con `splits.csv` y rastrear los peores casos; `y` permite desglosar IoU por clase de pez. Las figuras y las tablas de resultados de este cuaderno van en `reports/figures/` y `reports/tables/` con prefijo `01_` (sección 8).

## 3. Fundamento teórico para la sustentación

### 3.1 Exploración y duplicados (bloques A y B)

**Desbalance.** Con $N=9075$ imágenes y $N_c$ imágenes en la clase $c$, la proporción es $\pi_c=N_c/N$. La razón entre la clase mayor y la menor es $1670/525\approx 3.18$. Un clasificador que siempre responde "rui" obtiene accuracy $\max_c \pi_c = 1670/9075\approx 0.184$: esa es la línea base trivial, y la razón por la que se reporta F1 macro (promedia el F1 de cada clase con igual peso). El peso balanceado del contrato es

$$w_c=\frac{N}{K\,N_c},$$

con $K=9$. Con los conteos totales, $w_{\text{kalibaush}}\approx 1.92$ y $w_{\text{rui}}\approx 0.60$; en el entrenamiento se calcula con los conteos de train, no con los totales.

**Duplicados exactos (MD5).** MD5 resume los bytes del archivo en 128 bits. Archivos con bytes idénticos tienen el mismo hash, y la probabilidad de colisión accidental entre $9075$ archivos es despreciable. Limitación: una copia recomprimida, redimensionada o recortada tiene bytes distintos y no se detecta.

**Casi duplicados (dHash).** Se pasa la imagen a gris, se redimensiona a $9$ columnas por $8$ filas y se compara cada píxel con su vecino derecho:

$$h_{r,c}=\mathbb{1}\big[I(r,c+1)>I(r,c)\big],\qquad r,c\in\{0,\dots,7\},$$

lo que da un vector $h\in\{0,1\}^{64}$. Como codifica gradientes y no intensidades absolutas, es robusto a recompresión, cambios leves de brillo y reescalado. La similitud se mide con la distancia de Hamming

$$d_H(h,h')=\sum_{k=1}^{64}\big(h_k\oplus h'_k\big),$$

y el contrato declara casi duplicados a los pares con $d_H\le 5$ (menos del $8\%$ de bits distintos). Para compararlos todos de forma vectorizada conviene pasar a $s=2h-1\in\{-1,1\}^{64}$, porque entonces

$$d_H(h,h')=\frac{64-s^{\top}s'}{2},$$

y todas las distancias de un bloque salen de un producto matricial. El número de pares es $\binom{9075}{2}=41\,173\,275$: factible por bloques, inviable con un doble ciclo en Python puro.

**Atajos (shortcut learning).** Si una variable ajena al pez (brillo del fondo, tipo de superficie, marca de agua) cambia sistemáticamente con la clase, entonces $P(y\mid \text{fondo})\neq P(y)$ y la CNN puede aprender esa variable en lugar de la forma o el color del pez. La exploración no lo prueba ni lo descarta del todo, pero detecta señales de alarma.

### 3.2 Partición sin fuga de datos (bloque C)

**Test independiente.** El objetivo es estimar el riesgo real del modelo $f$,

$$R(f)=\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[\ell(f(x),y)\big],$$

mediante el promedio sobre el conjunto de test $T$,

$$\hat{R}_T(f)=\frac{1}{|T|}\sum_{i\in T}\ell\big(f(x_i),y_i\big).$$

$\hat{R}_T$ es un estimador insesgado de $R$ solo si los ejemplos de $T$ no intervinieron en la construcción de $f$. Si una casi copia $x_j\approx x_i$ de una imagen de test está en train, la red la memorizó, $f(x_i)\approx f(x_j)$ es correcta "por recuerdo" y $\hat{R}_T$ queda sesgado hacia abajo: la métrica sale optimista. Eso es fuga de datos (data leakage), y una partición aleatoria por imagen la produce siempre que haya duplicados.

**Grupos.** Se define un grafo $G=(V,E)$ cuyos vértices son las imágenes y donde hay arista entre dos imágenes si tienen el mismo MD5 o si $d_H\le 5$. Cada componente conexa es un grupo, con su `group_id`. Usar componentes conexas aplica la clausura transitiva: si $A\sim B$ y $B\sim C$, las tres van juntas aunque $d_H(A,C)>5$. La restricción de la partición es

$$\mathcal{G}_{\text{train}}\cap\mathcal{G}_{\text{val}}=\mathcal{G}_{\text{train}}\cap\mathcal{G}_{\text{test}}=\mathcal{G}_{\text{val}}\cap\mathcal{G}_{\text{test}}=\varnothing,$$

donde $\mathcal{G}_s$ es el conjunto de `group_id` presentes en el split $s$.

**Estratificación.** Se busca que en cada split $s$ la proporción de cada clase sea la global:

$$\frac{N_c^{(s)}}{N^{(s)}}\approx \pi_c=\frac{N_c}{N}.$$

Sin estratificar, el número de imágenes de una clase en test sigue aproximadamente una hipergeométrica. Para kalibaush ($\pi\approx 0.058$) en un test de 1362 imágenes, la media es $\approx 79$ y la desviación estándar $\approx 8$: una variación de $\pm 10\%$ solo por azar, que distorsiona el F1 de las clases pequeñas.

**Partición en dos pasos.** Primero se separa $70\%$ para train y $30\%$ temporal; luego el temporal se divide a la mitad en val y test: $0.30\times 0.5=0.15$. Ambos pasos se hacen sobre la tabla de grupos (una fila por grupo), estratificando por la clase del grupo, y después se propaga el split a cada imagen. Como los grupos tienen tamaños distintos, las proporciones por imagen son aproximadas. Resultado esperado: train $\approx 6352$, val $\approx 1361$, test $\approx 1362$.

### 3.3 Máscaras con OpenCV (bloque D)

**Espacios de color.** OpenCV lee en BGR; todo empieza convirtiendo a RGB. Canales candidatos:

- Gris, $Y=0.299R+0.587G+0.114B$. Separa bien si el pez es más oscuro que el fondo, pero también captura sombras y confunde peces claros o plateados.
- Saturación de HSV, $S=\dfrac{\max(R,G,B)-\min(R,G,B)}{\max(R,G,B)}$ (escalada a $[0,255]$ en `uint8`). Un fondo gris o blanco y una sombra gris tienen $S\approx 0$, así que la sombra queda fuera. Riesgo: peces de baja saturación (posiblemente ilish, de tono plateado; verificarlo) y reflejos especulares.
- Lab: $L^*$ es luminosidad; $a^*$ y $b^*$ son cromáticos. En `uint8` de OpenCV están desplazados en 128, de modo que un gris neutro tiene $a\approx b\approx 128$. *Decisión sugerida:* probar también el croma $C=\sqrt{(a-128)^2+(b-128)^2}$, la distancia al gris en ambos ejes a la vez.

La elección se hace con evidencia. Para cada imagen de la muestra y cada canal se calcula la separabilidad de Otsu

$$\eta=\frac{\sigma_B^2(t^*)}{\sigma_T^2}\in[0,1],$$

que vale 1 cuando el histograma son dos picos perfectamente separados. Se promedia $\eta$ por clase y se elige el canal que maximiza el peor promedio por clase (criterio maximin), porque una máscara mala en una clase entera es peor que una algo menos buena en todas. La decisión se confirma visualmente.

**Desenfoque Gaussiano.** Se convoluciona el canal con

$$G_\sigma(x,y)=\frac{1}{2\pi\sigma^2}\exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right).$$

Atenúa la textura de la superficie, el ruido del sensor y los bloques del JPEG; el histograma queda más bimodal y Otsu más estable. Con `cv2.GaussianBlur`, núcleo $5\times 5$ y `sigmaX=0`, OpenCV calcula $\sigma=0.3\big(\tfrac{k-1}{2}-1\big)+0.8=1.1$.

**Umbral de Otsu.** Sea $p_i$ la frecuencia relativa del nivel $i\in\{0,\dots,255\}$. Para un umbral $t$:

$$\omega_0(t)=\sum_{i=0}^{t}p_i,\qquad \omega_1(t)=1-\omega_0(t),\qquad \mu_0(t)=\frac{1}{\omega_0(t)}\sum_{i=0}^{t}i\,p_i,\qquad \mu_1(t)=\frac{1}{\omega_1(t)}\sum_{i=t+1}^{255}i\,p_i.$$

Otsu elige el umbral que maximiza la varianza entre clases:

$$t^*=\arg\max_{t}\ \sigma_B^2(t),\qquad \sigma_B^2(t)=\omega_0(t)\,\omega_1(t)\,\big(\mu_0(t)-\mu_1(t)\big)^2.$$

Como $\sigma_T^2=\sigma_W^2(t)+\sigma_B^2(t)$ no depende de $t$, maximizar $\sigma_B^2$ equivale a minimizar la varianza intraclase $\sigma_W^2=\omega_0\sigma_0^2+\omega_1\sigma_1^2$. Supuesto implícito: histograma bimodal. Falla cuando el pez ocupa una fracción muy pequeña de la imagen o tiene un tono parecido al del fondo.

**Polaridad.** La convención final es pez = 255. En gris el pez es más oscuro y se usa `cv2.THRESH_BINARY_INV`; en $S$ o en croma tiene valores mayores y se usa `cv2.THRESH_BINARY`. *Decisión sugerida:* regla automática: si más de la mitad de los píxeles del borde de la imagen quedaron en 255, la máscara está invertida y se invierte.

**Morfología.** Sea $A$ el conjunto de píxeles en 255 y $B$ el elemento estructurante (un disco discreto):

$$A\ominus B=\{z:\ B_z\subseteq A\},\qquad A\oplus B=\{z:\ (\hat{B})_z\cap A\neq\varnothing\},$$

con $B_z$ el trasladado de $B$ a $z$ y $\hat{B}$ su reflejado. Apertura y cierre son composiciones:

$$A\circ B=(A\ominus B)\oplus B,\qquad A\bullet B=(A\oplus B)\ominus B.$$

La apertura elimina objetos y salientes más pequeños que $B$ y cumple $A\circ B\subseteq A$; el cierre rellena huecos y grietas más pequeños que $B$ y cumple $A\subseteq A\bullet B$; ambas son idempotentes. El núcleo elíptico (`cv2.getStructuringElement` con `cv2.MORPH_ELLIPSE`) es aproximadamente isótropo y no deja esquinas artificiales en el contorno curvo del pez. Puntos de partida: apertura con núcleo $5\times 5$ y cierre entre $9\times 9$ y $15\times 15$, ajustados en la muestra. Una apertura grande se come aletas y cola; un cierre grande pega el pez a la sombra o a la marca de agua.

**Componentes conexas.** Dos píxeles de $A$ están conectados si existe un camino entre ellos dentro de $A$ con píxeles consecutivos en 8-vecindad. `cv2.connectedComponentsWithStats` con `connectivity=8` etiqueta cada componente y devuelve su área en `cv2.CC_STAT_AREA`; la etiqueta 0 es el fondo y se ignora. Conservar solo la de mayor área elimina la marca de agua de la esquina inferior izquierda (región separada del pez) y el ruido residual. El número de componentes antes de filtrar se guarda como `n_componentes`.

**Relleno de huecos.** Un hueco es una componente de $A^c$ que no toca el borde de la imagen:

$$\operatorname{fill}(A)=A\ \cup\ \{\,H:\ H \text{ componente conexa de } A^c,\ H\cap\partial\Omega=\varnothing\,\},$$

con $\Omega$ el dominio de la imagen. Se implementa dibujando el contorno externo (`cv2.findContours` con `cv2.RETR_EXTERNAL`) relleno con `cv2.drawContours` y grosor $-1$; la alternativa, `cv2.floodFill` desde una esquina sobre la máscara invertida, falla si el pez toca esa esquina. Los huecos aparecen por reflejos, el ojo o el vientre claro.

**GrabCut (refinamiento opcional).** Modela el color del pez y del fondo con dos mezclas de Gaussianas en RGB y busca la etiqueta $\alpha_m\in\{0,1\}$ de cada píxel que minimiza

$$E(\alpha)=\sum_{m}U_m(\alpha_m)+\gamma\sum_{(m,n)\in\mathcal{N}}\mathbb{1}[\alpha_m\neq\alpha_n]\,e^{-\beta\lVert z_m-z_n\rVert^2},$$

donde $U_m$ es menos el logaritmo de la verosimilitud del color $z_m$ bajo el modelo de su etiqueta (término de datos) y el segundo término penaliza cortar entre vecinos de color parecido (término de suavidad). La energía se representa como un grafo: un nodo por píxel, aristas entre vecinos con el peso de suavidad, y dos nodos terminales (pez y fondo) unidos a cada píxel con el peso de datos. El corte de capacidad mínima que separa los terminales es la segmentación de energía mínima (teorema de flujo máximo y corte mínimo). GrabCut alterna: reestimar las mezclas con la segmentación actual y recalcular el corte.

**Calidad de una máscara.** La fracción de área es

$$a=\frac{1}{HW}\sum_{u=1}^{H}\sum_{v=1}^{W}\mathbb{1}\big[M(u,v)=255\big],$$

y contra una máscara manual se usan las fórmulas del contrato:

$$\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|},\qquad \mathrm{Dice}(A,B)=\frac{2|A\cap B|}{|A|+|B|}.$$

### 3.4 Preprocesado y caché (bloque E)

Reducir de $500\times 500$ a $224$ o $256$ es submuestreo; `cv2.INTER_AREA` promedia sobre el área cubierta por cada píxel de salida y evita el aliasing que produciría tomar píxeles aislados. Una máscara, en cambio, es una etiqueta por píxel: interpolar crea valores intermedios que no son ni pez ni fondo, por eso se usa `cv2.INTER_NEAREST`, que solo reutiliza valores existentes y se binariza después. Como imagen y máscara se reducen con métodos distintos, puede haber una desalineación subpíxel en el borde; es una limitación aceptada y se declara.

Guardar `uint8` ocupa un byte por valor: train CNN $\approx 6352\cdot 224\cdot 224\cdot 3\approx 0.96$ GB (val y test $\approx 0.2$ GB cada uno); train U-Net, con imagen y máscara, $\le 6352\cdot(256\cdot 256\cdot 3+256\cdot 256)\approx 1.67$ GB. En `float32` serían cuatro veces más, y la capa `Rescaling(1/255)` dentro del modelo hace imposible olvidar la normalización en la demo.

## 4. Pipeline paso a paso

Formato del cuaderno: docstring de encabezado (Autor, Código, Materia, Tema); una sola celda de imports con comentario en línea por import (`os`, `hashlib`, `zipfile`, `random`, `json`, `numpy`, `pandas`, `cv2`, `matplotlib.pyplot`, `keras`, `sklearn.model_selection.train_test_split`, `sklearn.pipeline.make_pipeline`, `sklearn.preprocessing.StandardScaler`, `sklearn.linear_model.LogisticRegression`, `sklearn.model_selection.StratifiedKFold` y `cross_val_score`); celdas markdown cortas en primera persona plural ("Ahora vamos a ..."); una celda de código por paso, con comentarios en línea. Como no hay `src/`, todo vive en celdas de este cuaderno. Cada bloque abre releyendo desde Drive los archivos del bloque anterior, de modo que una sesión nueva puede retomar en cualquier bloque sin repetir los anteriores. En el bloque D, la marca entre paréntesis (D2 o D3) indica el día en que se ejecuta el paso.

### Bloque A — Configuración y exploración (D1)

1. **A1. Configuración.** Celda de constantes comunes, copiada idéntica en 02, 03 y 04: `SEED = 42` fijada en `random.seed`, `np.random.seed` y `keras.utils.set_random_seed`; tamaños de entrada (224 y 256); orden de clases; rutas de Drive y de `/content`; umbral de duplicados ($d_H\le 5$); umbral de máscara por defecto $0.5$; interpolaciones fijadas para el caché (`cv2.INTER_AREA` para imágenes y `cv2.INTER_NEAREST` para máscaras). Montar Drive con `google.colab.drive.mount`. *Verificar:* Drive montado y el zip visible.
2. **A2. Copia y descompresión.** Copiar el zip de `MyDrive/sylfish/data/raw/` a `/content/` y descomprimir con `zipfile.ZipFile`. *Verificar:* existen `classes.txt` y 9 subcarpetas en `images/`; el orden leído de `classes.txt` coincide con el contrato.
3. **A3. Listado de archivos.** Recorrer cada carpeta de clase con listados ordenados (`sorted`), aceptar solo `.jpg`/`.jpeg` y construir un `DataFrame` con `filepath`, `clase` y `label` (índice en `classes.txt`). *Verificar:* total 9075 y conteos por clase idénticos a los de la especificación; todo archivo extra (oculto, carpeta `__MACOSX`) se reporta y se excluye.
4. **A4. Conteos y barras.** `value_counts` por clase y `plt.bar`, anotando $N_c$ y $\pi_c$ sobre cada barra. *Verificar:* razón máx/mín $\approx 3.18$.
5. **A5. Grilla de muestras.** Nueve filas (una por clase) por seis columnas, elegidas al azar con `SEED`, en RGB. Observar tono y textura del fondo, sombras, escala del pez, orientación, objetos ajenos (manos, bandejas, etiquetas) y marcas de agua.

### Bloque B — Inventario y duplicados (D1)

6. **B1. Pasada única sobre todas las imágenes.** Un solo ciclo que, por imagen: lee los bytes y calcula `hashlib.md5`; decodifica con `cv2.imread` e `cv2.IMREAD_UNCHANGED` para conocer el número real de canales; registra alto, ancho y canales (o `legible = False` si `imread` devuelve `None`); convierte a RGB con `cv2.cvtColor` y `cv2.COLOR_BGR2RGB`; calcula el dHash (gris con `cv2.COLOR_RGB2GRAY`, `cv2.resize` a 9 de ancho por 8 de alto con `cv2.INTER_AREA`, comparación de columnas vecinas, empaquetado a hexadecimal); y calcula las estadísticas de fondo del paso B3. Leer 9075 archivos una vez ahorra minutos frente a varias pasadas. *Verificar:* ninguna fila con faltantes salvo las ilegibles, que se listan aparte.
7. **B2. Resolución y modo.** Tabla de combinaciones únicas (`alto`, `ancho`, `canales`) con su conteo. *Verificar:* si hay una sola fila $(500,500,3)$ queda confirmada la uniformidad; si aparecen tamaños no cuadrados, imágenes en gris (2 dimensiones) o con canal alfa (4 canales), se listan por clase y se decide su tratamiento en E1.
8. **B3. Estadísticas de color y fondo por clase.** Banda perimetral de ancho fijo (por ejemplo 25 píxeles, ajustable tras ver la grilla), que en estas fotos es casi toda fondo. Media de $L^*$ (`cv2.COLOR_RGB2LAB`) y de $S$ (`cv2.COLOR_RGB2HSV`) como `L_borde` y `S_borde`; diagramas de caja por clase con `plt.boxplot`. Diagnóstico opcional: `LogisticRegression` dentro de `make_pipeline` con `StandardScaler`, evaluada con `cross_val_score` y `StratifiedKFold` (5 pliegues, `shuffle=True`, `random_state=SEED`) usando solo esas dos variables. *Verificar:* si la accuracy media supera claramente $0.184$, el fondo contiene información de la clase y se documenta como riesgo de atajo para 02.
9. **B4. Marca de agua.** Recortar la región inferior izquierda (por ejemplo $40\%$ del ancho por $12\%$ del alto; ajustar con la foto conocida) y calcular una puntuación de texto: densidad de bordes de `cv2.Canny` en ese recorte menos la del recorte simétrico inferior derecho, que sirve de control. Ordenar por puntuación e inspeccionar en grillas las 60 mayores y 60 al azar; marcar `marca_agua` como si/no en las revisadas y `sin_revisar` en el resto. *Verificar:* tabla de imágenes con marca por clase; si se concentra en pocas clases es un atajo potencial que se reporta a 02.
10. **B5. Duplicados exactos.** Agrupar por `md5` y quedarse con los grupos de tamaño mayor que 1. *Verificar:* conteo de grupos, imágenes implicadas y grupos que mezclan clases distintas.
11. **B6. Casi duplicados.** Convertir los dHash a una matriz $9075\times 64$ en $\{-1,1\}$ y calcular distancias por bloques de filas (por ejemplo 1000), contando cada par una vez (triángulo superior, sin diagonal). Registrar los pares con $d_H\le 5$ que no sean ya duplicados exactos; inspeccionar lado a lado una muestra, priorizando $d_H\in\{4,5\}$, para validar el umbral. Guardar `pares_duplicados.csv` (los pares MD5 con `tipo = md5` y distancia 0). *Verificar:* histograma de distancias mínimas por imagen.
12. **B7. Guardado y conclusiones.** Escribir `inventario.csv` y cerrar con una celda markdown de conclusiones accionables: tamaño de entrada y tratamiento de imágenes no cuadradas; desbalance, `class_weight` y F1 macro; número de grupos de duplicados; riesgo de atajos (fondo y marca de agua); observaciones para elegir el canal de las máscaras.

### Bloque C — Partición (D1, publicar `splits.csv` por la noche)

13. **C1. Carga de insumos.** Leer `inventario.csv`, `pares_duplicados.csv` y `classes.txt`. Las imágenes ilegibles se excluyen de la partición y se listan. *Verificar:* filas esperadas y `label` coherente con el orden de `classes.txt`.
14. **C2. Construcción de `group_id`.** Unión-búsqueda (union-find) en Python: cada imagen empieza siendo su propio representante; por cada par de `pares_duplicados.csv` se unen los representantes; al final se renumeran como enteros consecutivos. Las imágenes sin pares forman grupos de tamaño 1. *Verificar:* número de grupos, tamaño máximo e histograma de tamaños; sin duplicados, 9075 grupos.
15. **C3. Conflictos de etiqueta.** Buscar grupos con más de una clase. *Decisión sugerida (confirmar con el equipo):* excluirlos de `splits.csv` y listarlos en el informe como ruido de etiqueta, porque no existe etiqueta correcta para ambas copias. *Verificar:* cuántas imágenes se excluyen (se espera muy pocas o ninguna).
16. **C4. Tabla de grupos.** Una fila por grupo con `group_id`, `clase` y `tamaño`. *Verificar:* la suma de tamaños es el número de imágenes retenidas.
17. **C5. Primer corte.** `train_test_split` sobre los `group_id` con `test_size=0.30`, `stratify` igual a la clase del grupo, `shuffle=True` y `random_state=SEED`.
18. **C6. Segundo corte.** Sobre el $30\%$ temporal, `train_test_split` con `test_size=0.50`, misma estratificación y misma semilla. Propagar el split a cada imagen con un `merge` por `group_id`.
19. **C7. Verificaciones.** Tabla de conteos y porcentajes por clase y split con `pd.crosstab`; desviación máxima de cada proporción respecto a $\pi_c$; totales por split; intersecciones vacías de `group_id` entre los tres pares de splits con operaciones de conjuntos; `filepath` sin repetidos; todos los archivos existen en disco. Guardar la figura de proporciones y la de tamaños de grupo.
20. **C8. Publicación de `splits.csv`.** `to_csv` con `index=False` y las columnas en el orden del contrato. *Decisión sugerida:* calcular el MD5 del propio `splits.csv`, anunciarlo al equipo y que cada integrante lo verifique al cargarlo. Desde este momento el archivo queda congelado: si hubiera que corregirlo, se publica con versión nueva y se avisa a los cinco.

### Bloque D — Máscaras y control de calidad (D2 y D3)

Las celdas D2 a D6 se escriben autocontenidas (una imagen RGB entra, una máscara sale) para que quien necesite el procedimiento, por ejemplo ML clásico, pueda copiarlas; quien solo necesite las máscaras usa los PNG.

21. **D1. Muestra de calibración (D2).** 5 imágenes por clase tomadas solo de train con `SEED` (45 en total), más la foto conocida con marca de agua. *Verificar:* las 9 clases representadas.
22. **D2. Comparación de canales (D2).** Para cada imagen de la muestra: RGB, `cv2.GaussianBlur`, y los canales gris (`cv2.COLOR_RGB2GRAY`), $S$ (`cv2.COLOR_RGB2HSV`), $L^*$, $a^*$, $b^*$ (`cv2.COLOR_RGB2LAB`) y croma. Grilla de canales por imagen y tabla de $\eta$ medio por clase y canal. *Verificar:* el canal elegido no tiene ninguna clase con $\eta$ claramente bajo; si la hay, se anota como clase de riesgo.
23. **D3. Umbral y polaridad (D2).** `cv2.threshold` con `cv2.THRESH_BINARY` o `cv2.THRESH_BINARY_INV` combinado con `cv2.THRESH_OTSU` sobre el canal elegido en `uint8`; guardar el $t^*$ devuelto y aplicar la regla del borde. *Verificar:* el fondo queda en 0 en toda la muestra.
24. **D4. Morfología (D2).** `cv2.morphologyEx` con `cv2.MORPH_OPEN` y luego `cv2.MORPH_CLOSE`, núcleos elípticos. *Verificar:* aletas y cola conservadas; motas eliminadas.
25. **D5. Componente principal (D2).** `cv2.connectedComponentsWithStats`; registrar `n_componentes`; conservar la de mayor área. *Decisión sugerida:* si la segunda componente tiene al menos el $30\%$ del área de la primera, marcar para revisar (el pez pudo partirse en dos). *Verificar:* la marca de agua desaparece en la foto conocida y en las detectadas en B4.
26. **D6. Relleno de huecos (D2).** Contorno externo relleno. *Verificar:* sin huecos interiores en la muestra.
27. **D7. Figura del pipeline (D2).** Para 3 imágenes de clases distintas, una fila por imagen con las etapas: original, canal, desenfoque, Otsu, apertura, cierre, componente, relleno, superposición. Es la figura central de la sustentación.
28. **D8. Versión v1 y revisión cruzada (D2).** Correr sobre 20 imágenes por clase de train, calcular las métricas del paso D10 y mostrar grillas de superposiciones por clase a Compañero DL e Integrantes ML para revisión cruzada. Ajustar parámetros y fijar v2 en `mask_params.json`.
29. **D9. Generación completa (D3).** Recorrer todas las imágenes de `splits.csv` y escribir cada máscara con `cv2.imwrite` en PNG, en `/content/masks/`. *Verificar:* conteo de archivos igual al de imágenes, forma idéntica a la original y `np.unique` exactamente $\{0,255\}$ en una muestra releída del disco.
30. **D10. Métricas y estado automático (D3).** Por máscara: `area_frac` $=a$, `n_componentes`, `toca_borde` (algún píxel en 255 en la primera o última fila o columna). Estado `revisar` si $a<0.05$, $a>0.80$, `toca_borde` es verdadero o se activó la regla de D5; `ok` en caso contrario. *Verificar:* histograma de $a$ por clase y conteo de estados por clase.
31. **D11. Revisión manual y GrabCut (D3).** Grillas de las marcadas `revisar`, clase por clase. Cada una pasa a `ok` (falsa alarma, por ejemplo un pez que de verdad toca el borde), se reprocesa con GrabCut o pasa a `descartada`. GrabCut: `cv2.grabCut` con `cv2.GC_INIT_WITH_MASK` y 3 a 5 iteraciones, inicializado así: interior de la máscara erosionada como `cv2.GC_FGD`, resto de la máscara como `cv2.GC_PR_FGD`, un anillo exterior obtenido por dilatación como `cv2.GC_PR_BGD` y lo demás como `cv2.GC_BGD`; después se reaplican componente principal y relleno y se recalculan las métricas.
32. **D12. Auditoría de las ok (D3).** Grilla aleatoria de 16 máscaras ok por clase; contar fallos visibles $f$ y estimar la tasa de falsos ok $\hat{p}=f/16$ por clase. Si alguna clase supera, por ejemplo, $\hat{p}>0.1$, revisar sus reglas.
33. **D13. Conjunto oro (D3, opcional y recomendado).** *Decisión sugerida:* 10 imágenes por clase de val (90), elegidas con `SEED` antes de mirar sus máscaras, corregidas a mano partiendo de la pseudo-máscara en un editor de imágenes, o anotadas por polígono en una herramienta externa y rasterizadas con `cv2.fillPoly`. Calcular IoU y Dice de cada pseudo-máscara contra su versión oro; reportar media global y por clase.
34. **D14. Congelamiento (D3).** Escribir `mask_qc.csv` con las columnas del contrato y `mask_params.json`, comprimir las máscaras en `masks.zip`, copiar a Drive y anunciar. Al congelar no debe quedar nada en `revisar`; lo que quede se trata como no ok. Tabla final de estados por clase y por split.

### Bloque E — Preprocesado y caché `.npz`

Las celdas E2 y E3 dependen solo de `splits.csv` y se ejecutan el D1 por la noche, justo después del bloque C. Las celdas E4 a E6 requieren las máscaras congeladas y se ejecutan la noche del D3.

35. **E1. Verificación de insumos y regla de forma.** Releer `splits.csv` y comprobar su MD5 publicado. Decidir, a partir de B2, el tratamiento de imágenes no cuadradas o con canales distintos de 3 (ver sección 5). Los tamaños y el método de redimensionado quedan fijos y se documentan, porque 04 debe replicarlos idénticos.
36. **E2. Caché CNN (D1).** Para cada split, en el orden de `splits.csv`: leer con `cv2.imread`, convertir con `cv2.COLOR_BGR2RGB`, redimensionar a $224\times 224$ con `cv2.resize` e `interpolation=cv2.INTER_AREA`, y llenar un arreglo `uint8` preasignado. Guardar `X`, `y` y `filepath` con `np.savez` primero en `/content/` y luego copiar a `MyDrive/sylfish/data/processed/`.
37. **E3. Verificación CNN (D1).** Formas, `dtype` `uint8`, rango $[0,255]$, distribución de `y` igual a la tabla de C7, tamaño del archivo en Drive igual al de `/content`, y una grilla de 8 imágenes con su clase como título (colores naturales confirman RGB).
38. **E4. Caché U-Net (D3).** Cruzar `splits.csv` con `mask_qc.csv` por `filepath` y quedarse con `estado = ok`. Por imagen: RGB y `cv2.resize` a $256\times 256$ con `cv2.INTER_AREA`; máscara leída con `cv2.IMREAD_GRAYSCALE`, redimensionada con `cv2.INTER_NEAREST`, binarizada a $\{0,1\}$ y expandida a un canal con `np.expand_dims`. Guardar `X`, `M`, `y` y `filepath`.
39. **E5. Verificación U-Net (D3).** `np.unique(M)` es exactamente $\{0,1\}$; la media de `M` por imagen se parece a su `area_frac`; superposición de 6 pares al azar; conteo de imágenes ok por split y por clase; tamaños copiados a Drive verificados.
40. **E6. Cierre (D3).** Celda markdown con la tabla final de conteos por split y clase (CNN y U-Net), número de grupos, exclusiones, estados de máscara y el MD5 de `splits.csv`.

## 5. Decisiones de diseño y alternativas

**Exploración y duplicados**

- **Un solo cuaderno de cinco bloques**, en lugar de un cuaderno por etapa: las constantes, el orden de clases y la conversión BGR→RGB se fijan en un único lugar y no hay módulos compartidos que sincronizar.
- **Pasada única** en lugar de un ciclo por análisis: menos E/S y un solo punto donde se fija BGR→RGB. Por eso las estadísticas de fondo, la resolución y la marca de agua viven en el bloque B.
- **dHash** en lugar de pHash o SSIM: dHash se implementa con `cv2.resize` y comparaciones, y su distancia se vectoriza. pHash (DCT con `cv2.dct`) es una alternativa válida y algo más robusta; SSIM por pares a resolución completa es inviable sobre 41 millones de pares.
- **Umbral $d_H\le 5$** fijado por contrato; se valida visualmente, no se ajusta a conveniencia.
- **Explorar todo el dataset antes de partir** es aceptable porque solo se describen datos; nada de lo calculado aquí elige hiperparámetros con test. La regresión logística de B3 es un diagnóstico, no un resultado del informe.
- **dHash no detecta copias espejadas.** *Decisión sugerida (confirmar con el equipo):* calcular también el dHash de la imagen volteada con `cv2.flip` y compararlo contra los demás; cuesta una pasada más de productos matriciales.

**Partición**

- **Agrupar en vez de borrar duplicados.** Se conservan todas las imágenes y solo se impide que copias queden en splits distintos; borrar también evita la fuga, pero pierde datos y obliga a elegir cuál copia se queda.
- **Dos llamadas a `train_test_split` sobre grupos** frente a `StratifiedGroupKFold` con `n_splits=20`, `shuffle=True` y `random_state=SEED`, asignando 3 pliegues a test, 3 a val y 14 a train ($3/20=0.15$). Es igualmente válida; se eligió la primera porque se explica en una frase.
- **70/15/15.** Validación suficiente para early stopping, selección de modelos y ajuste del umbral de máscara; test con al menos unas 79 imágenes de la clase más pequeña.

**Máscaras**

- **OpenCV clásico en vez de anotar a mano o usar un modelo preentrenado de segmentación.** Anotar 9075 imágenes es inviable en dos días; un modelo externo saldría del stack del curso. El fondo claro y liso hace que la umbralización sea suficiente en la mayoría de los casos.
- **Otsu global en vez de umbral adaptativo** (`cv2.adaptiveThreshold`): el fondo es uniforme y un umbral por imagen basta; el adaptativo responde a la textura local y fragmenta el interior del pez.
- **Canal elegido por $\eta$ con criterio maximin** en vez de "el que se vea mejor": decisión reproducible y defendible.
- **Componente más grande** en vez de filtrar por posición la marca de agua: es general y limpia cualquier objeto suelto. Supone un solo pez por imagen; si B detectó fotos con varios peces, se marcan `revisar`.
- **GrabCut solo sobre las marcadas.** Es mucho más lento que el resto del pipeline (medir su tiempo por imagen en la muestra); si alcanza el tiempo puede probarse sobre todo el dataset y compararse contra el oro.
- **Parámetros calibrados en train, oro en val.** Nada de test interviene en el diseño de las máscaras.

**Caché**

- **Caché redimensionada y sin normalizar.** Evita decodificar JPG en cada época y fija `uint8` con `Rescaling(1/255)` dentro del modelo. A cambio, fija el método de redimensionado, que 04 debe replicar idéntico; por eso la prueba de paridad de la demo contra la caché es obligatoria. El aumento de datos se hace en línea dentro del modelo, no en la caché.
- **Imágenes no cuadradas** (solo si B2 las encontró). *Decisión sugerida:* rellenar a cuadrado con `cv2.copyMakeBorder` y `cv2.BORDER_REPLICATE` antes de redimensionar, para no deformar el pez, aplicar el mismo relleno a la máscara con valor 0 y replicarlo en la demo. Las imágenes en gris o con canal alfa se convierten a 3 canales RGB.
- **`np.savez` sin compresión** para carga rápida; `np.savez_compressed` si el espacio de Drive es limitado.
- **CNN con todas las imágenes de `splits.csv`; U-Net solo con las ok.** Así la CNN no depende de la calidad de las máscaras.

## 6. Errores comunes y trampas

**Bloques A y B**

- Usar el orden de `os.listdir` como orden de clases: no está garantizado; el índice sale de `classes.txt`.
- Contar archivos basura del zip (`__MACOSX`, `.DS_Store`) y obtener un total distinto de 9075.
- Calcular el MD5 sobre el arreglo decodificado en vez de sobre los bytes del archivo: el contrato dice bytes.
- Invertir el orden en `cv2.resize`: el tamaño es (ancho, alto), es decir (9, 8) para dHash, no (8, 9).
- Construir la matriz completa de distancias $9075\times 9075$ en `float64` (unos 660 MB) y agotar la RAM; trabajar por bloques.
- Leer imágenes directamente desde Drive: cada lectura es una llamada de red y el ciclo tarda decenas de minutos. Siempre desde `/content`.
- Graficar arreglos de OpenCV sin convertir a RGB: los peces "cambian de color" y se sacan conclusiones falsas.
- Declarar "no hay duplicados" sin haber mirado una muestra de pares cercanos al umbral, o ignorar los duplicados entre clases distintas.

**Bloque C**

- Partir por imagen y agrupar después: el daño ya está hecho.
- Usar `test_size=0.15` en el segundo corte: produce $25.5\%$ y $4.5\%$ en vez de $15\%$ y $15\%$.
- Olvidar `random_state=SEED` en alguno de los dos cortes: la partición deja de ser reproducible.
- Guardar `filepath` absoluta (`/content/...`): rompe el contrato.
- Regenerar `splits.csv` después de que ML clásico haya empezado: sus `features.csv` quedan desalineados.

**Bloque D**

- No convertir de BGR a RGB y elegir el canal con colores cruzados (`cv2.COLOR_BGR2HSV` frente a `cv2.COLOR_RGB2HSV` cambia el tono).
- Pasar a `cv2.threshold` con Otsu una imagen que no es `uint8` de un solo canal.
- Guardar máscaras en JPG (la compresión crea valores intermedios en el borde; siempre PNG) o en $\{0,1\}$ en disco (el contrato dice $\{0,255\}$; la conversión a $\{0,1\}$ ocurre en E4).
- Cierre tan grande que une el pez con su sombra o con la marca de agua, y la componente principal arrastra la marca.
- Usar `cv2.floodFill` desde la esquina $(0,0)$ cuando el pez toca esa esquina: se rellena todo.
- Ajustar parámetros mirando una sola clase: los peces plateados y los oscuros se comportan distinto.
- Confundir `n_componentes` antes y después de filtrar (después siempre vale 1).
- Escribir 9075 PNG directamente en Drive.
- Cambiar parámetros después de congelar sin regenerar todo y sin avisar: la U-Net y ML clásico quedan con versiones mezcladas.

**Bloque E**

- Barajar `X` sin barajar `y` o `filepath` al construir la caché.
- Guardar la caché en `float32` o ya dividida entre 255: viola el contrato y la demo aplicaría doble normalización.
- Redimensionar máscaras con interpolación bilineal (aparecen valores intermedios) o dejarlas en $\{0,255\}$ (la BCE con objetivos de 255 no tiene sentido).
- Abrir el `.npz` de test "solo para mirar" antes del D7.
- Leer el `.npz` desde Drive justo después de escribirlo sin verificar que la copia terminó.

**Transversales**

- Ejecutar el bloque D antes de publicar `splits.csv`, o el bloque E4 antes de congelar las máscaras: las muestras de calibración y la caché quedarían sobre insumos que cambian.
- Modificar este cuaderno después de publicar contratos sin avisar en el canal; solo Jairo lo edita y los cambios que afecten contratos se anuncian.

## 7. Criterio de aceptación

**Bloque A**

- [ ] Constantes comunes definidas y 9075 imágenes con conteos por clase coherentes con la especificación.
- [ ] Gráfico de barras y grilla de muestras por clase guardados.

**Bloque B**

- [ ] Pasada única sobre TODAS las imágenes; tabla de resoluciones y canales sobre todas, no sobre una muestra; ilegibles listadas.
- [ ] Diagramas de caja de `L_borde` y `S_borde` por clase, con una conclusión escrita sobre riesgo de atajo.
- [ ] Imágenes con marca de agua por clase (al menos en la muestra revisada) y método usado.
- [ ] Duplicados exactos y casi duplicados contados, con inspección visual del umbral; duplicados entre clases distintas listados (o declarado que no hay).
- [ ] `inventario.csv` y `pares_duplicados.csv` en `MyDrive/sylfish/data/splits/` y celda final de conclusiones.

**Bloque C**

- [ ] `group_id` por componentes conexas de MD5 y dHash; conflictos de etiqueta resueltos y documentados.
- [ ] Desviación máxima por clase respecto a $\pi_c$ menor que un punto porcentual; totales cercanos a 6352 / 1361 / 1362 (la diferencia se explica por grupos o exclusiones).
- [ ] Intersecciones de `group_id` entre splits verificadas vacías.
- [ ] `splits.csv` publicado el D1 con columnas exactas y MD5 anunciado al equipo.

**Bloque D**

- [ ] Tabla de $\eta$ por clase y canal con el canal justificado; figura del pipeline para al menos 3 clases.
- [ ] Marca de agua eliminada en la foto conocida y en las detectadas en B4.
- [ ] Máscaras de todas las imágenes de `splits.csv` en PNG, 1 canal, $\{0,255\}$, resolución original.
- [ ] `mask_qc.csv` con las 5 columnas del contrato y ninguna fila en `revisar`; tabla de estados por clase y por split; histograma de `area_frac`.
- [ ] Auditoría de ok con $\hat{p}$ por clase; IoU medio contra el oro, global y por clase (si se hizo el oro).
- [ ] `mask_params.json` y `masks.zip` en Drive el D3, anunciados al equipo.

**Bloque E**

- [ ] `cnn_224_{train,val,test}.npz` con formas, `dtype`, rango y distribución verificados (D1).
- [ ] `unet_256_{train,val,test}.npz` (D3) con `M` en $\{0,1\}$ y solo máscaras ok.
- [ ] Archivos de Drive con el mismo tamaño que los de `/content`; `.npz` de test sin abrir para otro fin.

## 8. Artefactos que debe dejar

**En Drive (`MyDrive/sylfish/`)**

- `data/splits/inventario.csv`, `data/splits/pares_duplicados.csv`, `data/splits/splits.csv`
- `data/masks.zip`, `data/masks/mask_qc.csv`, `data/masks/mask_params.json` y, opcional, `data/masks_oro/`
- `data/processed/cnn_224_train.npz`, `cnn_224_val.npz`, `cnn_224_test.npz`
- `data/processed/unet_256_train.npz`, `unet_256_val.npz`, `unet_256_test.npz`

**Figuras (`reports/figures/`)**

- Bloque A y B: `01_conteos_por_clase.png`, `01_grilla_muestras.png`, `01_fondo_por_clase.png`, `01_marca_agua_ejemplos.png`, `01_hist_dhash.png`, `01_pares_duplicados.png`
- Bloque C: `01_proporciones_por_split.png`, `01_tamanos_de_grupo.png`
- Bloque D: `01_comparacion_canales.png` (con la tabla de $\eta$), `01_pipeline_etapas.png`, `01_hist_area_frac.png`, `01_grilla_qc_<clase>.png` (una por clase), `01_casos_descartados.png`

**Tablas (`reports/tables/`)**, una por cifra del informe. Los nombres son exactos y los demás documentos los citan tal cual:

- `01_resoluciones.csv`: combinaciones únicas de (`alto`, `ancho`, `canales`) con su conteo (B2).
- `01_marca_agua_por_clase.csv`: imágenes con marca de agua por clase y método usado (B4).
- `01_duplicados_resumen.csv`: grupos e imágenes implicadas por MD5 y por dHash, y grupos que mezclan clases (B5 y B6).
- `01_conteos_por_split.csv`: conteo y proporción por clase y split, número de grupos y exclusiones (C7).
- `01_eta_canales.csv`: $\eta$ medio por clase y canal (D2).
- `01_estados_por_clase.csv`: máscaras ok, revisar y descartada por clase y por split (D14).
- `01_auditoria_ok.csv`: fallos visibles $f$ y tasa $\hat{p}$ por clase (D12).
- `01_iou_oro.csv` (opcional): IoU y Dice de las pseudo-máscaras contra el conjunto oro, global y por clase (D13).

## 9. Preguntas probables de sustentación

1. **¿Por qué explorar antes de entrenar y qué encontraron que alimenta decisiones?** Porque tamaño de entrada, manejo del desbalance y partición dependen de hechos del dataset. Un modelo que no aprende nada y responde siempre "rui" obtiene $1670/9075\approx 0.184$, por eso reportamos F1 macro y usamos `class_weight`. Medimos fondo y marca de agua por clase para saber si la CNN podría clasificar por ellos y no por el pez.
2. **¿Por qué no basta con MD5 y cómo compararon 41 millones de pares?** MD5 solo detecta bytes idénticos; una foto recomprimida o redimensionada cambia el hash. dHash reduce a $9\times 8$ en gris y guarda 64 bits (cada píxel frente a su vecino derecho); con vectores en $\{-1,1\}^{64}$, $d_H=(64-s^{\top}s')/2$ y las distancias de un bloque salen de un producto matricial, por bloques para no agotar memoria.
3. **¿Qué es la fuga de datos y cómo la evitaron?** Que información del test influya en el entrenamiento. El riesgo eran fotos duplicadas o casi iguales repartidas entre train y test: las agrupamos con `group_id` (componentes conexas de MD5 y $d_H\le 5$), partimos a nivel de grupo y verificamos que ningún grupo aparece en dos splits. Los grupos con etiquetas contradictorias se excluyen.
4. **¿Por qué estratificar por grupo y por qué los totales no son exactamente 70/15/15?** Sin estratificar, una clase pequeña como kalibaush variaría unas 8 imágenes en test solo por azar. Partimos la tabla de grupos, estratificada por la clase del grupo, y luego asignamos el split a cada imagen; como los grupos tienen tamaños distintos y hay exclusiones, los totales son aproximados, con desviación por clase menor que un punto porcentual.
5. **¿Para qué sirve validación si ya hay test, y por qué todos usan el mismo `splits.csv`?** Validación se usa durante el desarrollo (early stopping, comparar modelos, umbral de la máscara); test se usa una sola vez, con modelos congelados, para que la métrica no esté optimizada sobre él. Con el mismo `splits.csv` la comparación entre CNN, U-Net y modelos clásicos se hace sobre las mismas imágenes.
6. **¿De dónde salen las máscaras si el dataset no las trae?** Las generamos con OpenCV: canal donde el pez contrasta con el fondo, Otsu (que maximiza $\sigma_B^2=\omega_0\omega_1(\mu_0-\mu_1)^2$), apertura y cierre con núcleo elíptico, componente conexa más grande (que elimina la marca de agua) y relleno de huecos. Son pseudo-máscaras y por eso llevan control de calidad.
7. **¿Por qué ese canal de color?** Medimos la separabilidad $\eta=\sigma_B^2/\sigma_T^2$ de cada canal por clase y elegimos el de mejor peor caso (maximin). La saturación, por ejemplo, deja fuera las sombras grises, que en escala de grises se confunden con el pez.
8. **¿Qué es GrabCut y para qué lo usaron?** Modela los colores de pez y fondo con mezclas de Gaussianas y plantea la segmentación como un corte mínimo en un grafo de píxeles: corta donde el color cambia y donde los modelos de color lo indican. Lo usamos solo para refinar las máscaras dudosas.
9. **¿Cómo saben que las máscaras son buenas, cuál es su limitación y para qué la U-Net si OpenCV ya segmenta?** Cada máscara tiene métricas con reglas de alarma, se revisaron las marcadas, se auditó una muestra de las ok por clase y se midió el IoU contra un conjunto oro hecho a mano. La limitación es que la U-Net aprende a imitar pseudo-máscaras, no una verdad experta. OpenCV depende de supuestos fuertes (fondo claro y liso, un pez); la U-Net aprende de los datos y entra en un solo paso de inferencia, aunque su generalización a fondos distintos se prueba en la demo.
10. **¿Por qué la caché es `uint8` y por qué la máscara se redimensiona con vecino más cercano?** `uint8` ocupa la cuarta parte que `float32` y la normalización vive dentro del modelo con `Rescaling(1/255)`, así la demo no puede olvidarla ni aplicarla dos veces. La máscara es una etiqueta por píxel: interpolar crea valores intermedios que no son pez ni fondo; vecino más cercano conserva solo valores originales, que luego se binarizan a $\{0,1\}$.
