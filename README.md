# Análisis de polarización política y desinformación en tuitter con técnicas de NLP

Este proyecto aborda el análisis de publicaciones políticas en tuitter en español, con dos objetivos principales:

1. **Clasificar la ideología** de las cuentas (izquierda / derecha) a partir del texto de sus tuits
.
2. **Explorar la relación entre desinformación y polarización**, utilizando un enfoque léxico sencillo sobre el mismo corpus.

Para ello se emplean distintas representaciones vectoriales de texto (TF-IDF, Word2Vec y BERT) y varios modelos supervisados (Logistic Regression, redes neuronales en PyTorch y fine-tuning de un modelo Transformer preentrenado).

---

## 1. Descripción del problema y del conjunto de datos

El problema se formula como una tarea de **clasificación de texto** binaria:

- **Entrada**: texto de un tuit político en español.
- **Salida**: etiqueta de ideología binaria `left` / `right` (`ideology_binary`).

Además, se quiere **caracterizar la presencia de discurso sobre desinformación** (bulos, fake news, etc.) y analizar su posible relación con la polarización ideológica.

### 1.1. Conjunto de datos: POLITiCES 2023

El proyecto utiliza el dataset **POLITiCES 2023**, que contiene tuits
 en español sobre política, anotados con metadatos como:

- `tweet`: texto original del tuit
- `ideology_binary`: ideología (left / right)
- Otros campos (género, profesión, partido, etc.) que no se explotan en profundidad en este trabajo.

Tras el preprocesado y filtrado, el conjunto de datos se separa en tres particiones no solapadas:

- **Train**: utilizado para entrenar los modelos.
- **Valid**: utilizado para selección de hiperparámetros y early stopping.
- **Test**: utilizado exclusivamente para la evaluación final.

La partición es **estratificada** respecto a `ideology_binary` y se mantiene fija en todos los experimentos, de forma que los resultados de los distintos modelos sean comparables.

### 1.2. Hipótesis de trabajo

A partir del análisis exploratorio inicial (distribución de clases, frecuencias léxicas, nubes de palabras y primer cálculo del proxy de desinformación), se plantean las siguientes hipótesis:

- **H1 (polarización léxica):**  
  Los tuits de izquierda y derecha utilizan marcos léxicos diferentes. En la izquierda se espera una mayor presencia de términos relacionados con derechos sociales y servicios públicos (por ejemplo, `derechos`, `mujeres`, `pública`, `social`, `personas`), mientras que en la derecha se anticipa más énfasis en identidad nacional y orden público (por ejemplo, `españoles`, `libertad`, `impuestos`, `sedición`, `eta`, `golpistas`).

- **H2 (desinformación y lenguaje polarizado):**  
  Los tuits que contienen vocabulario relacionado con desinformación (según el proxy de palabras clave) tienden a usar un lenguaje más conflictivo y emocional que los que no lo contienen, con referencias más frecuentes a delitos, fraudes, engaños, etc. Se espera que este tipo de mensajes aparezca en contextos donde la polarización ideológica es mayor.

- **H3 (asimetría ideológica suave):**  
  Aunque el proxy es muy limitado, la ligera sobrerrepresentación de tuits de izquierda entre los mensajes con palabras de desinformación sugiere que podría existir cierta asimetría en cómo cada polo habla de bulos y desinformación (ya sea para difundirlos o para denunciarlos). Esta hipótesis deberá contrastarse con modelos supervisados más robustos y con representaciones vectoriales del texto (TF-IDF, Word2Vec, BERT).

- **H4 (capacidad de los modelos):**  
  Se espera que los modelos basados en representaciones contextuales (por ejemplo, BERT en español) capturen mejor estos matices de polarización y desinformación que los baselines simples basados en bolsa de palabras (TF-IDF) o en promedios de Word2Vec.


---

## 2. Metodología

La metodología sigue las etapas marcadas en el enunciado del proyecto: análisis exploratorio, representación vectorial y modelado supervisado.

### 2.1. Preprocesado y análisis exploratorio

Antes de vectorizar el texto, se realiza un preprocesado básico:

- Conversión a minúsculas.
- Eliminación de URLs (`http...`), menciones (`@usuario`) y símbolo `#` manteniendo la palabra del hashtag.
- Definición de una columna `tweet_clean` con el texto limpio.

Para el análisis exploratorio:

- Se describen el **número de instancias y distribución de clases** (`left` / `right`).
- Se estudia la **distribución de longitudes de tuit**.
- Se generan **tablas de frecuencias** y **nubes de palabras** por ideología.

Los resultados muestran que ambos bloques ideológicos comparten un **núcleo léxico muy similar**, centrado en términos como `gobierno`, `españa`, `ley`, `madrid`, `política`, etc. Esto sugiere que las diferencias ideológicas no se manifiestan tanto en la frecuencia de palabras individuales aisladas, sino en combinaciones de términos y en el contexto.

Además, se calcula un conjunto de **palabras más características de cada ideología** utilizando una medida tipo log-odds entre frecuencias relativas de izquierda y derecha, lo que permite resaltar matices léxicos específicos de cada bloque.

### 2.2. Representación vectorial del texto

Se comparan tres estrategias de representación:

1. **TF-IDF**  
   - Se utiliza `TfidfVectorizer` de scikit-learn sobre `tweet_clean`.
   - Ajustado únicamente sobre el conjunto de entrenamiento.
   - Se experimenta con:
     - n-gramas (unigramas, y combinaciones hasta bigramas),
     - `min_df` (para filtrar términos muy raros),
     - uso de stopwords refinadas.

2. **Word2Vec**  
   - Se entrena un modelo Word2Vec sobre el corpus de tuits.
   - Cada documento se representa como el **promedio de los vectores de palabra** de sus tokens presentes en el modelo.
   - Aunque Word2Vec puede aprovechar todo el corpus para entrenarse, el **split train/valid/test se fija previamente** y solo se usan las representaciones correspondientes a cada partición para entrenar y evaluar los modelos supervisados.

3. **Embeddings contextuales (BERT)**  
   - Se emplea el modelo `dccuchile/bert-base-spanish-wwm-cased`.
   - Se obtiene una representación de tamaño fijo para cada tuit a partir del **vector [CLS]** o del promedio de las últimas capas.
   - Estas representaciones se utilizan inicialmente como features de entrada a modelos clásicos (Logistic Regression, MLP).
   - Finalmente, se entrena también un **modelo BERT con fine-tuning completo**, ajustando sus pesos finales para la tarea de clasificación binaria.

En todos los casos, la separación train/valid/test se mantiene constante, y cualquier transformación que requiera un `fit` se ajusta únicamente con el conjunto de entrenamiento.

### 2.3. Modelos supervisados

Se entrena y evalúa, para cada tipo de representación, al menos:

1. **Modelo clásico de scikit-learn**  
   - Se utiliza principalmente **Logistic Regression** (`LogisticRegression` y `LogisticRegressionCV`) dadas sus buenas propiedades como baseline lineal y su interpretabilidad.
   - Se ajustan hiperparámetros mediante validación cruzada:
     - `C` (fuerza de regularización),
     - `class_weight` (`None` vs `balanced`),
     - parámetros del vectorizador (en el caso de TF-IDF).

2. **Red neuronal en PyTorch (MLP)**  
   - Se implementa una **red feedforward** en PyTorch:
     - 1–2 capas ocultas (por ejemplo, 256 unidades),
     - activación ReLU,
     - dropout para regularización,
     - opcionalmente batch normalization.
   - Se realiza una pequeña búsqueda de hiperparámetros sobre:
     - tamaño de las capas ocultas,
     - tasa de dropout,
     - learning rate,
     - peso de la regularización L2.
   - Se entrena un MLP **por cada representación**:
     - TF-IDF + MLP  
     - Word2Vec + MLP  
     - BERT embeddings + MLP  

3. **Modelo Transformer preentrenado con fine-tuning (HuggingFace Transformers)**  
   - Se ajusta `AutoModelForSequenceClassification` / `BertForSequenceClassification` sobre el texto bruto.
   - Entrenamiento con GPU (PyTorch + CUDA).
   - Se usan `TrainingArguments` y `Trainer` (o un bucle equivalente en PyTorch) con:
     - número de épocas limitado (p.ej. 3),
     - batch size razonable para la GPU,
     - learning rate típico para BERT (≈ 2e-5),
     - warmup y weight decay.
   - El modelo se evalúa directamente sobre el conjunto de test y se compara con las otras aproximaciones.

En todos los modelos se usan métricas de:

- **Accuracy**
- **F1-macro**
- Informe de clasificación (`classification_report`) para ver el equilibrio entre clases.

---

## 3. Resultados experimentales y discusión

A continuación se resumen de forma cualitativa los principales resultados en el conjunto de test (27 000 tuits) para las distintas combinaciones de representación + modelo.

### 3.1. Resumen de resultados por combinación

**TF-IDF + Logistic Regression (LogRegCV, `class_weight='balanced'`)**

- En validación, el mejor modelo seleccionado mediante `LogisticRegressionCV` alcanza:
  - **Accuracy VALID ≈ 0.69**
  - **F1-macro VALID ≈ 0.69**
- El comportamiento por clase es bastante equilibrado:
  - `left`: F1 ≈ 0.72 (precision 0.73, recall 0.70)  
  - `right`: F1 ≈ 0.65 (precision 0.64, recall 0.67)
- Este modelo se usa como **baseline lineal fuerte** para comparar el resto de aproximaciones.

---

**Word2Vec + Logistic Regression (LogRegCV, `class_weight='balanced'`)**

- VALID:
  - **Accuracy ≈ 0.58**
  - **F1-macro ≈ 0.58**
- TEST:
  - **Accuracy ≈ 0.58**
  - **F1-macro ≈ 0.58**
- Por clase en test:
  - `left`: F1 ≈ 0.61 (precision 0.64, recall 0.59)  
  - `right`: F1 ≈ 0.55 (precision 0.53, recall 0.58)
- Conclusión: como representación media de embeddings, **Word2Vec + LogReg rinde claramente por debajo de TF-IDF + LogReg**, y tiende a ser menos estable entre clases.

---

**BERT (embeddings) + Logistic Regression (tuned)**

- VALID:
  - **Accuracy ≈ 0.62**
  - **F1-macro ≈ 0.62**
- TEST:
  - **Accuracy ≈ 0.61**
  - **F1-macro ≈ 0.61**
- Por clase en test:
  - `left`: F1 ≈ 0.64 (precision 0.67, recall 0.61)  
  - `right`: F1 ≈ 0.58 (precision 0.56, recall 0.61)
- Conclusión: usar **embeddings de BERT fijos con una LogReg** mejora a Word2Vec + LogReg, pero **no llega al rendimiento de TF-IDF + LogRegCV** sobre este conjunto de datos.

---

**TF-IDF + MLP (PyTorch, mejor configuración)**

- TEST:
  - **Accuracy ≈ 0.68**
  - **F1-macro ≈ 0.68**
- Por clase:
  - `left`: F1 ≈ 0.72 (precision 0.71, recall 0.73)  
  - `right`: F1 ≈ 0.63 (precision 0.65, recall 0.62)
- Comentario:
  - El MLP es capaz de **explotar algo mejor la representación TF-IDF** que la LogReg, situándose ligeramente por encima o en línea con el baseline lineal.
  - El rendimiento sigue siendo algo mejor para la clase `left`, pero sin desequilibrios extremos.

---

**Word2Vec + MLP (PyTorch, mejor configuración)**

- TEST:
  - **Accuracy ≈ 0.63**
  - **F1-macro ≈ 0.60**
- Por clase:
  - `left`: F1 ≈ 0.70 (precision 0.63, recall 0.79)  
  - `right`: F1 ≈ 0.50 (precision 0.61, recall 0.43)
- Comentario:
  - El MLP mejora claramente sobre Word2Vec + LogReg, pero el modelo sigue **mucho más fino en la clase `left` que en `right`** (recall derecha ≈ 0.43).
  - Esto refuerza la idea de que, en este problema, **la representación promedio de Word2Vec pierde información relevante para distinguir bien ambas ideologías**, especialmente la derecha.

---

**BERT (embeddings) + MLP (mejor configuración)**

- TEST:
  - **Accuracy ≈ 0.66**
  - **F1-macro ≈ 0.66**
- Por clase:
  - `left`: F1 ≈ 0.70 (precision 0.70, recall 0.70)  
  - `right`: F1 ≈ 0.62 (precision 0.62, recall 0.61)
- Comentario:
  - El MLP sobre embeddings de BERT logra **rendimiento intermedio**:
    - Mejor que cualquier variante basada en Word2Vec.
    - Pero **ligeramente por debajo de TF-IDF + MLP**, que en este dataset en concreto sigue siendo una referencia muy fuerte.

---

**BERT fine-tuned (modelo Transformer con ajuste de todos los pesos)**

- TEST:
  - **Accuracy ≈ 0.75**
  - **F1-macro ≈ 0.75**
- Por clase:
  - `left`: F1 ≈ 0.78 (precision 0.77, recall 0.79)  
  - `right`: F1 ≈ 0.72 (precision 0.73, recall 0.71)
- Comentario:
  - Es el **modelo con mejor rendimiento global**:
    - Gana ≈ 7 puntos porcentuales de F1-macro respecto a TF-IDF + MLP.
    - Mantiene un equilibrio bastante razonable entre `left` y `right`.
  - El precio a pagar es un **coste computacional mucho mayor** (entrenamiento en GPU, más memoria, más tiempo).

---

### 3.2. Comparativa global

De forma resumida, en el conjunto de test:

- **Mejor modelo**:  
  - **BERT fine-tuned**  
    - Accuracy ≈ 0.75, F1-macro ≈ 0.75.

- **Mejor combinación “clásica” (sin fine-tuning completo)**:  
  - **TF-IDF + MLP (PyTorch)**  
    - Accuracy ≈ 0.68, F1-macro ≈ 0.68.  
    - Competitivo, sencillo de entrenar y menos costoso que BERT fine-tuned.

- **Modelos intermedios**:
  - **BERT embeddings + MLP**: Accuracy ≈ 0.66, F1-macro ≈ 0.66.  
  - **BERT embeddings + LogReg**: Accuracy ≈ 0.61, F1-macro ≈ 0.61.  
  - **TF-IDF + LogRegCV**: F1-macro VALID ≈ 0.69; sirve como baseline lineal sólido.

- **Peor rendimiento relativo**:
  - **Word2Vec + LogReg**: Accuracy ≈ 0.58, F1-macro ≈ 0.58.  
  - **Word2Vec + MLP** mejora a la LogReg pero se queda en Accuracy ≈ 0.63, F1-macro ≈ 0.60, con claro sesgo a favor de la clase `left`.

En este contexto, **TF-IDF sigue siendo una representación extremadamente competitiva** para clasificación de texto político, y solo el **fine-tuning completo de BERT** consigue superarla de forma clara.


### 3.3. Desinformación y polarización ideológica

Para incorporar la dimensión de la desinformación se ha realizado un análisis léxico:

1. Se define un pequeño **léxico de términos asociados a desinformación**, que incluye palabras como:
   - `bulo`, `bulos`, `fake news`, `noticia falsa`, `desinformación`, `hoax`, `mentira`, `manipulado`, `conspiración`, etc.
2. Para cada tuit se calcula un **score de desinformación** `misinfo_score`, que cuenta cuántos de esos términos aparecen en `tweet_clean`, y un indicador binario `misinfo_flag` (1 si el tuit contiene al menos un término del léxico).

Resultados principales:

- Solo alrededor del **1,5 %** de los tuits
 contienen al menos un término del léxico de desinformación:
  - left: ≈ **1,50 %** (1503 de 100 400)
  - right: ≈ **1,42 %** (1131 de 79 600)
- El **score medio** de desinformación es muy bajo en ambos grupos (`≈ 0.02`), lo que indica que la mayoría de tuits no mencionan explícitamente bulos o fake news.
- Una prueba **chi-cuadrado** sobre la tabla de contingencia [ideología × `misinfo_flag`] arroja:
  - Chi² ≈ 1.73, p ≈ 0.188 (> 0.05),
  - por lo que **no se puede rechazar la hipótesis de independencia** entre ideología y presencia de términos de desinformación.

Interpretación:

- El **discurso explícito sobre desinformación** (es decir, hablar de “bulos”, “fake news”, “mentiras”, etc.) es minoritario en el dataset.
- La diferencia entre izquierda y derecha en este indicador es **pequeña y no significativa** estadísticamente.
- Este enfoque léxico no permite distinguir entre tuits que difunden desinformación y tuits que la critican, por lo que debe interpretarse como una **medida aproximada de la presencia del tema**, no como un detector de contenido falso.

---

## 4. Conclusiones

## 4. Resultados y conclusiones

### 4.1. Conclusiones

1. **Clasificación de ideología**  
   - Es posible predecir la ideología binaria (`left` / `right`) a partir del texto de los tuits con un rendimiento razonable: los modelos se sitúan aproximadamente entre un **65 % y un 75 % de accuracy y F1-macro**, según la arquitectura utilizada.  
   - Los modelos lineales (Logistic Regression) combinados con TF-IDF constituyen un **baseline sólido y estable**.  
   - Las redes neuronales (MLP) en PyTorch permiten exprimir mejor las representaciones TF-IDF y BERT, obteniendo mejoras moderadas sobre los modelos lineales.  
   - El fine-tuning de BERT se sitúa como la **estrategia más potente**, alcanzando alrededor de un 75 % de accuracy y F1-macro en test, a costa de un coste computacional sensiblemente mayor.

2. **Comparación de representaciones**  
   - **TF-IDF** resulta ser una representación muy efectiva para esta tarea, especialmente combinada con modelos lineales o con un MLP sencillo.  
   - **Word2Vec** con promedio de embeddings pierde parte de la información contextual y, en este problema concreto, ofrece un rendimiento inferior incluso cuando se usa con redes neuronales.  
   - Los **embeddings contextuales de BERT** aportan una representación más rica del texto:  
     - En modo congelado (BERT + LogReg / MLP) logran resultados competitivos, cercanos a TF-IDF + MLP.  
     - Cuando se realiza **fine-tuning completo**, BERT supera con claridad a las técnicas clásicas y se convierte en el mejor modelo del proyecto.

3. **Desinformación y polarización**  
   - El análisis léxico de desinformación (basado en un pequeño léxico de palabras clave como `bulo`, `fake news`, `mentira`, etc.) muestra que la mención explícita a estos términos es relativamente **poco frecuente** (en torno a un 1–2 % de los tuits).  
   - La distribución de este proxy de desinformación es **muy similar entre izquierda y derecha** y no se observa una relación estadísticamente significativa con la ideología binaria.  
   - Para profundizar en la relación entre desinformación y polarización sería necesario disponer de anotaciones explícitas de veracidad o aplicar modelos específicos de stance/sentimiento que vayan más allá del simple conteo de palabras clave.

4. **Líneas futuras**  
   - Incorporar modelos de **stance detection** o análisis de sentimiento para caracterizar mejor la postura del autor respecto a temas potencialmente desinformativos.  
   - Explorar mecanismos de **atención** o técnicas de explicabilidad que permitan identificar qué fragmentos del texto contribuyen más a la predicción de ideología o al posible vínculo con desinformación.   

### 4.2. Interpretación de resultados en relación con las hipótesis

A la luz de los resultados obtenidos, se puede hacer el siguiente contraste cualitativo de las hipótesis de partida:

- **H1 (polarización léxica)**  
  El análisis exploratorio muestra que izquierda y derecha comparten un núcleo de vocabulario muy similar (`gobierno`, `españa`, `ley`, `madrid`, `política`, etc.). Sin embargo, los modelos de clasificación alcanzan métricas claramente por encima del azar (en torno a 0.68 con TF-IDF + MLP y alrededor de 0.75 con BERT fine-tuned), lo que implica que existen patrones léxicos más sutiles (combinaciones de términos, n-gramas y contexto) que permiten discriminar la ideología.  
  **Conclusión:** H1 se ve parcialmente confirmada: hay señal de polarización léxica.

- **H2 (desinformación y lenguaje polarizado)**  
  El proxy de desinformación basado en palabras clave (`bulo`, `fake news`, `mentira`, etc.) identifica un porcentaje muy pequeño de tuits y no se ha complementado con un análisis específico del tono emocional o del grado de conflictividad del lenguaje.  
  **Conclusión:** con el enfoque actual, H2 no puede confirmarse y se requieren herramientas adicionales para evaluar el componente emocional o polarizado del leguaje asociado a desinformación.

- **H3 (asimetría ideológica suave en desinformación)**  
  Aunque inicialmente se observan ligeras diferencias en la proporción de tuits con vocabulario de desinformación entre izquierda y derecha, la prueba chi-cuadrado da un p-valor ≈ 0.18 (> 0.05). Es decir, las diferencias observadas son compatibles con la variabilidad aleatoria del muestreo y no alcanzan significación estadística.  
  **Conclusión:** H3 no queda confirmada ya que no se detecta una asimetría clara en la forma en que cada bloque menciona explícitamente bulos o desinformación.

- **H4 (capacidad de los modelos)**  
  La comparación sistemática entre representaciones y modelos muestra que:
  - Word2Vec con promedio de embeddings queda por debajo de TF-IDF incluso cuando se entrena un MLP.  
  - TF-IDF combinado con MLP configura un baseline clásico muy competitivo, sencillo de entrenar e interpretar.  
  - Los embeddings contextuales de BERT en modo congelado mejoran a Word2Vec y se sitúan cerca de TF-IDF + MLP, pero el salto cualitativo llega con el **fine-tuning completo**, que proporciona las mejores métricas del proyecto.  
  Esto confirma que los modelos contextuales son capaces de capturar mejor la estructura semántica y los matices de polarización cuando se les permite adaptar sus pesos.
  **Conclusión:** H4 se confirma: BERT fine-tuned supera claramente a las representaciones clásicas, aunque TF-IDF + MLP sigue siendo interesantes con una relación coste–beneficio muy razonable en este dataset.

En conjunto, los experimentos confirman que existe señal suficiente en el texto para predecir la ideología con un rendimiento razonable, pero la relación entre desinformación ) y polarización ideológica es más difícil de capturar y no muestra patrones claros con las métricas actuales.



---

## Referencias

[1] García-Díaz, J. A., et al. (2023). Overview of PoliticES at IberLEF 2023: Political Ideology and Misinformation in Spanish. *Procesamiento del Lenguaje Natural, 71*.

[2] Cañete, J., et al. (2020). Spanish Pre-trained BERT Model and Evaluation Data. PML4DC at ICLR 2020.

[3] Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.

[4] Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. *EMNLP 2020*.

[5] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR, 12*, 2825–2830.

[6] Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 32*.

[7] Řehůřek, R., & Sojka, P. (2010). Software Framework for Topic Modelling with Large Corpora. *LREC Workshop*.

[8] Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O’Reilly Media.

[9] Salton, G., & Buckley, C. (1988). Term-Weighting Approaches in Automatic Text Retrieval. *Information Processing & Management, 24*(5).

[10] Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781.

[11] Soler, B., et al. (2023). UC3M at PoliticES 2023: Applying The Basics. *IberLEF 2023*.
