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

A partir de los experimentos realizados se pueden extraer las siguientes conclusiones:

1. **Clasificación de ideología**  
   - Es posible predecir la ideología binaria (`left` / `right`) a partir del texto de los tuits con una accuracy en torno al **65–75 %** y F1-macro similar, dependiendo del modelo.
   - Los modelos lineales (Logistic Regression) combinados con TF-IDF ofrecen ya un baseline sólido.
   - Las redes neuronales (MLP) en PyTorch permiten exprimir algo más las representaciones TF-IDF y BERT, logrando ligeras mejoras.
   - El fine-tuning de BERT proporciona resultados competitivos en torno al **75 %**, aunque con un coste computacional sensiblemente mayor.

2. **Comparación de representaciones**  
   - **TF-IDF** sigue siendo una representación muy efectiva para tareas de clasificación de texto en dominios específicos como el político.
   - **Word2Vec** con promedio de embeddings pierde parte de la información contextual y, en este problema, rinde peor que TF-IDF. Aunque podría ser debido al refinado, habría que intentar mejorarlo.
   - Los **embeddings contextuales de BERT** aportan una representación rica, que combinada con un MLP o mediante fine-tuning alcanza niveles de rendimiento superiores a las técnicas clásicas de forma sencilla.

3. **Desinformación y polarización**  
   - El análisis léxico de desinformación muestra que la mención explícita a bulos y fake news es relativamente poco frecuente y se distribuye de forma bastante equilibrada entre izquierda y derecha.
   - No se observa, con este enfoque, una relación estadísticamente significativa entre la ideología binaria y la presencia de términos de desinformación.
   - Para profundizar en la relación entre desinformación y polarización sería necesario disponer de anotaciones explícitas de veracidad al igual que existe de ideología.

4. **Líneas futuras**  
   - Incorporar modelos de **stance detection** o análisis de sentimiento para caracterizar mejor la postura del autor respecto a temas potencialmente desinformativos.
   - Explorar mecanismos de **atención** o explicabilidad que permitan identificar fragmentos del texto que contribuyen más a la predicción de ideología.

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
