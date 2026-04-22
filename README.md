# 🏦 Evaluación de Riesgo Crediticio con Machine Learning

## 📌 Descripción

Este ejercicio práctico demuestra cómo aplicar técnicas de Machine Learning en el sector financiero para apoyar la toma de decisiones en la aprobación de créditos.

Se desarrolla en un entorno experimental utilizando **Python en Google Colab**, trabajando sobre un dataset de clientes bancarios con variables representativas del perfil financiero.

El objetivo es analizar el comportamiento de los clientes y entrenar un modelo capaz de estimar la probabilidad de aprobación o rechazo de crédito a partir de variables clave como ingresos y nivel de endeudamiento.

---

## 🎯 Objetivos

- Entender cómo se utilizan datos en la evaluación de riesgo crediticio  
- Analizar relaciones entre variables financieras  
- Construir un modelo predictivo basado en Árboles de Decisión  
- Interpretar las reglas generadas por el modelo  
- Simular decisiones de crédito en distintos escenarios  

---

## 🧰 Herramientas utilizadas

- Python 3  
- Google Colab  
- Pandas → manipulación de datos  
- Seaborn / Matplotlib → visualización  
- Scikit-learn → modelo de Machine Learning  

---

## 📂 Dataset

**Archivo:** `creditos_bancarios.csv`

Contiene información simulada de clientes, incluyendo:

- `Salario_Mensual_USD` → nivel de ingresos  
- `Endeudamiento_Actual_Pct` → porcentaje de deuda  
- `Credito_Aprobado` → variable objetivo (0 = rechazado, 1 = aprobado)  

---

## ⚙️ Flujo del ejercicio

### 1. Carga de datos
Se importa el dataset y se valida su estructura.

### 2. Análisis exploratorio
Se analizan patrones clave mediante:
- Correlaciones entre variables  
- Distribución de ingresos  
- Relación entre salario y endeudamiento  

### 3. Entrenamiento del modelo
Se entrena un **Árbol de Decisión** para clasificar clientes en:
- Aprobados  
- Rechazados  

### 4. Interpretación del modelo
Se visualiza el árbol para entender:
- Qué variables influyen en la decisión  
- Cómo se segmentan los clientes  

### 5. Simulación
Se evalúan perfiles de clientes para observar cómo el modelo toma decisiones.

---

## 🧠 Conceptos clave

- **Riesgo crediticio** → probabilidad de incumplimiento  
- **Segmentación** → agrupación de clientes por características  
- **Índice Gini** → medida de pureza en nodos del árbol  
- **Overfitting** → ajuste excesivo del modelo a los datos  

---

## 🚀 Cómo ejecutar

1. Abrir Google Colab  
2. Subir el archivo `creditos_bancarios.csv`  
3. Copiar y ejecutar el notebook paso a paso  
4. Analizar resultados y probar el simulador  

---

## 🧪 Ejemplo de uso

```python
evaluar_cliente(2500, 50)  # Cliente de alto riesgo
evaluar_cliente(4000, 10)  # Cliente ideal
