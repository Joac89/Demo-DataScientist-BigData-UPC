# ==============================================================================
# TALLER MAGISTRAL: IA APLICADA AL SECTOR FINANCIERO
# Caso: Decisiones de crédito con Machine Learning
# ==============================================================================

# ---------------------------
# LIBRERÍAS
# ---------------------------
# pandas → manejo de datos tipo tabla (DataFrame)
# seaborn / matplotlib → visualización
# sklearn → machine learning
# pathlib → validación de archivos

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

# ---------------------------
# CONFIGURACIÓN VISUAL
# ---------------------------
# Se define un estilo visual consistente para todos los gráficos
sns.set_theme(style="whitegrid")

# Paleta de colores para representar estados del negocio
COLORS = {
    "aprobado": "#2ca02c",   # verde
    "rechazado": "#d62728",  # rojo
    "neutro": "#1f77b4"      # azul
}

# Tamaño por defecto de gráficos
plt.rcParams["figure.figsize"] = (8, 5)

# ---------------------------
# 1. CARGA DE DATOS
# ---------------------------
print("\n--- 1. CARGA DE BASE DE DATOS BANCARIA ---")

FILE = "creditos_bancarios.csv"

# Validación: si el archivo no existe en el entorno, se detiene la ejecución
if not Path(FILE).exists():
    raise FileNotFoundError("❌ Sube el archivo 'creditos_bancarios.csv' al entorno.")

# Se carga el CSV en un DataFrame (estructura tipo tabla)
df = pd.read_csv(FILE)

print(f"✅ Historial cargado: {len(df)} clientes")

# Se muestran 5 registros aleatorios para inspección rápida
display(df.sample(5))

# ---------------------------
# 2. ANÁLISIS EXPLORATORIO
# ---------------------------
# Objetivo: entender los datos antes de modelar
print("\n--- 2. ANÁLISIS EXPLORATORIO ---")

# 🔥 MAPA DE CALOR (CORRELACIÓN)
print("\n[GRÁFICO 1] MAPA DE RELACIONES ENTRE VARIABLES")

# Calcula correlaciones SOLO entre variables numéricas
correlation = df.corr(numeric_only=True)

plt.figure(figsize=(7, 5))

# Heatmap → muestra relaciones entre variables
sns.heatmap(
    correlation,
    annot=True,        # muestra valores numéricos
    cmap="coolwarm",   # colores (rojo/azul)
    fmt=".2f",
    linewidths=0.5
)

plt.title("Relación entre variables (correlación)")
plt.show()

# Interpretación guiada del gráfico
print("""
Cómo leerlo:

- Valores cercanos a +1 → relación directa (sube uno, sube el otro)
- Valores cercanos a -1 → relación inversa (sube uno, baja el otro)

Punto clave del negocio:
- Endeudamiento vs Crédito Aprobado → relación negativa fuerte
- Salario vs Crédito Aprobado → relación positiva
""")

# 🔥 KDE (Distribución de probabilidad)
print("\n[GRÁFICO 2] DISTRIBUCIÓN DE SALARIOS")

plt.figure(figsize=(10, 5))

# Distribución de salarios de clientes APROBADOS
sns.kdeplot(
    data=df[df['Credito_Aprobado'] == 1],
    x='Salario_Mensual_USD',
    color=COLORS["aprobado"],
    fill=True,
    label="Aprobados",
    alpha=0.5
)

# Distribución de salarios de clientes RECHAZADOS
sns.kdeplot(
    data=df[df['Credito_Aprobado'] == 0],
    x='Salario_Mensual_USD',
    color=COLORS["rechazado"],
    fill=True,
    label="Rechazados",
    alpha=0.5
)

plt.title("Distribución de salario según resultado del crédito")
plt.xlabel("Salario Mensual (USD)")
plt.legend()
plt.show()

print("""
Interpretación:

- Los aprobados se concentran en salarios altos
- Los rechazados en salarios bajos

→ Insight: el ingreso influye fuertemente en la decisión
""")

# 🔥 SCATTER (Relación entre variables)
print("\n[GRÁFICO 3] RELACIÓN SALARIO VS ENDEUDAMIENTO")

plt.figure()

# Scatterplot → cada punto es un cliente
sns.scatterplot(
    data=df,
    x='Salario_Mensual_USD',
    y='Endeudamiento_Actual_Pct',
    hue='Credito_Aprobado',  # color según resultado
    palette=[COLORS["rechazado"], COLORS["aprobado"]],
    alpha=0.7
)

plt.title("Comportamiento de clientes")
plt.show()

print("""
Interpretación visual:

- Abajo derecha → clientes ideales (alto ingreso, baja deuda)
- Arriba izquierda → clientes riesgosos

→ Esto es EXACTAMENTE lo que el modelo aprenderá
""")

# ---------------------------
# 3. MODELO
# ---------------------------
print("\n--- 3. ENTRENAMIENTO DEL MODELO ---")

# Variables de entrada (features)
X = df[['Salario_Mensual_USD', 'Endeudamiento_Actual_Pct']]

# Variable objetivo (lo que queremos predecir)
y = df['Credito_Aprobado']

# División entrenamiento / prueba
# 80% para entrenar, 20% para evaluar
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo: Árbol de Decisión
# max_depth=3 → limita complejidad (evita sobreajuste)
modelo = DecisionTreeClassifier(max_depth=3, random_state=42)

# Entrenamiento (el modelo aprende patrones)
modelo.fit(X_train, y_train)

# Evaluación del modelo
accuracy = accuracy_score(y_test, modelo.predict(X_test))

print(f"✔️ Precisión del modelo: {accuracy:.2%}")

# ---------------------------
# 4. ÁRBOL DE DECISIÓN
# ---------------------------
print("\n--- 4. INTERPRETACIÓN DEL MODELO ---")

print("""
Qué está pasando aquí:

El modelo crea reglas tipo:

SI deuda > X → rechazar
SI salario > Y → aprobar

El índice Gini mide qué tan "puro" es un grupo:
- 0 → todos iguales
- alto → mezcla de aprobados/rechazados

El objetivo del árbol:
dividir clientes hasta que cada grupo sea lo más homogéneo posible.
""")

plt.figure(figsize=(16, 8))

# Visualización del árbol de decisión
plot_tree(
    modelo,
    feature_names=X.columns,           # nombres de variables
    class_names=['RECHAZADO', 'APROBADO'],
    filled=True,                      # colorea nodos
    rounded=True,
    proportion=True,                  # muestra proporciones
    fontsize=11
)

plt.title("Árbol de decisión: reglas del modelo")
plt.show()

# ---------------------------
# 5. SIMULADOR
# ---------------------------
print("\n--- 5. SIMULADOR ---")

# Función que permite probar clientes nuevos
def evaluar_cliente(salario, deuda):

    # Se construye un DataFrame con el formato esperado por el modelo
    input_df = pd.DataFrame([{
        'Salario_Mensual_USD': salario,
        'Endeudamiento_Actual_Pct': deuda
    }])

    # Predicción (0 o 1)
    pred = modelo.predict(input_df)[0]

    # Traducción a lenguaje de negocio
    return "APROBADO" if pred == 1 else "RECHAZADO"

# Ejemplos reales de uso
print("Cliente riesgoso ($2500, deuda 50%):", evaluar_cliente(2500, 50))
print("Cliente ideal ($4000, deuda 10%):", evaluar_cliente(4000, 10))
