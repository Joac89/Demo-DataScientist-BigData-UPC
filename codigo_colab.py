# ==============================================================================
# TALLER MAGISTRAL: IA APLICADA AL SECTOR FINANCIERO
# Caso: Decisiones de crédito con Machine Learning
# ==============================================================================

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
sns.set_theme(style="whitegrid")

COLORS = {
    "aprobado": "#2ca02c",
    "rechazado": "#d62728",
    "neutro": "#1f77b4"
}

plt.rcParams["figure.figsize"] = (8, 5)

# ---------------------------
# 1. CARGA DE DATOS
# ---------------------------
print("\n--- 1. CARGA DE BASE DE DATOS BANCARIA ---")

FILE = "creditos_bancarios.csv"

if not Path(FILE).exists():
    raise FileNotFoundError("❌ Sube el archivo 'creditos_bancarios.csv' al entorno.")

df = pd.read_csv(FILE)

print(f"✅ Historial cargado: {len(df)} clientes")
display(df.sample(5))

# ---------------------------
# 2. ANÁLISIS EXPLORATORIO
# ---------------------------
print("\n--- 2. ANÁLISIS EXPLORATORIO ---")

# 🔥 MAPA DE CALOR
print("\n[GRÁFICO 1] MAPA DE RELACIONES ENTRE VARIABLES")

correlation = df.corr(numeric_only=True)

plt.figure(figsize=(7, 5))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Relación entre variables (correlación)")
plt.show()

print("""
Cómo leerlo:

- Valores cercanos a +1 → relación directa (sube uno, sube el otro)
- Valores cercanos a -1 → relación inversa (sube uno, baja el otro)

Punto clave del negocio:
- Endeudamiento vs Crédito Aprobado → relación negativa fuerte
  → A mayor deuda, menor probabilidad de aprobación

- Salario vs Crédito Aprobado → relación positiva
  → Mayor ingreso, mejor perfil crediticio
""")

# 🔥 KDE
print("\n[GRÁFICO 2] DISTRIBUCIÓN DE SALARIOS")

plt.figure(figsize=(10, 5))

sns.kdeplot(
    data=df[df['Credito_Aprobado'] == 1],
    x='Salario_Mensual_USD',
    color=COLORS["aprobado"],
    fill=True,
    label="Aprobados",
    alpha=0.5
)

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
Lectura del gráfico:

- La curva de aprobados tiende a concentrarse en salarios más altos
- La de rechazados se desplaza hacia salarios más bajos

Interpretación:
El ingreso es un factor determinante en la decisión de crédito.
""")

# 🔥 SCATTER (añadido sin romper tu flujo)
print("\n[GRÁFICO 3] RELACIÓN SALARIO VS ENDEUDAMIENTO")

plt.figure()

sns.scatterplot(
    data=df,
    x='Salario_Mensual_USD',
    y='Endeudamiento_Actual_Pct',
    hue='Credito_Aprobado',
    palette=[COLORS["rechazado"], COLORS["aprobado"]],
    alpha=0.7
)

plt.title("Comportamiento de clientes")
plt.show()

print("""
Aquí se ve el patrón real:

- Zona inferior derecha → clientes ideales (alto salario, baja deuda)
- Zona superior izquierda → clientes de alto riesgo
""")

# ---------------------------
# 3. MODELO
# ---------------------------
print("\n--- 3. ENTRENAMIENTO DEL MODELO ---")

X = df[['Salario_Mensual_USD', 'Endeudamiento_Actual_Pct']]
y = df['Credito_Aprobado']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

accuracy = accuracy_score(y_test, modelo.predict(X_test))
print(f"✔️ Precisión del modelo: {accuracy:.2%}")

# ---------------------------
# 4. ÁRBOL DE DECISIÓN
# ---------------------------
print("\n--- 4. INTERPRETACIÓN DEL MODELO ---")

print("""
CÓMO ENTENDER EL GINI:

El índice Gini mide qué tan mezclado está un grupo:

- Gini = 0 → grupo puro (todos iguales)
- Gini alto → mezcla de perfiles distintos

Qué hace el modelo:
Divide a los clientes en grupos cada vez más homogéneos.

Ejemplo:
Primero separa por nivel de endeudamiento.
Luego refina usando el salario.
""")

plt.figure(figsize=(16, 8))

plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=['RECHAZADO', 'APROBADO'],
    filled=True,
    rounded=True,
    proportion=True,
    fontsize=11
)

plt.title("Árbol de decisión: reglas del modelo")
plt.show()

# ---------------------------
# 5. SIMULADOR
# ---------------------------
print("\n--- 5. SIMULADOR ---")

def evaluar_cliente(salario, deuda):
    input_df = pd.DataFrame([{
        'Salario_Mensual_USD': salario,
        'Endeudamiento_Actual_Pct': deuda
    }])
    pred = modelo.predict(input_df)[0]
    return "APROBADO" if pred == 1 else "RECHAZADO"

print("Cliente riesgoso ($2500, deuda 50%):", evaluar_cliente(2500, 50))
print("Cliente ideal ($4000, deuda 10%):", evaluar_cliente(4000, 10))
