# ---------------------------------------------------------
# TALLER: CIENCIA DE DATOS APLICADA A LA CALIDAD DE SOFTWARE
# ---------------------------------------------------------
# Caso de Estudio: Predicción de Bugs en Módulos de la NASA.
# Basado en el dataset histórico NASA Metrics Data Program.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# 1. CARGA DE MÉTRICAS DE SOFTWARE
# ==========================================
# Sube 'nasa_software_bugs.csv' a Google Colab
url = "nasa_software_bugs.csv" 

try:
    df = pd.read_csv(url)
    print("✅ Métricas de la NASA cargadas correctamente.")
    print(f"Total de módulos analizados: {len(df)}")
    print("\n--- Vista de métricas (LOC, McCabe, Operandos) ---")
    print(df[['lineas_codigo', 'complejidad_mccabe', 'tiene_bug']].head())
except:
    print("❌ Error: Sube el archivo 'nasa_software_bugs.csv' a Colab.")

# ==========================================
# 2. ANÁLISIS DE CALIDAD (QA Visual)
# ==========================================
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='lineas_codigo', y='complejidad_mccabe', hue='tiene_bug', palette='viridis', s=100)
plt.title('Relación: Tamaño del Código vs Complejidad Lógica')
plt.xlabel('Líneas de Código (LOC)')
plt.ylabel('Complejidad Ciclomática (McCabe)')
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 3. ENTRENAMIENTO DEL MODELO (Clasificador de Bugs)
# ==========================================
# Features: Métricas de complejidad y esfuerzo
X = df[['lineas_codigo', 'complejidad_mccabe', 'entradas_unicas', 'esfuerzo_desarrollo']]
y = df['tiene_bug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo: Árbol de Decisión (Fácil de interpretar para un desarrollador)
modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(X_train, y_train)

# Medición de eficiencia
precision = accuracy_score(y_test, modelo.predict(X_test))
print(f"\n📊 Precisión del Predictor de Bugs: {precision:.2%}")

# Visualización de las reglas de negocio aprendidas por la IA
plt.figure(figsize=(16,8))
plot_tree(modelo, feature_names=X.columns, class_names=['Estable', 'BUG'], filled=True, rounded=True)
plt.title("Árbol de Decisión: ¿Qué hace que un código sea propenso a errores?")
plt.show()

# ==========================================
# 4. TESTEA TU PROPIO MÓDULO
# ==========================================
print("\n--- ANALIZADOR DE CÓDIGO PREDICTIVO ---")
# [lineas_codigo, complejidad_mccabe, entradas_unicas, esfuerzo_desarrollo]
# Ejemplo: Un módulo de 100 líneas, muy complejo (20), con 30 entradas.
mi_codigo = [[100, 20, 30, 1500]] 

prediccion = modelo.predict(mi_codigo)
if prediccion[0] == 1:
    print("🚨 ALERTA: Este módulo tiene alta probabilidad de contener un BUG. ¡Revisar QA!")
else:
    print("🟢 STATUS: El módulo cumple con los estándares de estabilidad.")
