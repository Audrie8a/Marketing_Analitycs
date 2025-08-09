#Importar librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier


########################################### 2. Cargar el dataset
# a. Carga el conjunto de datos proporcionado (en formato CSV u otro compatible).
df = pd.read_csv('adult.csv')
df.head()

# c. Realiza un análisis exploratorio básico para identificar valores nulos, datos atípicos y la distribución de las variables.
#       Verificar columnas y tipos de datos
df.info()

#       Verificar valires nulos
print("Valores con '?':")
for col in df.columns:
    print(f"{col}: {(df[col] == '?').sum()}")

#       Reemplazar "?" por NaN y luego eliminar filas con valores faltantes
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

#       Verificación posterior a la limpieza
df.info()

# d. Convierte las variables categóricas en numéricas mediante técnicas de codificación, como la codificación de etiquetas o variables ficticias.
# Identificamos las columnas categóricas
cat_cols = df.select_dtypes(include='object').columns


label_encoders = {}

#       Aplicamos LabelEncoder a cada columna categórica
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('income', axis=1)
y = df['income']

# f. Divide el conjunto de datos en conjuntos de entrenamiento y prueba, utilizando una proporción adecuada (por ejemplo, 80% entrenamiento y 20% prueba).
#       Dividimos 80% para entrenamiento y 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# e. Estandariza o normaliza las variables numéricas para mejorar el rendimiento del modelo.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

########################################### 3. Crear el modelo

# a. Utiliza un clasificador de perceptrón simple para entrenar el modelo con el conjunto de datos de entrenamiento.
# b. Ajusta los parámetros de entrenamiento, como la tasa de aprendizaje y el número máximo de iteraciones.
perceptron = Perceptron(max_iter=1000, eta0=1.0, random_state=42)

#       Entrenar con los datos
perceptron.fit(X_train, y_train)

# c. Evalúa el modelo utilizando el conjunto de prueba para obtener métricas de rendimiento como la matriz de confusión, precisión, recall y F1-score.
#       Predecir en el conjunto de prueba
y_pred = perceptron.predict(X_test)

#       Evaluar el modelo
print("=== Perceptrón Simple ===")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

########################################### 4. Crear el modelo MLP


# a. Implementa un perceptrón multicapa utilizando un clasificador adecuado.
# b. Define la arquitectura de la red neuronal, incluyendo el número de capas ocultas, el número de neuronas por capa y la función de activación.
# c. Ajusta los hiperparámetros relevantes, como el algoritmo de optimización, la tasa de aprendizaje y el número de iteraciones.

mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                    max_iter=300, random_state=42)

#       Entrenar el modelo
mlp.fit(X_train, y_train)

# d. Evalúa el modelo con el conjunto de prueba y analiza las métricas de rendimiento obtenidas.
#       Predecir
y_pred_mlp = mlp.predict(X_test)

# Evaluar
print("\n=== Perceptrón Multicapa (MLP) ===")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred_mlp))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_mlp))