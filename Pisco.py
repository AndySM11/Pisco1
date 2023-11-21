#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Cargar datos desde el archivo Excel
df = pd.read_excel(r"C:\Users\Andy\Desktop\Data3.xlsx", header=0)


# In[3]:


print(df)


# In[4]:


print(df.head())


# In[5]:


# Visualizar la distribución de las clases
class_distribution = df['Clasificacion'].value_counts()
class_distribution.plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Distribución de Clases')
plt.xlabel('Clasificación (0: No Aromático, 1: Aromático)')
plt.ylabel('Número de Muestras')
plt.show()


# In[6]:


# Extraer características espectrales y etiquetas
X = df.iloc[:, 2:-1].values  # Características espectrales
y = df['Clasificacion'].values  # Etiquetas


# In[7]:


# Escalar las características para tener media cero y desviación estándar uno
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[8]:


# Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[9]:


# Inicializar modelos de clasificación
knn_model = KNeighborsClassifier(n_neighbors=3)
nb_model = GaussianNB()


# In[10]:


# Entrenar modelos
knn_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)


# In[11]:


# Realizar predicciones
knn_predictions = knn_model.predict(X_test)
nb_predictions = nb_model.predict(X_test)


# In[12]:


# Evaluar rendimiento
knn_accuracy = metrics.accuracy_score(y_test, knn_predictions)
nb_accuracy = metrics.accuracy_score(y_test, nb_predictions)

print(f'Accuracy KNN: {knn_accuracy}')
print(f'Accuracy Naive Bayes: {nb_accuracy}')


# In[13]:


from sklearn.metrics import confusion_matrix, classification_report

print("Métricas para KNN:")
print(confusion_matrix(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions))


# In[14]:


print("Métricas para Naive Bayes:")
print(confusion_matrix(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))


# In[15]:


from sklearn.metrics import confusion_matrix, classification_report

# Métricas para KNN
print("Métricas para KNN:")
print("Accuracy:", knn_accuracy)
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, knn_predictions))
print("\nInforme de Clasificación:")
print(classification_report(y_test, knn_predictions))


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Función para visualizar la matriz de confusión como un mapa de calor
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))

    # Mapa de calor
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=True)

    # Añadir texto con los valores de la matriz
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j + 0.5, i + 0.5, str(cm[i, j]), ha='center', va='center', color='red')

    plt.title("Matriz de Confusión")
    plt.xlabel("Predicciones")
    plt.ylabel("Valores Reales")
    plt.show()

# Visualizar la matriz de confusión para KNN
plot_confusion_matrix(y_test, knn_predictions, labels=[0, 1])

# Visualizar la matriz de confusión para Naive Bayes
plot_confusion_matrix(y_test, nb_predictions, labels=[0, 1])



# In[ ]:




