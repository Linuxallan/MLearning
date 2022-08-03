#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 02:40:21 2022

@author: allancortez
"""

# Plantilla de clasificacion 

# Librerias
import numpy as np # Libreria de matematicas para los algoritmos MLeaning
import matplotlib.pyplot as plt # Diseño de graficos
import pandas as pd # Carga de datos (como para big data)

# importar el data set
dataset = pd.read_csv('Social_Network_Ads.csv')
# Vector: lista de 1 x muchos (una columna por muchas filas o 
#          una fila x muchas columnas)
# X = dataset.iloc[:, 1].values # vector
# Matriz: un array de 'n' filas por 'm' columnas.
# X = dataset.iloc[:, 1:2].values # matriz
X = dataset.iloc[:, [2,3]].values # vector
Y = dataset.iloc[:, 4].values # vector

# Dividir los datos en 80% en entrenamiento y 20% en test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# Escalar datos 
from sklearn.preprocessing import StandardScaler # 'StandardScaler' es una Clase de sklearn
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Ajustar el clasificador al conjunto de entrenamiento
# Crear modelo de clasificacion
classifier = 0 # Editar por que debe ser un objeto complejo


# Prediccion
Y_pred = classifier.predict(X_test)


# Matriz de Confusion para visualizar los datos
# esta matriz muestra la cantidad de datos correcto y errores del modelo, asi podemos aproximar
# un % de eficacia del modelo en la tarea dew prediccion.
from sklearn.metrics import confusion_matrix # 'confusion_matrix' es una funcion
cm = confusion_matrix(Y_test, Y_pred)


# Representacion Grafica del Conjunto de Entrenamiento
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador')
plt.xlabel('Edad')
plt.ylabel('Sueldos estimados')
plt.legend()
plt.show()