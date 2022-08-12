#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 02:45:05 2022

@author: allancortez
"""

# Regresion con Arboles de Desicion

# Librerias
import numpy as np # Libreria de matematicas para los algoritmos MLeaning
import matplotlib.pyplot as plt # Diseño de graficos
import pandas as pd # Carga de datos (como para big data)

# importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
# Vector: lista de 1 x muchos (una columna por muchas filas o 
#          una fila x muchas columnas)
# X = dataset.iloc[:, 1].values # vector
# Matriz: un array de 'n' filas por 'm' columnas.
X = dataset.iloc[:, 1:2].values # matriz
Y = dataset.iloc[:, 2].values # vector

"""
# Dividir los datos en 80% en entrenamiento y 20% en test.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
"""

"""
# Scalar datos.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
"""

# Ajustar regresion con el dataset
# Documentacion Web : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state = 0)
regression.fit(X, Y)

# Prediccion para un dato de entrada
#Y_pred = regression.predict(6.5)


# Parrilla de datos que hacen de la funcion mas fina (no construida por rectas)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

# Visualizacion del modelo de Regresion
plt.scatter(X, Y, color = "red")
# python 3.7++
# lin_reg_2.predict([[X]])
# Forma de construccion fina de la funcion
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
# Forma bruta (funcion construida por rectas)
# plt.plot(X, lin_reg_2.predict(X_poly), color = "blue")
# Forma de construccion diferente pero igual
# plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Sueldos vs Años de Experiencia")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldos")
plt.show()

