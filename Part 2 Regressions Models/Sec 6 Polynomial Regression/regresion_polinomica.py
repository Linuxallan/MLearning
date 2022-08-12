#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 23:41:24 2022

@author: allancortez
"""

# Regresion Polinomica

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

# Ajustar regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Ajustar regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
# Por defecto es 2: (nivel 2 en exponencial)
# poly_reg = PolynomialFeatures(degree = 2) # grado: 2
poly_reg = PolynomialFeatures(degree = 6) # (grado) Exponencial: 6 (optimo en este caso)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualizacion del modelo Lineal
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
# python 3.7++
# lin_reg.predict([[X]])
# plt.plot(X, lin_reg.predict([[X]]), color = "blue") # Ej.
plt.title("Modelo de Regresion Lineall")
plt.xlabel("Niveless")
plt.ylabel("Sueldoss")
plt.show()

# Parrilla de datos que hacen de la funcion mas fina (no construida por rectas)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
# Visualizacion del modelo Polinomico
plt.scatter(X, Y, color = "red")
# Forma de construccion fina de la funcion
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
# Forma bruta (funcion construida por rectas)
# plt.plot(X, lin_reg_2.predict(X_poly), color = "blue")
# Forma de construccion diferente pero igual
# plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Sueldos vs Años de Experiencia")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldos")
plt.show()

# Prediccion para un dato de entrada
# Ej: Se quiere saber que sueldo se le debe pagar a un empleado de nivel 6.4
# Prediccion Lineal
lin_reg.predict(6.4)
# Prediccion Polinomica
lin_reg_2.predict(poly_reg.fit_transform(6.4))