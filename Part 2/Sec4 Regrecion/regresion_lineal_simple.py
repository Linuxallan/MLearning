#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 18:16:42 2022

@author: allancortez
"""

# Plantilla pre procesado

# Librerias
# import numpy as np # Libreria de matematicas para los algoritmos MLeaning
import matplotlib.pyplot as plt # Diseño de graficos
import pandas as pd # Carga de datos (como para big data)

# importar el data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Dividir los datos totales del Dataset en conjuntos diferentes para entrenamiento y pruebas
# X_train, Y_train = datos para el entrenamiento
# X_test, Y_test = datos para las pruebas
# PD: al dividir los datos del DataSet estos se desordenan, lo cual esta muy bien
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state=0)

"""
### Scalar datos, 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
"""

# Crear modelo de Regresion Lineal con los datos del data set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)

# Predecir modelo de test
# el predict(solo lleva los datos dependientes, No los independientes)
Y_pred = regression.predict(X_test)

# Visualizar en diagrama los datos del entrenamiento
plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color= "blue")
plt.title("Sueldo vs años de experiencia")
plt.xlabel("Años de experiencia")
plt.ylabel("sueldos")
plt.show()

# Visualizar en diagrama los datos del test
plt.scatter(X_test, Y_test, color = "green")
plt.plot(X_train, regression.predict(X_train), color= "blue")
plt.title("Sueldo vs años de experiencia")
plt.xlabel("Años de experiencia")
plt.ylabel("sueldos")
plt.show()