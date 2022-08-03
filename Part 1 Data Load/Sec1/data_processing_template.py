#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:55:34 2022

@author: allancortez
"""

# Plantilla pre procesado

# Librerias
import numpy as np # Libreria de matematicas para los algoritmos MLeaning
import matplotlib.pyplot as plt # Diseño de graficos
import pandas as pd # Carga de datos (como para big data)

# importar el data set
dataset = pd.read_csv('Data.csv')
# Vector: lista de 1 x muchos (una columna por muchas filas o 
#          una fila x muchas columnas)
# X = dataset.iloc[:, 1].values # vector
# Matriz: un array de 'n' filas por 'm' columnas.
# X = dataset.iloc[:, 1:2].values # matriz
X = dataset.iloc[:, :-1].values # vector
Y = dataset.iloc[:, 3].values # vector

# Dividir los datos en 80% en entrenamiento y 20% en test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

"""
### Scalar datos, 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # fit_transform : crear escalado y aplicarlo (transform)
X_test = sc_X.transform(X_test) # transform : solo se aplica (transform)
"""


