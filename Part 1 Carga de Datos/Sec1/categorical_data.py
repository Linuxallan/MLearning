#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:52:03 2022

@author: allancortez
"""

# Plantilla pre procesado CATEGORICO

# Librerias
import numpy as np # Libreria de matematicas para los algoritmos MLeaning
import matplotlib.pyplot as plt # Diseño de graficos
import pandas as pd # Carga de datos (como para big data)

# importar el data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Codificar datos categoricos (categorizar string en int, creando mas columnas que hacer referencia al valor string)
from sklearn import preprocessing
le_X = preprocessing.LabelEncoder()
X[:,0] = le_X.fit_transform(X[:,0])
le_Y = preprocessing.LabelEncoder()
Y = le_Y.fit_transform(Y)

# One Hot Encoder y crear variables Dummy (dividir en 3 culumnas int una columna con datos string)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
     
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)