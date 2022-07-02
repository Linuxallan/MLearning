#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:18:44 2022

@author: allancortez
"""

# Plantilla pre procesado

# Librerias
import numpy as np # Libreria de matematicas para los algoritmos MLeaning
import matplotlib.pyplot as plt # Diseño de graficos
import pandas as pd # Carga de datos (como para big data)

"""
Leer excel, csv he integrandolos como un grid
"""
# importar el data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

"""
Si encuentra algun valor null, entnces lo setea con el valor promedio
de todos los datos.
"""
# Tratamiento de datos N/A
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

""" Tratamiento de datos N/A
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
"""

"""
Este transforma valores string relacionandolos a un indoce de un array
por lo que da una alor numero a un string especifico
"""
# Codificar datos categoricos (categorizar string en int, creando mas columnas que hacer referencia al valor string)
from sklearn import preprocessing
le_X = preprocessing.LabelEncoder()
X[:,0] = le_X.fit_transform(X[:,0])
le_Y = preprocessing.LabelEncoder()
Y = le_Y.fit_transform(Y)

"""
# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
"""

"""
Este metodo lo que hace es crear una columna con valor int que hace referencia
a un valor de string, una columna para cada valor string y entonces si el valor
es 1  quiere decir que el valor string es uno especifico, si es cero es por que
no corresponde el string asociado
"""
# One Hot Encoder y crear variables Dummy (dividir en 3 culumnas int una columna con datos string)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
     
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)

# Dividir los datos en 80% en entrenamiento y 20% en test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

"""
### Scalar datos, 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
"""


