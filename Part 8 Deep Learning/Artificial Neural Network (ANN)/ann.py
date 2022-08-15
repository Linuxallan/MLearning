#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 23:03:42 2022

@author: allancortez
"""

# Redes Neuronales Artificales

# ----------------------------------------------------------------------------
# Instalar scikit-learn
# pip install -U scikit-learn ||ver_version|| python -m pip show scikit-learn

# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras
# conda create -n [env_name] tensorflow python=3.7 ||ver_vesion|| pip show tensorflow
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Parte 1 - Pre procesado de datos
# ----------------------------------------------------------------------------

# Librerias
# conda install -c anaconda numpy  
import numpy as np # Libreria de matematicas para los algoritmos MLeaning
# conda install -c conda-forge matplotlib
import matplotlib.pyplot as plt # Diseño de graficos
# conda install pandas 
import pandas as pd # Carga de datos (como para big data)


# importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')
# Vector: lista de 1 x muchos (una columna por muchas filas o 
#          una fila x muchas columnas)
# X = dataset.iloc[:, 1].values # vector
# Matriz: un array de 'n' filas por 'm' columnas.
# X = dataset.iloc[:, 1:2].values # matriz
X = dataset.iloc[:, 3:13].values # vector
Y = dataset.iloc[:, 13].values # vector


# Codificar Datos Categoricos (Variables Dummy) transformas a matriz de int una 
# columna de datos strings.
from sklearn import preprocessing
# Dummyzar la columna 'Geography'
geography_dummy_X = preprocessing.LabelEncoder()
X[:,1] = geography_dummy_X.fit_transform(X[:,1])
# Dummyzar la columna 'Gender'
gender_dummy_X = preprocessing.LabelEncoder()
X[:,2] = gender_dummy_X.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
     
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]


# Dividir los datos en 80% en 'Training' y 20% en 'Testing'.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# Escalar Variables 
from sklearn.preprocessing import StandardScaler # 'StandardScaler' es una Clase de sklearn
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Parte 2 - Construir RNA (Red Neuronal Artificial)
# ----------------------------------------------------------------------------

# Importar Librerias 
import keras
from keras.models import Sequential
from keras.layers import Dense


# Inicializar la RNA: ANN (Artificial Neural Network)
classifier = Sequential()

# Añadir 'Capa de Entrada' y la primera 'Capa Oculta'
classifier.add(Dense(units = 6,     # Agregar una 'Capa Oculta' de 6 'neuronas'. (6 = media entre Nº variables y Nº salidas(es opcional, lo ideal es experimentar)))
                     kernel_initializer = "uniform",  # Inicializar los pesos con valores pequeños cercanos a 0
                     activation = "relu",   # Funcion de Activacion
                     input_dim = 11))   #Agregar 'Capa de Entrada' (input) con 11 neuronas.

# Añadir una segunda 'Capa Oculta'
classifier.add(Dense(units = 6,     # Agregar una 'Capa Oculta' de 6 'neuronas'. (6 = media entre Nº variables y Nº salidas(es opcional, lo ideal es experimentar)))
                     kernel_initializer = "uniform",  # Inicializar los pesos con valores pequeños cercanos a 0
                     activation = "relu")) # Funcion de Activacion

# Añadir 'Capa de Salida'
classifier.add(Dense(units=1,
                     kernel_initializer="uniform",
                     activation="sigmoid"))

# Compilar la RNA (ANN)
# 'adam' es un algoritmo con el mismo proposito que los algoritmos de:
# Gradiente Descendiente, y Gradiente Descendiente Estocástico.
classifier.compile(optimizer = "adam", # Algoritmo que logra que los pesos de la red mejoren hasta ser efectivo. 
                   loss = "binary_crossentropy", # Algoritmo de 'Funcion de Perdida'
                   metrics = ["accuracy"]) # Metrica de 'Presicion' para evaluar el numero de predicciones correctar respecto al total

# Ajustar la RNA (ANN) al conjunto de Entrenamiento
classifier.fit(X_train, Y_train, 
               batch_size=10, # Cada 10 datos se evalua los pesos de la red.
               epochs=5)  # Iteraciones de 'Entrenamiento' sobre el conjunto de 'Entrenamiento'


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Parte 3 - Evaluar Modelo y Calcular Predicciones Finales
# ----------------------------------------------------------------------------

# Prediccion de los resultados con el conjunto de Testing
Y_pred_porcent = classifier.predict(X_test)
Y_pred_boolean = (Y_pred_porcent>0.5)


# Matriz de Confusion para visualizar los datos
# esta matriz muestra la cantidad de datos correcto y errores del modelo, asi podemos aproximar
# un % de eficacia del modelo en la tarea dew prediccion.
from sklearn.metrics import confusion_matrix # 'confusion_matrix' es una funcion
cm = confusion_matrix(Y_test, Y_pred_boolean)
print(cm)
