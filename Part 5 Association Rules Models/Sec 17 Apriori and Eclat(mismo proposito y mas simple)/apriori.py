#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 18:49:36 2022

@author: allancortez
"""

# Apriori Association Algorithm

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar DataSet
# En este data set el programa reconoce por defecto la primera fila como si fuera
# los nombres de las columnas (variables).
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None) # 'header=None' : le dice que la primera fila del data set es una fila de datos, no correspondiente a los nombres de las columnas (variables).

# Transformar los datos a lista de strings para una visualizacion de los datos como de Diccionario
transactions = []
for i in range(0, 7501):
    
    # Tiene datos 'nan' (nulos) la lista final de transacciones.
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

"""
# MI CREACION: sin datos 'nan' (nulos) de la lista final de datos.
for i in range(0, 7501):
    temp = []
    for j in range(0, 20):
        dato = str(dataset.values[i, j])
        
        if dato != "nan":
            temp.append(str(dataset.values[i, j]))
    
    transactions.append(temp)
"""

# Utilizar archivo adjunto que contiene el algoritmo de 'Apriori'
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
result = list(rules)

# Visualizar los datos
print(result[0])
print(result[153])