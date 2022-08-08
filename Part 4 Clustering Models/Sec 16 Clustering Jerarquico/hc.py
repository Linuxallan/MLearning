#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 00:00:40 2022

@author: allancortez
"""

# Clustering Jerarquico

# Importar Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carga de Datos con pandas : Data Set
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

# Usar Dendrograma para averiguar numero optimo de Clusteres
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method="ward"))

plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclidea")
plt.show()

# Alustar el Clustering Jerarquico a nuestro data set
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
Y_pred = hc.fit_predict(X)

# Visualizacion de los Clusters
# Visualizacion de los Clusters
plt.scatter(X[Y_pred == 0, 0], X[Y_pred == 0, 1], s = 100, c = "red", label = "Cluster 1")
plt.scatter(X[Y_pred == 1, 0], X[Y_pred == 1, 1], s = 100, c = "blue", label = "Cluster 2")
plt.scatter(X[Y_pred == 2, 0], X[Y_pred == 2, 1], s = 100, c = "green", label = "Cluster 3")
plt.scatter(X[Y_pred == 3, 0], X[Y_pred == 3, 1], s = 100, c = "cyan", label = "Cluster 4")
plt.scatter(X[Y_pred == 4, 0], X[Y_pred == 4, 1], s = 100, c = "magenta", label = "Cluster 5")
plt.title("Clusters de Clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuacion de gastos (1-100)")
plt.legend()
plt.show()