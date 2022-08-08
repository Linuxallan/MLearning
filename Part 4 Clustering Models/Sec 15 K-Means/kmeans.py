#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 22:07:46 2022

@author: allancortez
"""

# K-Means Algorithm : Algoritmo de Clusterizacion 

# Importar Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carga de Datos con pandas : Data Set
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

# Tecnica del codo para averiguar el numero optimo de 'clusters'
from sklearn.cluster import KMeans
# wcss (formula matematica) : Within-Cluster-Sum-of-Squares
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    print("W")

plt.plot(range(1,11), wcss)
plt.title("Método del codo")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS(k)")
plt.show()

# Aplicar KMeans para segmentar ek data set
kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10, random_state=0)
Y_pred = kmeans.fit_predict(X)


# Visualizacion de los Clusters
plt.scatter(X[Y_pred == 0, 0], X[Y_pred == 0, 1], s = 100, c = "red", label = "Cluster 1")
plt.scatter(X[Y_pred == 1, 0], X[Y_pred == 1, 1], s = 100, c = "blue", label = "Cluster 2")
plt.scatter(X[Y_pred == 2, 0], X[Y_pred == 2, 1], s = 100, c = "green", label = "Cluster 3")
plt.scatter(X[Y_pred == 3, 0], X[Y_pred == 3, 1], s = 100, c = "cyan", label = "Cluster 4")
plt.scatter(X[Y_pred == 4, 0], X[Y_pred == 4, 1], s = 100, c = "magenta", label = "Cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c="black", label="Baricentros")
plt.title("Clusters de Clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuacion de gastos (1-100)")
plt.legend()
plt.show()