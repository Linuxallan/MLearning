#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 19:11:48 2022

@author: allancortez
"""

# Generar aleatoriamente un Data Set con 1000 filas de datos y 10 columnas de variables
# con 0 y 1 aleatorios. Los 0 y 1 representan si un usuario hizo click en un anuncio o no.

# Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar Data set
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Random Selection Implementing
import random as rd
N = 10000
D = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = rd.randrange(D)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
    
# Visualizar los datos
plt.hist(ads_selected)
plt.title("Histogram of Ads Selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()