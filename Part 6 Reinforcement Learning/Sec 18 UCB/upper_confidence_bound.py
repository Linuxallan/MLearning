#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 18:58:59 2022

@author: allancortez
"""

# Upper Confidence Bound Algorithm

# Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar Data set
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Algoritmo de UCB
import math
N = 10000
D = 10 # (nª variables del dataset)
# Vectores
number_of_selections = [0] * D
sums_of_rewards = [0] * D
ads_selected = []
total_reward = 0
for n in range(0, N):
    
    max_upper_bound = 0
    ad = 0
    for i in range(0, D):
        
        if number_of_selections[i] > 0 :
            # Calcular valor medio
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            # Calcular intervalo de confianza
            delta_i = math.sqrt(3/2 * math.log(n+1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
# Visualizar datos en un Histogram
plt.hist(ads_selected)
plt.title("Histogram of Ads")
plt.xlabel("Ads ID")
plt.ylabel("Visualization Frecuency")
plt.show()
