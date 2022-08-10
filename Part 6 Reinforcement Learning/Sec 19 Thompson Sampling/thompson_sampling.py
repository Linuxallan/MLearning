#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 00:30:43 2022

@author: allancortez
"""

# Muestreo Thompson

# Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar Data set
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Algoritmo de Muestreo Thompson
# import math
import random
N = 10000
D = 10 # (nª variables del dataset)
# Vectores
number_of_rewards_1 = [0] * D
number_of_rewards_0 = [0] * D
ads_selected = []
total_reward = 0
for n in range(0, N):
    
    max_random = 0
    ad = 0
    for i in range(0, D):
        
        random_beta = random.betavariate(number_of_rewards_1[i]+1, number_of_rewards_0[i]+1)
        
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1[ad] =  number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
    
# Visualizar datos en un Histogram
plt.hist(ads_selected)
plt.title("Histogram of Ads")
plt.xlabel("Ads ID")
plt.ylabel("Visualization Frecuency")
plt.show()
