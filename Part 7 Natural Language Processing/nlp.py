#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 22:07:59 2022

@author: allancortez
"""

# Natural Language Processing Algorithm

# Importar Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar Data Set
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

# Limpieza de DataSet (quitar los signos y caracteres gramaticos a las fraces de  palabras)
import re # Regular Expression. 'Expresiones Regulares'
import nltk # Natural Language Toolkit.
# Descargar diccionario de parabras 'Malas' que no sirven, asi busco estas palabras
# en el dataset para borrarlas por ser inutiles (de todos los idiomas del mundo).
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    # Quedarme solo con caracteres de la a-z (minusculas) y A-Z (mayusculas)
    # remplazandolas los caracteres que no quiero por: ' '.
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # Pasar todo a minusculas
    review = review.lower()
    # Quitar espacios de una frace y agregar las palabras a un array
    review = review.split()
    # Estemizar: identificar la raiz gramatical de una palabra para re setear la variable
    # --> esto es para reducir la lista de palabras del conjunto a valores raiz, sin tiempo y conjugaciones, etc.
    ps = PorterStemmer()
    # Comparar cada palabra del array con las palabras inutiles de la libreria 'nltk'
    # y solo guardar las palabras que no coincidan en ambos lados.
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Transformar un array de string a un solo string separados por un: ' '.
    review = ' '.join(review)
    corpus.append(review)
    # dataset['Review'][i] = review

# Crear un Bag of Words : Saco de Palabras
# Mover las palabras de cada frase a un array y no permitir el duplicado de palabras
# en la misma fila del array.
from sklearn.feature_extraction.text import CountVectorizer
# Transformar una frase a vector: word to vector
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values


# ------------------------------------------------
# Matriz de Confusion para Visualizar los datos
# NAIVE BAYES ALGORITHM (classification secction)

# Dividir los datos en 80% en entrenamiento y 20% en test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

"""
# Escalar datos 
from sklearn.preprocessing import StandardScaler # 'StandardScaler' es una Clase de sklearn
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Ajustar el clasificador al conjunto de entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)


# Prediccion
Y_pred = classifier.predict(X_test)


# Matriz de Confusion para visualizar los datos
# esta matriz muestra la cantidad de datos correcto y errores del modelo, asi podemos aproximar
# un % de eficacia del modelo en la tarea dew prediccion.
from sklearn.metrics import confusion_matrix # 'confusion_matrix' es una funcion
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
