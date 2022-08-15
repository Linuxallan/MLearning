#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 00:21:26 2022

@author: allancortez
"""

# Redes Neuronales Convolucionales
# Convolucion : Multiplicacion de Matrices (pueden ser de dimensiones diferentes)


# ----------------------------------------------------------------------------
# Instalar scikit-learn
# pip install -U scikit-learn ||ver_version|| python -m pip show scikit-learn

# Instalar Theano
# conda install git
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# SI NO: pip install --upgrade --no-deps git+https://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras
# conda create -n [env_name] tensorflow python=3.7 ||ver_vesion|| pip show tensorflow
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Parte 1 - Contruir el modelo CNN (Convolutional Neural Network)
# ----------------------------------------------------------------------------

# Importar Librerias y paquetes
#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# Inicializar la CNN (Convolutional Neural Network)
classifier = Sequential()

# Añadir 'Capa de Entrada' y la primera 'Capa Oculta'
# Paso 1 - Convolucion  
# Crear la capar de 'Mapa de Caracteristicas' convolucionando
# la imagen original con una matriz especificada ( (3, 3) ), y operacion especificada (relu).
classifier.add(Conv2D(filters = 32,     # 32 Mapas de Caracteristicas
                      kernel_size = (3, 3),  # Matriz de convolucion 
                      input_shape = (64, 64, 3), # Conservar imagenes de 64x64 pixeles en 3 niveles de color.
                      activation = "relu")) 

# Paso 2 - Max Pooling 
# Crear la capa 'Pooling' convolucionado la 'Capa de Caracteristicas' con una matriz
# especificada ( (2, 2) )
classifier.add(MaxPooling2D(pool_size=(2, 2), ))

# Paso 3 - Flattening (convertir la 'Matriz' final en un 'Vector'(lista)) : 
# es para adaptar los datos del ultimo nodo con la Red Neuronal
classifier.add(Flatten())


# Paso 4 - Full Connection : Creacion de la Red Neuronal
classifier.add(Dense(units = 128, #Agregar 'Capa Oculta' con 128 neuronas.
                     activation = "relu")) # Funcion de Activacion
classifier.add(Dense(units = 1, #Agregar 'Capa de Salida' con 1 neuronas.
                     activation = "sigmoid")) # Funcion de Activacion
                     

# Paso 5 - Compilar la CNN
classifier.compile(optimizer="adam",    # Algoritmo para optimizar los pesos de la Red.
                   loss = "binary_crossentropy", # Algoritmo para extraer valor de perdida.
                   metrics=["accuracy"])  # Metrica de 'Presicion' para evaluar el numero de predicciones correctar respecto al total



# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Parte 3 - Ajustar la CNN a las imagenespara entrenar
# ----------------------------------------------------------------------------

# Importar Librerias
from keras.preprocessing.image import ImageDataGenerator

# Extraer y ajustar set de imagenes de Training y Testing
train_datagen = ImageDataGenerator(rescale = 1./255, # escalar
                                  shear_range = 0.2, # rotacion
                                  zoom_range = 0.2, # zoom
                                  horizontal_flip = True) # giro horizontal

test_datagen = ImageDataGenerator(rescale = 1./255)

training_dataset = train_datagen.flow_from_directory('dataset/training_set', # url donde estan el set de entrenamiento
                                                    target_size = (64, 64),
                                                    batch_size = 32, # cada 32 tomas de datos se efectua una funcion
                                                    class_mode = "binary") # solo dos clases a predecir

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = "binary")

classifier.fit_generator(training_dataset, # set de training
                    steps_per_epoch = 8000, # datos a esperar en el set de training
                    epochs = 2, # ciclos (iteracion) de entrenamiento
                    validation_data = testing_dataset, # set de testing
                    validation_steps = 2000) # datos a esperar en el set de testing

