#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 21:33:47 2022

@author: allancortez
"""

# Regresion Lineal Multiple

# Librerias
import numpy as np # Libreria de matematicas para los algoritmos MLeaning
import matplotlib.pyplot as plt # Diseño de graficos
import pandas as pd # Carga de datos (como para big data)

# importar el data set
dataset = pd.read_csv('50_Startups.csv')
# Extraer en un DataSet 'X' todas las columnas a excepcion de la ultima '-1'
X = dataset.iloc[:, :-1].values
# Extraer en un Data Set solo la columna 4
Y = dataset.iloc[:, 4].values

# Categoriza variable Dummy (int) para columna string del DataSet (Le asocia
# un valor int a un valor string especifico)
from sklearn import preprocessing
le_X = preprocessing.LabelEncoder()
X[:,3] = le_X.fit_transform(X[:,3])

# One Hot Encoder y crear variables Dummy (dividir en N culumnas int una columna con datos string)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
     
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float64)

# Evitar trampa de variables Dummy (eliminar una si es mayor o igual a 2)
X = X[:, 1:]

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

# Modelar Regresion Lineal Multiple del DataSet train y test
# 'fit' reconoce y transforma automaticamente s se requiere regression simple o multiple
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)

# Prediccion de los datos
Y_pred = regression.predict(X_test)

# Encontrar modelo optimo con el metodo: Eliminacion hacia atras
# import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm
# 'X' va a ser igual a una columna de 1 (50 filas y una columna de 1) de tipo int + 
# los valores originales de 'X' (asi se deja la columna de 1 al inicio).
# axis = 0 --> fila   ||    axis = 1 --> columna
X = np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)
# valor de significancia del P-Valor: si es mayor estonces tiene poca significancia
SL = 0.05
# Filtrar solo la variables estadisticamente significativas para ser capaces de 
# predecir la variables independientes
# todas las variables son consideradas al inicio (para que el sistema valla evaluandolas)
X_opt = X[:, [0,1,2,3,4,5]].tolist()
# Identificar las variables con SL mayor y menos a 0.05 y re-crear un modelo 
# de regresion lineal multiple (fit())
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regression_OLS.summary()


### Se iran eliminando hacia atras dependiendo si el P-Valor es superior al
### LS definido : 0.05
X_opt = X[:, [0,1,2,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,1,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3,5]].tolist()
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3]].tolist()
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regression_OLS.summary()
### en este punto los P_Valor son inferiores al SL: 0.05
### EL MODELO ESTA LISTO.


"""
    import statsmodels.formula.api as sm
    def backwardElimination(x, sl):    
        numVars = len(x[0])    
        for i in range(0, numVars):        
            regressor_OLS = sm.OLS(y, x.tolist()).fit()        
            maxVar = max(regressor_OLS.pvalues).astype(float)        
            if maxVar > sl:            
                for j in range(0, numVars - i):                
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                        x = np.delete(x, j, 1)    
        regressor_OLS.summary()    
        return x 
     
    SL = 0.05
    X_opt = X[:, [0, 1, 2, 3, 4, 5]]
    X_Modeled = backwardElimination(X_opt, SL)
    
    ### ---------------------
    
    import statsmodels.formula.api as sm
def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
"""