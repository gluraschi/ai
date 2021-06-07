# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 15:58:39 2021

@author: gabriell
"""

# plantilla de pre-procesado
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

# importar el csv a la variable dataset
dataset = pd.read_csv("Data.csv")

# se crea la matrix de dimensiones y el vector de salida
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# transformación de los valores NaN en la media de la columna
# la media es la estrategia más común, puede ser la mediana también
# axis = 0 indica la media por columna, si se pone axis = 1 sería la de la fila
imputer = Imputer(
                  missing_values = "NaN", 
                  strategy = "mean",
                  axis = 0)
# aplica la configuración establecida a la matriz X, pero hay una columna
# que tiene los nombres de los países, se debe omitir
# se asigna a imputer de nuevo para no trabajarlo en una variable separada
imputer = imputer.fit(X[:,1:])

# las dos columnas (de todas las filas) que se van a transformar se reemplazan
# en la matriz X original
X[:,1:] = imputer.transform(X[:,1:])

# codificar datos categóricos
label_encoder_X = LabelEncoder()
X[:,0] = label_encoder_X.fit_transform(X[:,0])

# codificar los datos categóricos en diferentes columnas, primero se configura
# la columna 0 y después se pasa todo X total ya está configurada
# siempre hay que pasarlo a número con labelenconder o dará error
one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()
X

# hacer lo mismo con el vector y
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# dividir el dataset en conjunto de entrenamiento y conjunto de testing
# ojo, debería haber un tercer grupo que es el de validación cruzada
# se usa un random state fijo para que siempre divida igual los datos
X_train, X_test, y_train, y_test = train_test_split(
                                                    X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# escalado de datos, el conjunto de entrenamiento se escala pero el de test
# se escala con los mismos valores que el de entrenamiento para evitar
# dos escalados diferentes
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)