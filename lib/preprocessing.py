# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:14:15 2021

@author: gabriell
"""

import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

def vectorizeCsv(file):
    '''Importar el archivo file y devolver una matrix con las dimensiones y
    un vector con las etiquetas. Importante: las etiquetas siempre deben estar
    en la última columna del csv.'''
    
    # importar el csv a la variable dataset
    dataset = pd.read_csv("Data.csv")
    
    # se crea una matriz con todos los valores salvo la última columna del csv
    X = dataset.iloc[:,:-1].values

    # se crea un vector que contiene únicamente la última columna del csv
    y = dataset.iloc[:,-1].values

    return X, y
    
def fillNaN(matrix, strategy = "mean"):
    '''Transforma los valores NaN de acuerdo a la estrategia enviada'''
    
    # transforma los valores NaN que haya en el set de datos con la media
    # la media es la estrategia más común, puede ser la mediana o moda también
    # axis = 0 indica la media por columna, si se pone axis = 1 sería la de la fila
    imputer = Imputer(
                      missing_values = "NaN",
                      strategy = "mean",
                      axis = 0
                      )
    # aplica la configuración establecida a la matriz X, pero hay una columna
    # que tiene los nombres de los países, se debe omitir
    # se asigna a imputer de nuevo para no trabajarlo en una variable separada    
    imputer = imputer.fit(matrix)
    
    # las dos columnas (de todas las filas) que se van a transformar se reemplazan
    # en la matriz X original
    matrix = imputer.transform(matrix)
    
    return matrix

def createCategoricalFeatures(matrix, feature):
    '''Crea dimensiones categóricas de acuerdo a la dimensión enviada'''
    
    # convierte los textos a valores en la misma dimensión
    matrix[:,feature] = labelEncoder(matrix[:,feature])
    
    # se toma la misma dimensión y se crea una nueva dimensión por cada valor,
    # cada dimensión va a tener los valores 0 o 1 como flag booleano
    one_hot_encoder = OneHotEncoder(categorical_features=[feature])
    matrix = one_hot_encoder.fit_transform(matrix).toarray()
    
    return matrix
    
def labelEncoder(vector):
    '''Realiza la tarea de codificado de vector, de texto a valor'''
    
    # se conviernen los valores encontrados en la dimensión indicada a valores
    # numéricos que los representen, el resultado se almacena en la misma
    # dimensión
    label_encoder = LabelEncoder()
    vector = label_encoder.fit_transform(vector)
    
    return vector

def splitTrainTestVal(X, y, randomState = None, crossValidation = False, trainSize = 0.8):
    '''Divide la matrix y el vector en set de entrenamiento y test, si se 
    indica crossValidation, además devuelve el set de validación cruzada'''
    
    # genera el set de entrenamiento y el set de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - trainSize, random_state = randomState)
    
    # si no se indicó validación cruzada, devuelve solamente el set de
    # entrenamiento y el set de test
    if crossValidation == False:
        return X_train, X_test, y_train, y_test
    
    # si se indicó validación cruzada, divide el set de test en dos partes
    # iguales para separarlo entre set de test y set de validación cruzada
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = randomState)
    
    return X_train, X_test, y_train, y_test, X_val, y_val
    
def standardScaler(train, test):
    '''Escalado estándar'''
    scaler = StandardScaler()
    
    # se usa el escalado estándar para el conjunto de datos de entrenamiento
    train = scaler.fit_transform(train)
    
    # para el conjunto de test se usa el mismo escalado que para entrenamiento
    # con el fin de no tener dos escalados diferentes, por eso se usa transform
    # en lugar de fit_transform
    test = scaler.transform(test)
    
    return train, test