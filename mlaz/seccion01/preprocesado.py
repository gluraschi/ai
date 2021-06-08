# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 15:58:39 2021

@author: gabriell
"""
# === EJERCICIO DE PREPROCESAMIENTO ===

# agrega al path de python la ruta hacia las librerias definidas
import sys
sys.path.insert(0, "V:/ai/lib/")

# importar el módulo que contiene todas las funciones de preprocesado
import preprocessing

# crear matrix y vector a partir de un csv
X, y = preprocessing.vectorizeCsv("Data.csv")

# llenar los vacíos pero omitiendo las columnas de texto
X[:,1:] = preprocessing.fillNaN(X[:,1:])

# se codifican los datos categóricos
X = preprocessing.createCategoricalFeatures(X, 0)

# hacer lo mismo con el vector y
y = preprocessing.labelEncoder(y)

# dividir el dataset en conjunto de entrenamiento y conjunto de testing
X_train, X_test, y_train, y_test = preprocessing.splitTrainTestVal(X, y, randomState = 0)

# escalado de datos
X_train, X_test = preprocessing.standardScaler(X_train, X_test)
