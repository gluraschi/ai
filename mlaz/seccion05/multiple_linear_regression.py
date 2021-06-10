# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 15:45:04 2021

@author: gabriell
"""

# === EJERCICIO DE REGRESIÓN LINEAL MÚLTIPLE ===

# agrega al path de python la ruta hacia las librerias definidas
import sys
sys.path.insert(0, "V:/ai/lib/")

# importar el módulo que contiene todas las funciones de preprocesado
import preprocessing

# cargar el archivo, son dos vectores
X, y = preprocessing.vectorizeCsv("50_Startups.csv")

# crear las dummy a partir de la categórica
X = preprocessing.createDummyFeatures(X, 3, removeOneDummy=True)

# dividir conjunto de entrenamiento y de testing
X_train, X_test, y_train, y_test = preprocessing.splitTrainTestVal(X, y, randomState=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# predecir el conjunto de test
y_pred = lr.predict(X_test)

# construir el modelo óptimo mediante eliminación hacia atrás
import featureselection
X_opt = featureselection.backguard(X, y)
    




























