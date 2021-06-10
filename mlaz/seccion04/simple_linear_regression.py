# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:04:26 2021

@author: gabriell
"""

# === EJERCICIO DE REGRESIÓN LINEAL SIMPLE ===

# agrega al path de python la ruta hacia las librerias definidas
import sys
sys.path.insert(0, "V:/ai/lib/")

# importar el módulo que contiene todas las funciones de preprocesado
import preprocessing

# cargar el archivo, son dos vectores
x, y = preprocessing.vectorizeCsv("Salary_Data.csv")

# no hay valores nulos

# en el caso de la regresión lineal simple no es necesario el escalado

# dividir el set de datos
x_train, x_test, y_train, y_test = preprocessing.splitTrainTestVal(x, y, randomState = 0, trainSize = 2.05/3)

# crear el modelo de regresión
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train) # se generó el modelo que representa a los datos de entrenamiento

# predecir el conjunto de test
y_pred = lr.predict(x_test)

# visualización de los resultados del conjunto de entrenamiento
import matplotlib.pyplot as plt
plt.scatter(x_train, y_train, color = "red") # nube de puntos
plt.plot(x_train, lr.predict(x_train), color = "blue") # la línea de recta de regresión
plt.title("Regresión lineal simple - Training Set")
plt.xlabel("Años")
plt.ylabel("Salario")
plt.show()

# visualización de los resultados del conjunto de test
import matplotlib.pyplot as plt
plt.scatter(x_test, y_test, color = "red") # nube de puntos
# la línea de recta de regresión es única, es la misma que la de entrenamiento
plt.plot(x_train, lr.predict(x_train), color = "blue")
plt.title("Regresión lineal simple - Test Set")
plt.xlabel("Años")
plt.ylabel("Salario")
plt.show()