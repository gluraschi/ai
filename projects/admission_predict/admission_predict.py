# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:52:54 2021

@author: gabriell
"""

# agrega al path de python la ruta hacia las librerias definidas
import sys
sys.path.insert(0, "V:/ai/lib/")

# importar el módulo que contiene todas las funciones de preprocesado
import preprocessing

# cargar el archivo, son dos vectores
X, y = preprocessing.vectorizeCsv("Admission_Predict_Ver1.1.csv")

# la primer columna de x no es necesaria, se quita del set de datos
X = X[:,1:]

# completar valores NaN
X = preprocessing.fillNaN(X)

# escalado de características
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import Normalizer
# normalizer termina haciendo un gráfico más bien redondo con lo cual se pierde
# la noción de la distribución de características, la línea de regresión termina
# siendo algo inexacto
ss = StandardScaler()
X = ss.fit_transform(X)

# selección de la característica a usar para la regresión lineal simple
x = X[:,0:1]

# visualización previa
import matplotlib.pyplot as plt
plt.scatter(x, y, color = "red") # nube de puntos
#plt.plot(x_train, lr.predict(x_train), color = "blue") # la línea de recta de regresión
plt.title("Nube de Puntos - StandardScaler")
plt.xlabel("GRE Score")
plt.ylabel("Chance of Admit")
plt.show()

# divido el set de entrenamiento y el set de test
x_train, x_test, y_train, y_test = preprocessing.splitTrainTestVal(x, y)

# entreno el modelo con el set de entrenamiento
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# visualización del conjunto de entrenamiento
import matplotlib.pyplot as plt
plt.scatter(x, y, color = "red") # nube de puntos
plt.plot(x_train, lr.predict(x_train), color = "blue") # la línea de recta de regresión
plt.title("Nube de Puntos - Entrenamiento")
plt.xlabel("GRE Score")
plt.ylabel("Chance of Admit")
plt.show()

# grafica con los puntos de entrenamiento y la línea de regresión
import matplotlib.pyplot as plt
plt.scatter(x_test, y_test, color = "red") # nube de puntos
plt.plot(x_train, lr.predict(x_train), color = "blue") # la línea de recta de regresión
plt.title("Testing set")
plt.xlabel("GRE Score")
plt.ylabel("Chance of Admit")
plt.show()