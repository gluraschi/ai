# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:14:15 2021

@author: gabriell
"""

import statsmodels.formula.api as sm
import numpy as np

def backguard(X, y, sl = 0.05):
    '''
    X = matriz
    sl = umbral de significación estadística
    '''
    
    # agregar unos al principio de la matriz
    X = addOnes(X)
    
    # crear una nueva matriz que contenga las características de mayor
    # significación estadística con la que vamos a trabajar
    X_opt = X
    
    while True:
        # crear un nuevo regresor con todos los nuevos datos pero desde otra librería
        # ordinary list square
        # endog = variable endógena, la que se desea predecir
        # exog = variable exógena, las caracteríticas
        # el .fit() genera que se haga el fit sobre los datos
        lr_ols = sm.OLS(endog=y, exog=X_opt).fit()
        
        # ver un resumen completo
        #lr_ols.summary() 
        
        # obtener el valor máximo de p-values
        max_pvalue = max(lr_ols.pvalues)
        
        # si el valor de significación más elevado no supera el umbral
        # establecido, es porque ya tenemos el modelo listo
        if max_pvalue < sl:
            break;
        
        # obtener el índice correspondiente al valor máximo
        pvalue_index = lr_ols.pvalues.argmax()
        
        # removerlo de la matrix, con axis = 1 se indica que quite la columna 
        # pvalue_index de X_opt
        X_opt = np.delete(X_opt, pvalue_index, axis = 1)
        
    # devolver la matriz que contenga únicamente los valores más significativos
    return X_opt
        

def addOnes(X):
    # determinar la cantidad de filas que hay en la matriz
    rows = X.shape[0]

    # crear un vector de unos de tipo int
    ones = np.ones([rows, 1]).astype(int)
    
    # agregar al vector la matriz X, se hace de esta forma para que el
    # vector quede al principio de la nueva matriz X
    X = np.append(arr = ones, values = X, axis = 1)
    
    return X
        
        
        
        