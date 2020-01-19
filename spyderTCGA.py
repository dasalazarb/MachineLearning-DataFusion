# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 17:45:44 2019

@author: dsala
"""

# %%
# ## Origen de los datos
# 
# Los datos provienen de __[FireHose Brad GDAC](https://gdac.broadinstitute.org/)__ y del paquete de R __[TCGA-Assembler](https://github.com/compgenome365/TCGA-Assembler-2)__. 
# 
# El siguiente documento vendra organizado por: <br>
# 1. Librerias de python y creadas
# 2. Carga y pre-procesamiento de datos
#	a. Cargar datos y preprocesamiento
#	b. Eliminacion variables con varianza cero
#	c. Seleccion de variables
#	d. Particion de train y test
#	e. Funcion evaluacion modelo en grilla
# 3. Modelos
#	a. Integración temprana
#		- modelos lineales
#		- modelos no-lineales
#		- modelos de ensamblaje
#		- ensamblaje de ensamblaje
#	b. Integracion intermedia
#		- Multi-modal NN mrna, cnv y meth
#		- Multi-modal NN mrna, cnv, meth y protein
#		- Multi-modal NN (mrna, cnv, meth) y protein
#	c. Integración tardía
#		- Stacking (SVR)
#		- Stacking (SVR) + NMF
#		- Stacking (SVR) + PCA

# %%
# *************************************************************************************
# ..................................... Librerias ................................... #
# *************************************************************************************
## Librerias generales
import os
import numpy as np
import pandas as pd

## Libreria para cargar datos
from LGG import loadTCGA 

## Libreria busqueda de hiperparametros, calculo de MSE y procesamiento de datos
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from TCGA_Integrator_data import normalizar, scalarRango, logPseudo, imputeSoft, varianceSelection, random_variables, train_test_profile

## Funciones para Integracion temprana - modelos lineales
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, MultiTaskLasso, ElasticNet, MultiTaskElasticNet, Lars, LassoLars, 
    OrthogonalMatchingPursuit, BayesianRidge, PassiveAggressiveRegressor, 
    TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.kernel_ridge import KernelRidge

## Funciones para Integracion temprana - modelos no-lineales
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

## Funciones para Integracion temprana - modelos de ensamblaje
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor

## Funciones para integracion intermedia
from nnTCGA import multiModelGenLayer, multiModelGenProtLayer, allToDNN
from numpy.random import seed; seed(1)
from tensorflow import set_random_seed; set_random_seed(2)

## Funciones para Integracion tardia
from StackingModelTCGA import Stacking
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import ParameterGrid

# Libreria para ocultar warnings
import warnings
warnings.filterwarnings("ignore")

# %%
# *************************************************************************************
# .................................... Cargar datos ................................. #
# *************************************************************************************

#_____________________________________________________#
## ************ a. Cargar y preprocesar ************ ##

## - Usando la libreria LGG cargue los datos.
## - El perfil de mrna se normalizo con logaritmo y escalandolo entre 0 y 1: log2(conteo + 1) -> mrna
## - Los demas perfiles se escalaron entre 0 y 1: min-max(cnv), min-max(meth) y min-max(protein)

# ruta donde estan almacenados los datos
path = "C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA/DeepTCGA"
# funcion para traer los datos listos para usar
tcga = loadTCGA(path)
# procesamiento de los perfiles excepto mrna
tcgaNew = {key: scalarRango(imputeSoft(value)) for key, value in tcga.items() if key != "lgg_mrnaTCGA" or key != "lgg_survivalTCGA"}
# procesmaiento del perfil de mrna
tcgaNew["lgg_mrnaTCGA"] = scalarRango(logPseudo(tcga["lgg_mrnaTCGA"]))

# ___________________________________________________________________ #
## ************ b. Eliminar variables con varianza cero ************ ##

# Eliminar variables con poca varianza
tcgaNew["lgg_cnvTCGA"]=varianceSelection(tcgaNew["lgg_cnvTCGA"], THRESHOLD=0.001)
tcgaNew["lgg_mrnaTCGA"]=varianceSelection(tcgaNew["lgg_mrnaTCGA"], THRESHOLD=0.001)
tcgaNew["lgg_methTCGA"]=varianceSelection(tcgaNew["lgg_methTCGA"], THRESHOLD=0.001)
tcgaNew["lgg_proteinTCGA"]=varianceSelection(tcgaNew["lgg_proteinTCGA"], THRESHOLD=0.001)

# ¿Como quedaron las dimensiones de cada perfil?
[print("key: {} -> value.shape: {} \n".format(k, v.shape)) for k, v in tcgaNew.items()]

# _____________________________________________________ #
## ************ c. Seleccion de variables ************ ##

# Seleccion de genes aleatorios
tcgaNew = random_variables(tcgaNew, 100)
tcga_train, tcga_test = train_test_profile(tcgaNew) ## si se escoge este caso partir en train y test set

# Seleccion de variables usando MalaCard Database
geneList = pd.read_csv(path+"/GeneListGlioma.csv") ## lista de genes obtenida en MalaCard
geneList = list(geneList.GeneSymbol) ## procesamiento en lista
lgg_names_0 = ["lgg_cnvTCGA", "lgg_mrnaTCGA", "lgg_methTCGA"] ## solo para estos tres perfiles
prefijos = {k: k.replace("lgg_", "").replace("hTCGA", "_").replace("TCGA", "_") for k in lgg_names_0} ## arreglar nombres para hacer busqueda
for k, v in prefijos.items(): ## crear nombres para hacer la interseccion
    prefijos[k] = [v+i for i in geneList]

prefijos = {k: list(set.intersection(set(v), list(tcgaNew[k].columns))) for k, v in prefijos.items()} ## escoger los nombres que se intersectan
prefijos["lgg_proteinTCGA"] = list(tcgaNew["lgg_proteinTCGA"].columns)[0:100] ## escoger primeros 100 proteinas de tcga
prefijos["lgg_survivalTCGA"] = list(tcgaNew["lgg_survivalTCGA"].columns) ## traer variable respuesta

tcgaNew = {k: v.loc[:, prefijos[k]] for k, v in tcgaNew.items()} ## traer los datos para los nombres intersectados

# ________________________________________________________ #
## ************ d. Particion en train y test ************ ##

tcga_train, tcga_test = train_test_profile(tcgaNew) ## si se escoge este caso partir en train y test set
lgg_names = ["lgg_cnvTCGA", "lgg_mrnaTCGA", "lgg_methTCGA", "lgg_proteinTCGA"]
X_train = pd.concat([tcga_train[i] for i in lgg_names], axis=1)
X_test = pd.concat([tcga_test[i] for i in lgg_names], axis=1)
y_train = tcga_train["lgg_survivalTCGA"].iloc[:,0].values
y_test = tcga_test["lgg_survivalTCGA"].iloc[:,0].values

# __________________________________________________________________ #
## ************ e. Funcion evaluacion modelo en grilla ************ ##
## (Se queda aqui para tener mas control sobre cv)
def modelos_ml(model, grilla, X_train, X_test, y_train, y_test):
    grid = GridSearchCV(model, param_grid=grilla, cv=5, refit=True, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("...................................................................")
    print("modelo: {} mse: {}".format(type(model).__name__, mse))
    print("mejores parametros: {}".format(grid.best_params_))
    print("................................................................... \n")
    return grid.best_estimator_, mse

# %%
# *************************************************************************************
# ...................................... Modelos .................................... #
# *************************************************************************************

#_____________________________________________________#
## ************ a. Integracion temprana ************ ##
### ... Modelos lineales ... ###
regressors = { ## modelos
    'OLS': LinearRegression(),
    'ridge': Ridge(), 
    'lasso': Lasso(),
    #'multi-lasso': MultiTaskLasso(), 
    'elasticnet': ElasticNet(), 
    #'multi-elasticnet': MultiTaskElasticNet(),
    'lars': Lars(), 
    'lassolars': LassoLars(), 
    'orthogonalmatchingpursuit': OrthogonalMatchingPursuit(), 
    'bayesianridge': BayesianRidge(), 
    'passiveaggressivregressor': PassiveAggressiveRegressor(), 
    'ransacregressor': RANSACRegressor(), 
    'theilsenregressor': TheilSenRegressor(), 
    'huberregressor': HuberRegressor()
    }

otra_param_grid = { ## grilla
    'OLS': {},
    "ridge": {'alpha': [0.01, 0.1, 1, 5, 100]},
    "lasso": {'alpha': [0.01, 0.1, 1, 5, 100]},
    #"multi-lasso": {'alpha': [0.01, 0.1, 1]}, 
    "elasticnet": {'alpha': [0.001, 0.05, 0.1, 1, 100], 'l1_ratio': [0.001, 0.05, 0.01, 0.1, 1, 100]}, 
    #"multi-elasticnet": {'alpha': [0.01, 0.1, 1], 'l1_ratio': [0.01, 0.1, 1]}, 
    "lars": {'n_nonzero_coefs': [20, 30, 50]}, 
    "lassolars": {'alpha': [0.01, 0.1, 1], 'fit_intercept': [True, False]}, 
    "orthogonalmatchingpursuit": {'n_nonzero_coefs': [10, 20, 30, 50]}, 
    #"bayesianridge": {'alpha_init': [0.01, 0.1, 1], 'lambda_init': [0.01, 0.1, 1]}, 
    "bayesianridge": {},
    "passiveaggressivregressor": {'C': [0.1, 1, 3, 10, 20]}, 
    "ransacregressor": {'min_samples': [0.5, 0.7, 0.85, 1]},
    "theilsenregressor": {}, 
    "huberregressor": {}
    }

## Evaluacion de grilla para modelos lineales
for key, value in regressors.items():
    modelos_ml(value, otra_param_grid[key], X_train, X_test, y_train, y_test)

### ... Modelos no-lineales ... ###
non_regressors = { ## modelos
    'svr': SVR(),
    'knn': KNeighborsRegressor(), 
    'gaussian': GaussianProcessRegressor(), 
    "tree": DecisionTreeRegressor(),
    'nn': MLPRegressor(), 
    'kernelridge': KernelRidge()
    }

non_params = { ## grilla
    'svr': {'kernel': ["linear","poly","rbf","sigmoid"], 'C': [0.1, 1, 3, 10, 20], 'degree': [2, 4, 6, 8], 'gamma': [0.001, 0.05, 0.01, 0.1, 1, 100]},
    'knn': {'n_neighbors': [5,10, 35,50,100], 'weights': ['uniform', 'distance'], 'leaf_size': [10, 30, 50], 'p': [1, 2]}, 
    'gaussian': {}, 
    'tree': {'max_depth': [2, 20, 100], 'min_samples_split': [2,10, 150], 'max_features': ['auto', 'sqrt', 'log2', 35]}, 
    'nn': {'hidden_layer_sizes': [(100,3), (200,30), (50,10), (100,50)], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'alpha': [0.0001, 0.001], 'early_stopping': [True]},
    "kernelridge": {'alpha': [0.001, 0.05, 0.1, 1, 100], 'kernel': ["linear","poly","rbf","sigmoid", "laplacian"], 'degree': [2, 4, 6, 8], 'gamma': [0.001, 0.05, 0.01, 0.1, 1, 100]}
    }

## Evaluacion de grilla para modelos no-lineales
for key, value in non_regressors.items():
    modelos_ml(value, non_params[key], X_train, X_test, y_train, y_test)

### ... Modelos de ensamblaje ... ###
ensembles = { ## modelos
    'randomforest': RandomForestRegressor(), 
    'bagging': BaggingRegressor(), 
    'adaboost': AdaBoostRegressor(), 
    #'voting': VotingRegressor(estimators=[('svr', SVR()), ('ridge', Ridge()), ('elastic', ElasticNet()), ('kernel', KernelRidge())]), 
    #'stacking': StackingRegressor(estimators=[('svr', SVR()), ('ridge', Ridge()), ('elastic', ElasticNet()), ('kernel', KernelRidge())])
    }

ensemble_params = { ## grilla
    'randomforest': {'n_estimators': [100, 200, 500], 'max_depth': [10, 25, 50, None], 'min_samples_split': [2, 10, 20, 30], 'max_features': ['auto', 'sqrt', 'log2', None]}, 
    'bagging': {'base_estimator': [SVR(C=0.1, degree=2, gamma=0.05, kernel='rbf'), Ridge(alpha=100), ElasticNet(alpha=0.05, l1_ratio=0.01), KernelRidge(alpha=100, degree=4, gamma=0.01, kernel="poly")], 'n_estimators': [10, 50], 'bootstrap_features': [True, False]}, 
    'adaboost': {}, 
    # 'voting': {'estimators': [('svr', SVR(C=0.1, degree=2, gamma=0.05, kernel='rbf')), 
    #                           ('ridge', Ridge(alpha=100)), ('elastic', ElasticNet(alpha=0.05, l1_ratio=0.01)), ('kernel', KernelRidge(alpha=100, degree=4, gamma=0.01, kernel="poly"))]},
    # 'stacking': {'estimators': [[('svr', SVR(C=0.1, degree=2, gamma=0.05, kernel='rbf')), 
    #                           ('ridge', Ridge(alpha=100)), ('elastic', ElasticNet(alpha=0.05, l1_ratio=0.01)), ('kernel', KernelRidge(alpha=100, degree=4, gamma=0.01, kernel="poly"))]], 'final_estimator': [RandomForestRegressor(), BaggingRegressor(), VotingRegressor()]}
    }

## Evaluacion de grilla para modelos de ensamblaje
for key, value in ensembles.items():
    modelos_ml(value, ensemble_params[key], X_train, X_test, y_train, y_test)

### ... Ensamblaje de ensamblajes ... ###
## Voting
model = VotingRegressor(estimators=[('svr', SVR()), ('ridge', Ridge()), ('elastic', ElasticNet()), ('kernel', KernelRidge())])

params_voting = {'svr__C': [0.1, 1, 3, 10, 20], 'svr__kernel': ["linear","poly","rbf","sigmoid"], 
                 'svr__degree': [2, 4, 6, 8], 'svr__gamma': [0.001, 0.05, 0.01, 0.1, 1, 100]}

grid = GridSearchCV(estimator=model, param_grid=params_voting, cv=10)

grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("...................................................................")
print("modelo: {} mse: {}".format(type(model).__name__, mse))
print("mejores parametros: {}".format(grid.best_params_))
print("................................................................... \n")

## Stacking
model = StackingRegressor(estimators=[('svr', SVR()), ('ridge', Ridge()), ('elastic', ElasticNet()), ('kernel', KernelRidge())])

params_stacking = {'svr__C': [0.1, 1, 3, 10, 20], 'svr__kernel': ["linear","poly","rbf","sigmoid"], 
                 'svr__degree': [2, 4, 6, 8], 'svr__gamma': [0.001, 0.05, 0.01, 0.1, 1, 100], 
                 'final_estimator': [RandomForestRegressor(), BaggingRegressor(), AdaBoostRegressor()]}

grid = GridSearchCV(estimator=model, param_grid=params_stacking, cv=10)

grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("...................................................................")
print("modelo: {} mse: {}".format(type(model).__name__, mse))
print("mejores parametros: {}".format(grid.best_params_))
print("................................................................... \n")

#_______________________________________________________#
## ************ b. Integracion intermedia ************ ##

### ... Multi-modal NN mrna, cnv y meth ... ###
model = multiModelGenLayer(tcgaNew["lgg_mrnaTCGA"], tcgaNew["lgg_cnvTCGA"], tcgaNew["lgg_methTCGA"])
consolidado = []
for j in range(1,10):
    for i in range(2):
        model.fit([tcga_train["lgg_mrnaTCGA"], tcga_train["lgg_cnvTCGA"], tcga_train["lgg_methTCGA"]], tcga_train["lgg_survivalTCGA"].iloc[:,0],
                  epochs=j, batch_size=10, verbose=0)
        
        y_pred = model.predict([tcga_test["lgg_mrnaTCGA"], tcga_test["lgg_cnvTCGA"], tcga_train["lgg_methTCGA"]])
        consolidado.append(mean_squared_error(tcga_test["lgg_survivalTCGA"].iloc[:,0], y_pred))
    print("epochs: " + str(j) + " Mean-> mseTest: " + str(np.mean(consolidado)) + " Variance: " + str(np.var(consolidado)))

### ... Multi-modal NN mrna, cnv, meth y protein ... ###
model = multiModelGenProtLayer(tcgaNew["lgg_mrnaTCGA"], tcgaNew["lgg_cnvTCGA"], tcgaNew["lgg_methTCGA"], tcgaNew["lgg_proteinTCGA"])
consolidado = []
for j in range(0,10):
    for i in range(3):
        model.fit([tcga_train["lgg_mrnaTCGA"], tcga_train["lgg_cnvTCGA"], tcga_train["lgg_methTCGA"], tcga_train["lgg_proteinTCGA"]], tcga_train["lgg_survivalTCGA"].iloc[:,0],
                  epochs=j, batch_size=10, verbose=0)
        
        y_pred = model.predict([tcga_test["lgg_mrnaTCGA"], tcga_test["lgg_cnvTCGA"], tcga_test["lgg_methTCGA"], tcga_test["lgg_proteinTCGA"]])
        consolidado.append(mean_squared_error(tcga_test["lgg_survivalTCGA"].iloc[:,0], y_pred))
    print("epochs: " + str(j) + " Mean-> mseTest: " + str(np.mean(consolidado)) + " Variance: " + str(np.var(consolidado)))

### ... Multi-modal NN (mrna, cnv, meth) y protein ... ###
model = allToDNN(tcga_train["lgg_mrnaTCGA"], tcga_train["lgg_cnvTCGA"], tcga_train["lgg_methTCGA"], tcga_train["lgg_proteinTCGA"], neurons=100, dropOut=.4, activation="tanh")
consolidado = []
for j in range(3,10):
    for i in range(3):
        model.fit([tcga_train["lgg_mrnaTCGA"], tcga_train["lgg_cnvTCGA"], tcga_train["lgg_methTCGA"], tcga_train["lgg_proteinTCGA"]], tcga_train["lgg_survivalTCGA"].iloc[:,0], epochs=j, verbose=0)
        
        y_pred = model.predict([tcga_test["lgg_mrnaTCGA"], tcga_test["lgg_cnvTCGA"], tcga_test["lgg_methTCGA"], tcga_test["lgg_proteinTCGA"]])
        consolidado.append(mean_squared_error(tcga_test["lgg_survivalTCGA"].iloc[:,0], y_pred))
    print("epochs: " + str(j) + " Mean-> mseTest: " + str(np.mean(consolidado)) + " Variance: " + str(np.var(consolidado)))
    #print("Test MSE:", mean_squared_error(dsurv_test, y_pred))

#___________________________________________________#
## ************ c. Integracion tardia ************ ##

### ... Stacking (SVR) ... ###
## Parte 1. Predicciones de cada perfil
parametros = { ## grilla
        "kernel": ["linear","poly","rbf","sigmoid"], 
        'C': [1, 10],'gamma': [0.001],"degree": [3,10], "coef0": [0,1]        
        }

lgg_names = ["lgg_cnvTCGA", "lgg_mrnaTCGA", "lgg_methTCGA", "lgg_proteinTCGA"]
## Predicciones para cada perfil con grilla de validacion
pred = {key: Stacking("SVR", tcga_train[key], tcga_train[key].iloc[:,0].values, tcga_test[key], 10, parametros, True) for key in lgg_names}
## concatenacion de resultados para train y test
stack = pd.concat([i["train_pred"] for i in pred.values()], axis=1)
df_test = pd.concat([i["test_pred"] for i in pred.values()], axis=1)

## Parte 2. concatenacion de predicciones parte 1 y creacion de un modelo de stacking.
model = SVR()
params = { ## nueva grilla para stacking
        "kernel": ["linear","poly","rbf","sigmoid"], 
        'C': [1, 5, 10],'gamma': [0.0001, 0.001, 0.01],"degree": [2, 4, 10], "coef0": [0, 0.5, 1]        
        }

modelGrid = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, verbose=1)
modelGrid.fit(X=stack,y=tcga_train["lgg_survivalTCGA"].iloc[:,0].values)

y_pred=modelGrid.predict(df_test)

print(modelGrid.best_params_)

mean_squared_error(tcga_test["lgg_survivalTCGA"].iloc[:,0].values, y_pred)

### ... Stacking (SVR) + NMF ... ###
# Revisar: https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html
## Parametros de NMF en grilla
NMF_params = {"n_components": [2, 4, 6, 8], "l1_ratio": [0.0001, 0.001, 0.01, 0.05]}

grid = list(ParameterGrid(NMF_params))
lgg_names = ["lgg_cnvTCGA", "lgg_mrnaTCGA", "lgg_methTCGA", "lgg_proteinTCGA"]

## Parametros de SVR en grilla
params = { ## grilla
        "kernel": ["linear","poly","rbf","sigmoid"], 
        'C': [1, 5, 10],'gamma': [0.0001, 0.001, 0.01],"degree": [2, 4, 10], "coef0": [0, 0.5, 1]        
        }

for param_i in grid:
    train_NMF = {key: NMF(n_components = param_i["n_components"], init="random", random_state=0, l1_ratio = param_i["l1_ratio"]).fit_transform(tcga_train[key]) for key in lgg_names}
    test_NMF = {key: NMF(n_components = param_i["n_components"], init="random", random_state=0, l1_ratio = param_i["l1_ratio"]).fit_transform(tcga_test[key]) for key in lgg_names}
    
    pred = {key: Stacking("SVR", train_NMF[key], tcga_train[key].iloc[:,0].values, test_NMF[key], 10, parametros, True) for key in lgg_names}
    
    stack = pd.concat([i["train_pred"] for i in pred.values()], axis=1)
    df_test = pd.concat([i["test_pred"] for i in pred.values()], axis=1)
    
    model = SVR()
    
    modelGrid = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1, verbose=1)
    modelGrid.fit(X=stack,y=tcga_train["lgg_survivalTCGA"].iloc[:,0].values)
    
    y_pred=modelGrid.predict(df_test)
    
    print(modelGrid.best_params_)
    print("")
    
    print("'l1_ratio: {}, 'n_components: {}".format(param_i["l1_ratio"], param_i["n_components"]))
    
    print("MSE: {}".format(mean_squared_error(tcga_test["lgg_survivalTCGA"].iloc[:,0].values, y_pred)))

### ... Stacking (SVR) + PCA ... ###
PCA_params = {"n_components": [2, 4, 6, 8], "svd_solver": ["full", "randomized", "arpack"]}

grid = list(ParameterGrid(PCA_params))
lgg_names = ["lgg_cnvTCGA", "lgg_mrnaTCGA", "lgg_methTCGA", "lgg_proteinTCGA"]

params = {
        "kernel": ["linear","poly","rbf","sigmoid"], 
        'C': [1, 5, 10],'gamma': [0.0001, 0.001, 0.01],"degree": [2, 4, 10], "coef0": [0, 0.5, 1]        
        }

for param_i in grid:
    train_PCA = {key: PCA(n_components = param_i["n_components"], svd_solver=param_i["svd_solver"]).fit_transform(tcga_train[key]) for key in lgg_names}
    test_PCA = {key: PCA(n_components = param_i["n_components"], svd_solver=param_i["svd_solver"]).fit_transform(tcga_test[key]) for key in lgg_names}
    
    pred = {key: Stacking("SVR", train_PCA[key], tcga_train[key].iloc[:,0].values, test_PCA[key], 10, parametros, True) for key in lgg_names}
    
    stack = pd.concat([i["train_pred"] for i in pred.values()], axis=1)
    df_test = pd.concat([i["test_pred"] for i in pred.values()], axis=1)
    
    model = SVR()
    
    modelGrid = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1, verbose=1)
    modelGrid.fit(X=stack,y=tcga_train["lgg_survivalTCGA"].iloc[:,0].values)
    
    y_pred=modelGrid.predict(df_test)
    
    print(modelGrid.best_params_)
    print("")
    
    print("'svd_solver: {}, 'n_components: {}".format(param_i["svd_solver"], param_i["n_components"]))
    
    print("MSE: {}".format(mean_squared_error(tcga_test["lgg_survivalTCGA"].iloc[:,0].values, y_pred)))



#### Codigo residual ####

### Ensamblaje de ensamblajes

# # %%

# # param_grid = {
# #         "n_components": [2,4,6,8,10], 
# #         "l1_ratio":[0,0.5,1]
# #         }
# params = {
#         "kernel": ["linear","poly","rbf","sigmoid"], 
#         'C': [1, 10],'gamma': [0.001],"degree": [3,10], "coef0": [0,1]        
#         }

# # %%
# model = NMF(n_components = 25, init='random', random_state=0, l1_ratio = 0.5)
# mrna_train_NMF = model.fit_transform(tcga_train["lgg_mrnaTCGA"])
# mrna_test_NMF = model.fit_transform(tcga_test["lgg_mrnaTCGA"])
# #H = model.components_
# #grid = GridSearchCV(model, cv=3, n_jobs=-1, param_grid=param_grid)
# #grid.fit(mrna_train, dsurv_train)
# # %%
# np.savetxt("NonMatrixFactorizationTCGA/mrna_train_NMF.txt", mrna_train_NMF)
# np.savetxt("NonMatrixFactorizationTCGA/mrna_test_NMF.txt", mrna_test_NMF)
# # %%
# test_pred1,train_pred1=Stacking(model="SVR",n_fold=10, train=pd.DataFrame(mrna_train_NMF),test=pd.DataFrame(mrna_test_NMF),y=tcga_train["lgg_survivalTCGA"].iloc[:,0].values, params=params, grilla=False)

# train_pred1=pd.DataFrame(train_pred1)
# test_pred1=pd.DataFrame(test_pred1)

# train_pred1.to_csv("NonMatrixFactorizationTCGA/train_pred1.csv")
# test_pred1.to_csv("NonMatrixFactorizationTCGA/test_pred1.csv")
# #{'C': 1, 'coef0': 0, 'degree': 3, 'gamma': 0.001, 'kernel': 'rbf'}

# # %%
# model = NMF(n_components = 25, init='random', random_state=0, l1_ratio = 0.5)
# cnv_train_NMF = model.fit_transform(tcga_train["lgg_cnvTCGA"])
# cnv_test_NMF = model.fit_transform(tcga_test["lgg_cnvTCGA"])

# # %%
# np.savetxt("NonMatrixFactorizationTCGA/cnv_train_NMF.txt", cnv_train_NMF)
# np.savetxt("NonMatrixFactorizationTCGA/cnv_test_NMF.txt", cnv_test_NMF)

# # %%
# test_pred2,train_pred2=Stacking(model="SVR",n_fold=10, train=pd.DataFrame(cnv_train_NMF),test=pd.DataFrame(cnv_test_NMF),y=tcga_train["lgg_survivalTCGA"].iloc[:,0].values, params=params, grilla=True)

# train_pred2=pd.DataFrame(train_pred2)
# test_pred2=pd.DataFrame(test_pred2)

# train_pred2.to_csv("NonMatrixFactorizationTCGA/train_pred2.csv")
# test_pred2.to_csv("NonMatrixFactorizationTCGA/test_pred2.csv")
# #{'C': 1, 'coef0': 1, 'degree': 3, 'gamma': 0.001, 'kernel': 'poly'}

# # %%
# model = NMF(n_components = 1000, init='random', random_state=0, l1_ratio = 0.5)
# meth_train_NMF = model.fit_transform(meth_train)
# meth_test_NMF = model.fit_transform(meth_test)

# # %%
# np.savetxt("NonMatrixFactorizationTCGA/meth_train_NMF.txt", meth_train_NMF)
# np.savetxt("NonMatrixFactorizationTCGA/meth_test_NMF.txt", meth_test_NMF)

# # %%

# meth_train_NMF = np.loadtxt("NonMatrixFactorizationTCGA/meth_train_NMF.txt")
# meth_test_NMF = np.loadtxt("NonMatrixFactorizationTCGA/meth_test_NMF.txt")

# params={}
# test_pred3,train_pred3=Stacking(model="SVR",n_fold=10, train=pd.DataFrame(meth_train_NMF),test=pd.DataFrame(meth_test_NMF),y=dsurv_train.ravel(), params=params, grilla=False)

# train_pred3=pd.DataFrame(train_pred3)
# test_pred3=pd.DataFrame(test_pred3)

# train_pred3.to_csv("NonMatrixFactorizationTCGA/train_pred3.csv")
# test_pred3.to_csv("NonMatrixFactorizationTCGA/test_pred3.csv")
# #{'C': 1, 'coef0': 1, 'degree': 3, 'gamma': 0.001, 'kernel': 'poly'}

# # %%
# model = NMF(n_components = 75, init='random', random_state=0, l1_ratio = 0.5)
# prot_train_NMF = model.fit_transform(protein_train)
# prot_test_NMF = model.fit_transform(protein_test)

# # %%
# np.savetxt("NonMatrixFactorizationTCGA/prot_train_NMF.txt", prot_train_NMF)
# np.savetxt("NonMatrixFactorizationTCGA/prot_test_NMF.txt", prot_test_NMF)

# # %%
# test_pred4,train_pred4=Stacking(model="SVR",n_fold=10, train=pd.DataFrame(prot_train_NMF),test=pd.DataFrame(prot_test_NMF),y=dsurv_train.ravel(), params=params, grilla=True)

# train_pred4=pd.DataFrame(train_pred4)
# test_pred4=pd.DataFrame(test_pred4)

# train_pred4.to_csv("NonMatrixFactorizationTCGA/train_pred4.csv")
# test_pred4.to_csv("NonMatrixFactorizationTCGA/test_pred4.csv")

# # %%
# from sklearn.model_selection import GridSearchCV
# train_pred1 = pd.read_csv("NonMatrixFactorizationTCGA/train_pred1.csv", header=0)
# test_pred1 = pd.read_csv("NonMatrixFactorizationTCGA/test_pred1.csv", header=0)
# train_pred2 = pd.read_csv("NonMatrixFactorizationTCGA/train_pred2.csv", header=0)
# test_pred2 = pd.read_csv("NonMatrixFactorizationTCGA/test_pred2.csv", header=0)
# train_pred3 = pd.read_csv("NonMatrixFactorizationTCGA/train_pred3.csv", header=0)
# test_pred3 = pd.read_csv("NonMatrixFactorizationTCGA/test_pred3.csv", header=0)
# train_pred4 = pd.read_csv("NonMatrixFactorizationTCGA/train_pred4.csv", header=0)
# test_pred4 = pd.read_csv("NonMatrixFactorizationTCGA/test_pred4.csv", header=0)

# stack = pd.concat([train_pred1, train_pred2, train_pred3, train_pred4], axis=1)
# df_test = pd.concat([test_pred1, test_pred2, test_pred3, test_pred4], axis=1)

# model = SVR()

# params = {
#         "kernel": ["linear","poly","rbf","sigmoid"], 
#         'C': [1, 10],'gamma': [0.001],"degree": [3,10], "coef0": [0,1]        
#         }

# modelGrid = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, verbose=1)
# modelGrid.fit(X=stack,y=dsurv_train.ravel())
# print(modelGrid.best_params_)

# y_pred=modelGrid.predict(df_test)

# mean_squared_error(dsurv_test, y_pred)

# %% SVR con NMF



# # %%

# #model1 = SVR()


# # test_pred1, train_pred1 = Stacking(model="SVR", 
# #                                    n_fold=10, 
# #                                    train=tcga_train["lgg_mrnaTCGA"], 
# #                                    test=tcga_test["lgg_mrnaTCGA"], 
# #                                    y=tcga_train["lgg_survivalTCGA"].iloc[:,0], params=parametros, True)

# test_pred1, train_pred1 = Stacking("SVR", 
#                                    tcga_train["lgg_mrnaTCGA"], 
#                                    tcga_train["lgg_survivalTCGA"].iloc[:,0].values,
#                                    tcga_test["lgg_mrnaTCGA"], 
#                                    10, parametros, True)

# train_pred1=pd.DataFrame(train_pred1)
# test_pred1=pd.DataFrame(test_pred1)

# train_pred1.to_csv("StackingTCGA/train_pred1.csv")
# test_pred1.to_csv("StackingTCGA/test_pred1.csv")
# #best_params_={'C': 1, 'coef0': 1, 'degree': 3, 'gamma': 0.001, 'kernel': 'poly'}

# # %%

# #model2 = SVR()

# test_pred2, train_pred2 = Stacking("SVR", 
#                                    tcga_train["lgg_cnvTCGA"], 
#                                    tcga_train["lgg_survivalTCGA"].iloc[:,0].values,
#                                    tcga_test["lgg_cnvTCGA"], 
#                                    10, parametros, True)

# train_pred2=pd.DataFrame(train_pred2)
# test_pred2=pd.DataFrame(test_pred2)

# train_pred2.to_csv("StackingTCGA/train_pred2.csv")
# test_pred2.to_csv("StackingTCGA/test_pred2.csv")
# #best_params_={'C': 1, 'coef0': 0, 'degree': 3, 'gamma': 0.001, 'kernel': 'rbf'}

# # %%

# #model3 = SVR()

# test_pred3, train_pred3 = Stacking("SVR", 
#                                    tcga_train["lgg_methTCGA"], 
#                                    tcga_train["lgg_survivalTCGA"].iloc[:,0].values,
#                                    tcga_test["lgg_methTCGA"], 
#                                    10, parametros, True)

# train_pred3=pd.DataFrame(train_pred3)
# test_pred3=pd.DataFrame(test_pred3)

# train_pred3.to_csv("StackingTCGA/train_pred3.csv")
# test_pred3.to_csv("StackingTCGA/test_pred3.csv")

# # %%

# #model4 = SVR()

# test_pred4, train_pred4 = Stacking("SVR", 
#                                    tcga_train["lgg_proteinTCGA"], 
#                                    tcga_train["lgg_survivalTCGA"].iloc[:,0].values,
#                                    tcga_test["lgg_proteinTCGA"], 
#                                    10, parametros, True)

# train_pred4=pd.DataFrame(train_pred4)
# test_pred4=pd.DataFrame(test_pred4)

# train_pred4.to_csv("StackingTCGA/train_pred4.csv")
# test_pred4.to_csv("StackingTCGA/test_pred4.csv")
# #best_params_={'C': 1, 'coef0': 0, 'degree': 3, 'gamma': 0.001, 'kernel': 'rbf'}

# # %%
# from sklearn.model_selection import GridSearchCV
# stack = pd.concat([train_pred1, train_pred2, train_pred3, train_pred4], axis=1)
# df_test = pd.concat([test_pred1, test_pred2, test_pred3, test_pred4], axis=1)

# model = SVR()

# params = {
#         "kernel": ["linear","poly","rbf","sigmoid"], 
#         'C': [1, 10],'gamma': [0.001],"degree": [3,10], "coef0": [0,1]        
#         }

# modelGrid = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, verbose=1)
# modelGrid.fit(X=stack,y=tcga_train["lgg_survivalTCGA"].iloc[:,0].values)
# print(modelGrid.best_params_)

# y_pred=modelGrid.predict(df_test)

# mean_squared_error(tcga_test["lgg_survivalTCGA"].iloc[:,0].values, y_pred)





# # %%
# # Correr en python 2.7
# # Multiple Kernel Learning
# from mklTCGA import ICDoneKernelOneTCGA, ridgeLowRankOneKernel, variousKernelVariousMethodsOneTCGA, mklTCGA

# # %%

# '''
# Kernels:
#     linear_kernel
#     sigmoid_kernel
#     poly_kernel:
#         {"degree": 3}
#     rbf_kernel:
#         {"sigma": 30}
# methods:
#     "nystrom", "icd"
# '''
# from mklaren.kernel.kernel import linear_kernel, poly_kernel, sigmoid_kernel, rbf_kernel, exponential_absolute, periodic_kernel, matern_kernel
# model = ICDoneKernelOneTCGA(mrna, kernel=linear_kernel, kernel_args={}, rank=15)
# model = ICDoneKernelOneTCGA(mrna, kernel=poly_kernel, kernel_args={"degree": 3}, rank=15)
# model = ICDoneKernelOneTCGA(mrna, kernel=sigmoid_kernel, kernel_args={}, rank=15)
# model = ICDoneKernelOneTCGA(mrna, kernel=rbf_kernel, kernel_args={"sigma": 30}, rank=15)
# # %%
# kernels = {"linear_kernel":linear_kernel, 
#            "poly_kernel":poly_kernel,
#            "sigmoid_kernel":sigmoid_kernel, 
#            "rbf_kernel":rbf_kernel, 
#            "exponential_absolute":exponential_absolute, 
#            "periodic_kernel":periodic_kernel, 
#            "matern_kernel":matern_kernel
#            }

# def frange(x, y, jump):
#   while x < y:
#     yield x
#     x += jump

# for i in xrange(1,10,1):#frange(0, 1, .1): #0.5, 1.5,2.5:
#     consolidado = []
#     for ix_train, ix_test in kfoldProfiles(mrna_train, n_splits=5, seed=10):
        
#         mrna_tr=mrna_train[ix_train]
#         cnv_tr=cnv_train[ix_train]
#         meth_tr=meth_train[ix_train]
#         protein_tr=protein_train[ix_train]
#         dsurv_tr=dsurv_train[ix_train]
        
# #        if key == "linear_kernel":
# #            kernel_args={'b': i}
# #        elif key == "poly_kernel":
# #            kernel_args={"degree": i}
# #        elif key == "sigmoid_kernel":
# #            kernel_args={'c': i}
# #        elif key == "rbf_kernel":
# #            kernel_args={"sigma": i}
# #        elif key == "exponential_absolute":
# #            kernel_args={"sigma": i}
# #        elif key == "periodic_kernel":
# #            kernel_args={"sigma": i}
# #        else:
# #            kernel_args={"nu": i}
# #        
#         try: 
#             model = ridgeLowRankOneKernel(mrna_train, dsurv_train, kernel=poly_kernel, kernel_args={"degree": i}, rank=15, method="icd")
#             yp = model.predict([np.array(mrna_test)])
#             consolidado.append(mean_squared_error(dsurv_test, yp))
#         except:
#             pass
        
#     print(" i: " + str(i) + " Mean-> iternal train: " + str(np.mean(consolidado)) + " Variance: " + str(np.var(consolidado)))
        




# #model = ridgeLowRankOneKernel(mrna_train, mrna_test, dsurv_train, dsurv_test, rbf_kernel, kernel_args={"sigma": 110}, rank=15)

# #model = variousKernelVariousMethodsOneTCGA(mrna_train, mrna_test, dsurv_train, dsurv_test, method="nystrom", rank=15)

# # %% mkl

# def frange(x, y, jump):
#   while x < y:
#     yield x
#     x += jump
    
# #for i in xrange(15, 25, 1): 
# #for i in 0.5, 1.5,2.5:
# #for i in frange(.9, 1.015, .001): 
# '''
# default parameters:
#     rank = 15 ## 
#     sigmaKernel = 2.0 ## exponential_kernel == rfb_kernel
#     degreeKernel = 2 ## poly_kernel
#     biasKernel = 0 ## linear_kernel
#     cKernel = 1 ## sigmoid_kernel
#     sigmaABSKernel = 2.0 ## exponential_absolute
#     sigmaPerKernel = 1 ## periodic_kernel
#     nuKernel = 1.5 ## matern_kernel
#     L2Ridge = 0 ## lbd
#     LookAhead = 10 ## delta
# '''
# consolidado = []
# for ix_train, ix_test in kfoldProfiles(mrna_train, n_splits=5, seed=10):
    
#     mrna_tr=mrna_train[ix_train]
#     #mrna_te=mrna_test[ix_test]
#     cnv_tr=cnv_train[ix_train]
#     #cnv_te=cnv_test[ix_test]
#     meth_tr=meth_train[ix_train]
#     #meth_te=meth_test[ix_test]
#     protein_tr=protein_train[ix_train]
#     #protein_te=protein_test[ix_test]
#     dsurv_tr=dsurv_train[ix_train]
#     #dsurv_te=dsurv_test[ix_test]
    
#     try: 
#         '''
#         Parametros para PCA
#         rank=17, sigmaKernel=120, degreeKernel=3,
#                         biasKernel=0, cKernel=0, sigmaABSKernel=2, sigmaPerKernel=2,
#                         nuKernel=1.5, L2Ridge=.991, LookAhead=40
#         '''
#         model = mklTCGA(mrna_tr, cnv_tr, meth_tr, protein_tr, dsurv_tr, rank=18, sigmaKernel=2, degreeKernel=2,
#                         biasKernel=8, cKernel=1, sigmaABSKernel=2, sigmaPerKernel=1,
#                         nuKernel=1.5, L2Ridge=.923, LookAhead=4)
#         yp = model.predict([mrna_test, mrna_test, mrna_test, mrna_test, mrna_test, mrna_test, mrna_test,
#                             cnv_test, cnv_test, cnv_test, cnv_test, cnv_test, cnv_test, cnv_test,
#                             meth_test, meth_test, meth_test, meth_test, meth_test, meth_test, meth_test, 
#                             protein_test, protein_test, protein_test, protein_test, protein_test, protein_test, protein_test])
#         consolidado.append(mean_squared_error(dsurv_test, yp))
#     except:
#         pass
    
# #print(" Mean-> iternal train: " + str(np.mean(consolidado)) + " Variance: " + str(np.var(consolidado)))
# print("i: " + str(i) + " Mean-> iternal train: " + str(np.mean(consolidado)) + " Variance: " + str(np.var(consolidado)))
#         #print("TestVal MSE:", mean_squared_error(dsurv_test, yp))


# %%
#svmLinearParam = {
#        "kernel":"linear", 
#        "C":1, 
#        "epsilon": 0.1
#        }
#svmPolyParam = {
#        "kernel":"poly", 
#        "C":1, 
#        "degree": 3, 
#        "gamma": "auto",
#        "coef0":0
#        }
#svmRbfParam = {
#        "kernel":"rbf", 
#        "C":1, 
#        "epsilon": 0.1
#        }
#svmSigmoidParam = {
#        "kernel":"sigmoid", 
#        "gamma": "auto",
#        "coef0": 0
#        }
## %%
#from mlxtend.regressor import StackingCVRegressor
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import GridSearchCV
#
#svrLinear = SVR(kernel="linear")
#svrPoly = SVR(kernel="poly")
#svrRbf = SVR(kernel="rbf")
#svrSigmoid = SVR(kernel="sigmoid")
#rf = RandomForestRegressor(random_state=10)
#sclf = StackingCVRegressor(regressors=[svrLinear, svrPoly, svrSigmoid], 
#                          meta_regressor=rf)
#
#params = {'SVR__C': [1, 10],'SVR__gamma': [0.001, 0.0001],"SVR__degree": [3,10,15],
#          "SVR__epsilon": [0.01, 0.1], "SVR__coef0": [0,1],
#          'meta-randomforestregressor__n_estimators': [10, 100]}
#
#grid = GridSearchCV(estimator=sclf, 
#                    param_grid=params, 
#                    cv=3,
#                    refit=True)
#
##svmLinear = SklearnHelper(model=SVR,params=svmLinearParam)
##svmPoly = SklearnHelper(model=SVR, params=svmPolyParam)
##svmRbf = SklearnHelper(model=SVR, params=svmRbfParam)
##svmSigmoid = SklearnHelper(model=SVR, params=svmSigmoidParam)
#
## %%
#grid.fit(mrna_train, dsurv_train)