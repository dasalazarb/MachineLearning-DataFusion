# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:02:26 2019

@author: da.salazarb
"""

# ******************************************************************************************************************************************

# ## ANN
# [Retornar al inicio](#Data)

# Se tomo de __[Neural Network Model for House Prices (TensorFlow)](https://www.kaggle.com/zoupet/neural-network-model-for-house-prices-tensorflow)__.

# 1. Particion de datos usando sklearn.
# 2. Conversion de variables continuas en formato especial.
# 3. Definicion de la red

# ## Keras

# Para usar GridSearch con Keras ver este tutorial: __[Need help with Deep Learning? Take the FREE Mini-Course
# How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)__. Se buscan diferentes parametros. Sin embargo, se requiere mas poder computacional y reguralizacion.

# In[32]:

import keras
#import numpy as np
from keras.layers import Input, Dense, Dropout#, Activation
from keras.models import Model
from DenseMoE import DenseMoE
#from keras.models import Sequential
#from keras import regularizers
#from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline

# ## The Merge layers

# Regresar a: 
# 1. [Funciones](#Funciones)
# 2. [PreprossData](#PreprossData)

# ### (mrna + cnv + meth)

# In[14]:


def multiModelGenLayer(mrna_train, cnv_train, meth_train):
    
    def modelKeras(input_shape, neurons):
        main_input = Input(shape=(input_shape)) #inputshape=x_train.shape[1:]
        nn = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(main_input)
        nn = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(nn)
        nn = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(nn)
        nn = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(nn)
        nn = Dense(neurons, activation="relu")(nn)
        return nn, main_input
    
    mrna_nn, mrna_input = modelKeras(mrna_train.shape[1:], 100)
    cnv_nn, cnv_input = modelKeras(cnv_train.shape[1:], 100)
    meth_nn, meth_input = modelKeras(meth_train.shape[1:], 100)
    #protein_nn, protein_input = modelKeras(protein_train.shape[1:], 100)
    mrna_cnv_meth = keras.layers.concatenate([mrna_nn, cnv_nn, meth_nn])
    x = Dense(100, activation='relu')(mrna_cnv_meth)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    main_output = Dense(1, activation='relu', name='main_output')(x)
    model = Model(inputs=[mrna_input, cnv_input, meth_input], outputs=main_output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    return model

# In[458]:


def multiModelGenProtLayer(x_train, cnv_train, meth_train, protein_train):
    
    def modelKeras(input_shape, neurons):
        main_input = Input(shape=(input_shape)) #inputshape=x_train.shape[1:]
        nn = DenseMoE(neurons, 3, expert_activation="tanh", gating_activation='softmax')(main_input)
        nn = DenseMoE(neurons, 3, expert_activation="tanh", gating_activation='softmax')(nn)
        nn = DenseMoE(neurons, 3, expert_activation="tanh", gating_activation='softmax')(nn)
        nn = DenseMoE(neurons, 3, expert_activation="tanh", gating_activation='softmax')(nn)
        nn = Dense(neurons, activation="relu")(nn)
        return nn, main_input
    
    mrna_nn, mrna_input = modelKeras(x_train.shape[1:], 75)
    cnv_nn, cnv_input = modelKeras(cnv_train.shape[1:], 75)
    meth_nn, meth_input = modelKeras(meth_train.shape[1:], 75)
    protein_nn, protein_input = modelKeras(protein_train.shape[1:], 75)
    mrna_cnv_meth = keras.layers.concatenate([mrna_nn, cnv_nn, meth_nn])
    x = Dense(100, activation='relu')(mrna_cnv_meth)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    all_profiles = keras.layers.concatenate([x,protein_nn])
    x = Dense(100, activation='relu')(all_profiles)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    main_output = Dense(1, activation='relu', name='main_output')(x)
    model = Model(inputs=[mrna_input, cnv_input, meth_input, protein_input], outputs=main_output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    return model

# In[12]:

def allToDNN(mrna, cnv, meth, prot, neurons, dropOut, activation):
    
    def modelKeras(input_shape, neurons, dropOut, activation):
        main_input = Input(shape=(input_shape)) #inputshape=x_train.shape[1:]
        nn = DenseMoE(neurons, n_experts=10, expert_activation=activation, gating_activation=activation)(main_input)
        nn = Dropout(dropOut)(nn)
        nn = DenseMoE(neurons, n_experts=10, expert_activation=activation, gating_activation=activation)(nn)
        nn = Dropout(dropOut)(nn)
        nn = DenseMoE(neurons, n_experts=10, expert_activation=activation, gating_activation=activation)(nn)
        nn = Dropout(dropOut)(nn)
        nn = DenseMoE(neurons, n_experts=10, expert_activation=activation, gating_activation=activation)(nn)
        nn = Dropout(dropOut)(nn)
        nn = DenseMoE(neurons, n_experts=10, expert_activation=activation, gating_activation=activation)(nn)
        nn = Dropout(dropOut)(nn)
        nn = DenseMoE(neurons, n_experts=10, expert_activation=activation, gating_activation=activation)(nn)
        nn = Dropout(dropOut)(nn)
        nn = DenseMoE(neurons, n_experts=10, expert_activation=activation, gating_activation=activation)(nn)
        nn = Dropout(dropOut)(nn)
        nn = DenseMoE(neurons, n_experts=10, expert_activation=activation, gating_activation=activation)(nn)
        nn = Dropout(dropOut)(nn)
        nn = DenseMoE(neurons, n_experts=10, expert_activation=activation, gating_activation=activation)(nn)
        nn = Dropout(dropOut)(nn)
        nn = DenseMoE(neurons, n_experts=10, expert_activation=activation, gating_activation=activation)(nn)
        nn = Dropout(dropOut)(nn)
        nn = DenseMoE(neurons, n_experts=10, expert_activation=activation, gating_activation=activation)(nn)
        nn = Dropout(dropOut)(nn)
        nn = DenseMoE(neurons, n_experts=10, expert_activation=activation, gating_activation=activation)(nn)
        nn = Dropout(dropOut)(nn)
        return nn, main_input
    
    mrnaDNN, mrnaInput = modelKeras(mrna.shape[1:], neurons, dropOut, activation)
    cnvDNN, cnvInput = modelKeras(cnv.shape[1:], neurons, dropOut, activation)
    methDNN, methInput = modelKeras(meth.shape[1:], neurons, dropOut, activation)
    proteinDNN, proteinInput = modelKeras(prot.shape[1:], neurons, dropOut, activation)
    
    all_profiles = keras.layers.concatenate([mrnaDNN, cnvDNN, methDNN, proteinDNN])
    
    x = Dense(neurons, activation=activation)(all_profiles)
    x = Dropout(dropOut)(x)
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropOut)(x)
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropOut)(x)
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropOut)(x)
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropOut)(x)
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropOut)(x)
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropOut)(x)
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropOut)(x)
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropOut)(x)
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropOut)(x)
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropOut)(x)
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropOut)(x)
    main_output = Dense(1, activation=activation, name='main_output')(x)
    
    model = Model(inputs=[mrnaInput, cnvInput, methInput, proteinInput], outputs=main_output)
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    return model


# In[22]:


#model = allToDNN(mrna, cnv, meth, protein, neurons=100, dropOut=.5, activation="relu")
#model.fit([mrna, cnv, meth, protein], dsurv, epochs=100, verbose=1, validation_split=0.2)


## In[15]:
#
#
#from sklearn.model_selection import LeaveOneOut
#
#
## In[16]:
#
#
## define 10-fold cross validation test harness
#loco = LeaveOneOut()
#cvscores = []
#
#
## In[17]:
#
#
#def modelKeras(input_shape, neurons):
#    main_input = Input(shape=(input_shape)) #inputshape=x_train.shape[1:]
#    nn = Dense(neurons, activation="relu")(main_input)
#    nn = Dense(neurons, activation="relu")(nn)
#    nn = Dense(neurons, activation="relu")(nn)
#    nn = Dense(neurons, activation="relu")(nn)
#    return nn, main_input
#
#def allToDNN(mrna, cnv, meth, prot, neurons):
#    mrnaDNN = modelKeras(mrna.shape[1:], neurons)
#    cnvDNN = modelKeras(cnv.shape)
#
#
## In[25]:
#
#
##print(np.mean(cvscores))
##print(np.var(cvscores))
#
#
## In[367]:
#
#
#model.fit([x_train, cnv_train, meth_train], y_train,
#          epochs=25, batch_size=32)
#
#
## In[366]:
#
#
## 50 ephocs
#model.evaluate([x_test, cnv_test, meth_test], y_test, batch_size=128)
#
#
## In[368]:
#
#
## early stop -> 25 stops
#model.evaluate([x_test, cnv_test, meth_test], y_test, batch_size=128)
#
#
## ### Protein + (mrna + cnv + meth)
#
## In[18]:
#
#
#import keras
#from keras.layers import Input, Dropout, Dense
#from sklearn.model_selection import LeaveOneOut
#from keras.models import Model
#
#
## In[451]:
#
#

#
#
## In[460]:
#
#
## define 10-fold cross validation test harness
#loco = LeaveOneOut()
#cvscores = []
#
#
## In[461]:
#
#
#for train, test in loco.split(mrna):
#    # Crear el modelo
#    model = multiModelKerasComplet(mrna, cnv, meth, protein)
#    # Fit the model
#    model.fit([mrna.iloc[train,:], cnv.iloc[train,:], meth.iloc[train,:], protein.iloc[train,:]], dsurv.iloc[:,1:].iloc[train,:], epochs=150, batch_size=10, verbose=0)
#    # Evaluar el modelo
#    scores = model.evaluate([mrna.iloc[test,:], cnv.iloc[test,:], meth.iloc[test,:], protein.iloc[test,:]], dsurv.iloc[:,1:].iloc[test,:])
#    print("MSE: %.2f%" % (scores[0]))
#    cvscores.append(scores[0])
#
#
## In[377]:
#
#
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])
#
#
## In[381]:
#
#
#model.fit([x_train, cnv_train, meth_train, protein_train], y_train,
#          epochs=50, batch_size=32)
#
#
## In[382]:
#
#
## 50 ephocs
#model.evaluate([x_test, cnv_test, meth_test, protein_test], y_test, batch_size=128)
#
#
## In[380]:
#
#
## 25 ephocs
#model.evaluate([x_test, cnv_test, meth_test, protein_test], y_test, batch_size=128)
#
#
## In[59]:
#
#
#d=[1,2,3,34,4,4,5,566,53,2,2]
#d.append(12)
#np.mean(d)
#
#
## ## mrna + cnv + meth + prot
#
## In[10]:
#
#
#import keras
#from keras.layers import Input, Dropout, Dense
#from sklearn.model_selection import LeaveOneOut
#from keras.models import Model
#
#
## In[11]:
#
#
#mrna, cnv, meth, dsurv, protein = reRunFiles(varX=True, logSurv=True)
#mrna, cnv, meth, dsurv, protein, x_train, x_test, y_train, y_test, cnv_train, cnv_test, meth_train, meth_test, protein_train, protein_test = demoTrainDevSet(mrna, cnv, meth, dsurv, protein)
#
#

#
#
## ## Evaluacion de parametros
#
## In[17]:
#
#
## Parametros a evaluar
#parametros = {
#    "neurons":[10,100, 500, 1000, 5000],
#    "dropOut":[0,.2,.4,.6,.8], 
#    "activation":["selu", "relu" ,"tanh", "sigmoid", "hard_sigmoid"]
#}
#
#
## In[ ]:
#
#
#keysParams=list(parametros.keys())
#consolidado = {}
#for neuron in parametros[keysParams[0]]:
#    for drop in parametros[keysParams[1]]:
#        for activ in parametros[keysParams[2]]:
#            saveEval=[]
#            for ix_train, ix_test in kfoldProfiles(mrna, n_splits=3, seed=10):
#                mrna_train=mrna.iloc[ix_train,]
#                mrna_test=mrna.iloc[ix_test,]
#                cnv_train=cnv.iloc[ix_train,]
#                cnv_test=cnv.iloc[ix_test,]
#                meth_train=meth.iloc[ix_train,]
#                meth_test=meth.iloc[ix_test,]
#                protein_train=protein.iloc[ix_train,]
#                protein_test=protein.iloc[ix_test,]
#                dsurv_train=dsurv.iloc[ix_train,]
#                dsurv_test=dsurv.iloc[ix_test,]
#
#                model = allToDNN(mrna_train, cnv_train, meth_train, protein_train, neurons=neuron, dropOut=drop, activation=activ)
#                model.fit([mrna_train, cnv_train, meth_train, protein_train], dsurv_train, epochs=100, verbose=0)
#                saveEval.append(model.evaluate([mrna_test, cnv_test, meth_test, protein_test], dsurv_test, batch_size=20)[1])
#            consolidado["neuron: " + str(neuron) + " - " + "dropOut: " + str(drop) + " - " + "activation: " + str(activ)] = [np.mean(saveEval), np.var(saveEval)]
#
#
## In[ ]:
#
#
#parametros = {
#    "activation": ["softmax", "elu", "selu", "softplus", "softsign", "relu" ,"tanh", "sigmoid", "hard_sigmoid", "linear"],
#    "optimizer": ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"], 
#    "drop": [0.2,0.4,0.6,0.8,1],
#    "kernel": ["random_normal", "he_uniform", "lecun_normal", "he_normal", "glorot_uniform", "glorot_normal", "lecun_uniform", "zeros", "ones"], 
#    "neurons": [5,10,50,100]
#    #"moreHiddenLayers": [True, False]
#}
#
#
## In[16]:
#
#
#consolidado
#
#
## ******************************************************************************************************************************************
#
## ### Usando un cluster - "Dask"
#
## No para keras?
#
## In[269]:
#
#
#from dask.distributed import Client
#client = Client(processes=False)
#client
#
#
## In[270]:
#
#
#import dask_ml.joblib
#from sklearn.externals import joblib
#
#
