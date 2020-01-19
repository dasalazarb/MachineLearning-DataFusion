# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:35:36 2018

@author: da.salazarb
"""
import os
os.chdir("C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA/moeTCGA")
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Reshape, merge, Lambda, concatenate, BatchNormalization
from keras.layers.merge import multiply, dot
#from keras.engine import Merge
from keras.models import Model
from keras import backend as K
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from DenseMoE import DenseMoE

#from DenseMoE import DenseMoE

#MoE原理用原文中的一幅图就可以清晰的缕清，实现的Keras代码如下：
# %%
def sliced(x,expert_num):
    return x[:,:,:expert_num]
def reduce(x, axis):
    return K.sum(x, axis=axis, keepdims=True)


def gatExpertLayer(inputGate, inputExpert, expert_num, nb_class):
    
    # The Inputs
    input_vector1 = Input(shape=(inputGate.shape[1:]))
    input_vector2 = Input(shape=(inputExpert.shape[1:]))
    
    # The Gate
    gate = Dense(expert_num*nb_class, activation='softmax')(input_vector1)
    gate = Reshape((nb_class, expert_num))(gate)
    gate = Lambda(lambda x: tf.split(x,num_or_size_splits=1,axis=1))(gate)
    
    # The Expert
    expert = DenseMoE(nb_class*expert_num, 3, expert_activation="softmax", gating_activation='softmax')(input_vector2)
    #expert = Dense(nb_class*expert_num, activation='tanh')(input_vector2)
    expert = Reshape((nb_class, expert_num))(expert)
    
    # The Output
    output = multiply([gate, expert])
    output = Lambda(reduce, output_shape=(nb_class,), arguments={'axis': 2})(output)
    
    #The model
    model = Model(inputs=[input_vector1, input_vector2], outputs=output)
    model.compile(loss='mean_squared_error', metrics=['mse'], optimizer='adam')
    
    return model
# %%
model=gatExpertLayer(cnv_train, mrna_train, 100, 1)
model.summary()
#plot_model(model, to_file='model.png')
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
model.fit(x=[cnv_train, mrna_train], y=dsurv_train, batch_size=32, verbose=0, epochs=1000)
model.predict([cnv_test, mrna_test])
score = model.evaluate([cnv_test, mrna_test], dsurv_test, batch_size=32, verbose=1)
print(score[1])

# %%

def gatExpertLayer(inputGate, inputExpert, expert_num, nb_class):

    # The Inputs
    input_vector1 = Input(shape=(inputGate.shape[1:]))
    input_vector2 = Input(shape=(inputExpert.shape[1:]))
    
    # The Gate
    gate = Dense(expert_num*nb_class, activation='softmax')(input_vector1)
    gate = Reshape((nb_class, expert_num))(gate)
    gate = Lambda(lambda x: tf.split(x,num_or_size_splits=1,axis=1))(gate)
    
    # The Expert
    neurons=nb_class*expert_num
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(input_vector2)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    expert = DenseMoE(neurons, 3, expert_activation="relu", gating_activation='softmax')(expert)
    #expert = DenseMoE(nb_class*expert_num, 3, expert_activation="softmax", gating_activation='softmax')(input_vector2)
    #expert = Dense(nb_class*expert_num, activation='tanh')(input_vector2)
    expert = Reshape((nb_class, expert_num))(expert)
    
    # The Output
    output = multiply([gate, expert])
    output = Lambda(reduce, output_shape=(nb_class,), arguments={'axis': 2})(output)
    
    #The model
    model = Model(inputs=[input_vector1, input_vector2], outputs=output)
    model.compile(loss='mean_squared_error', metrics=['mse'], optimizer='adam')
    
    return model
# %%
model=gatExpertLayer(cnv_train, mrna_train, 1000, 1)
#model.summary()
model.fit(x=[cnv_train, mrna_train], y=dsurv_train, batch_size=32, verbose=0, epochs=1000)
model.evaluate([cnv_test, mrna_test], dsurv_test, batch_size=32, verbose=1)
#model.predict([cnv_test, mrna_test])

# %%

def moeGenome(inputGateCnv, inputGatMeth, inputExpert, expert_num, nb_class, neurons):
    
    # The Inputs
    input_vector1 = Input(shape=(inputGateCnv.shape[1:]))
    input_vector2 = Input(shape=(inputExpert.shape[1:]))
    input_vector3 = Input(shape=(inputGatMeth.shape[1:]))
    
    # The GateCnv
    gatecnv = Dense(expert_num*nb_class, activation='softmax', name='densaGatecnv')(input_vector1)
    gatecnv = Reshape((nb_class, expert_num))(gatecnv)
    gatecnv = Lambda(lambda x: tf.split(x,num_or_size_splits=1,axis=1))(gatecnv)
    
    # The GateMeth
    gatemeth = Dense(expert_num*nb_class, activation='softmax')(input_vector3)
    gatemeth = Reshape((nb_class, expert_num))(gatemeth)
    gatemeth = Lambda(lambda x: tf.split(x,num_or_size_splits=1,axis=1))(gatemeth)
    
    # The Expert
    expert = DenseMoE(nb_class*expert_num, 3, expert_activation="softmax", gating_activation='softmax')(input_vector2)
    #expert = Dense(nb_class*expert_num, activation='tanh')(input_vector2)
    expert = Reshape((nb_class, expert_num))(expert)
    
    # The OutputCnv
    outputcnv = multiply([gatecnv, expert])
    outputcnv = Lambda(reduce, output_shape=(nb_class,), arguments={'axis': 2})(outputcnv)
    
    # The OutputMeth
    outputmeth = multiply([gatemeth, expert])
    outputmeth = Lambda(reduce, output_shape=(nb_class,), arguments={'axis': 2})(outputmeth)
    
    # The FinalOutput
    all_profiles = keras.layers.concatenate([outputcnv, outputmeth])
    
    x = Dense(neurons, activation='softmax')(all_profiles)
    #x = Dropout(.5)(x)
    x = Dense(neurons, activation='softmax')(x)
    #x = Dropout(.5)(x)
    output = Dense(1, activation='softmax', name='main_output')(x)
    
    #The model
    model = Model(inputs=[input_vector1, input_vector2], outputs=output)
    model.compile(loss='mean_squared_error', metrics=['mse'], optimizer='adam')
    
    return model
# %%
model=moeGenome(cnv_train, meth_train, mrna_train, 100, 1, 100)
model.summary()
#plot_model(model, to_file='model.png')
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
model.fit(x=[cnv_train, mrna_train], y=dsurv_train, batch_size=32, verbose=0, epochs=1000)
model.predict([cnv_test, mrna_test])
score = model.evaluate([cnv_test, mrna_test], dsurv_test, batch_size=32, verbose=1)
print(score[1])
