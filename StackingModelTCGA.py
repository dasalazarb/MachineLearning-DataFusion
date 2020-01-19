# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:40:02 2019

@author: da.salazarb
"""
# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# %%
def Stacking(model,train,y,test,n_fold, params, grilla):
    '''
    train: mrna_train, cnv_train o meth_train
    test: mrna_test, cnv_test o meth_test
    y: dsurv
    '''
    folds=KFold(n_splits=n_fold,random_state=1)
    test_pred=np.empty((test.shape[0],1),float)
    train_pred=np.empty((0,1),float)
    if model == "SVR" and grilla == True:
        modelGrid = GridSearchCV(estimator=SVR(), param_grid=params, cv=3, n_jobs=-1, verbose=0)
        modelGrid.fit(X=train,y=y)
        #print(modelGrid.best_params_)
    train = pd.DataFrame(train)
    y = pd.DataFrame(y)

    for train_indices,val_indices in folds.split(train,y.values.ravel()):
        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
        y_train, _=y.iloc[train_indices],y.iloc[val_indices]
        if model == "SVR":
            if grilla == True:
                model=SVR(**modelGrid.best_params_)
            else:
                model=SVR(**{'C': 1, 'coef0': 0, 'degree': 3, 'gamma': 0.001, 'kernel': 'rbf'})
            model.fit(X=x_train,y=y_train.values.ravel())
        #modelGrid.fit(X=x_train,y=y_train.values.ravel())
        train_pred=np.append(train_pred,model.predict(x_val))
        test_pred=np.concatenate((test_pred,model.predict(test).reshape(test.shape[0],1)), axis=1)
    test_pred=test_pred[:,1:]
    test_pred=np.mean(test_pred,axis=1)
    
    pred = {'test_pred': pd.DataFrame(test_pred.reshape(-1,1)), 'train_pred': pd.DataFrame(train_pred)}
    
    return pred
