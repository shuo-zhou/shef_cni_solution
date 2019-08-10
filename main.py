# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 11:41:51 2019

@author: sz144
"""

import numpy as np
from numpy.linalg import multi_dot
import pandas as pd
import problem
#from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from did import DISVM

def sex_converter(sex_):
    sex = np.ones(sex_.shape)
    for i in sex_.index.values:
        if sex_.loc[i] == 'M':
            sex[i] = 1
        else:
            sex[i] = -1
            
    return sex

def get_hsic(X, Y, kernel_x='linear', kernel_y='linear', **kwargs):
    n = X.shape[0]
    I = np.eye(n)
    H = I - 1. / n * np.ones((n, n))
    Kx = pairwise_kernels(X, metric = kernel_x, **kwargs)
    Ky = pairwise_kernels(Y, metric = kernel_y, **kwargs)
    return np.trace(multi_dot([Kx, H, Ky, H])) / (n*n)
    

X_cc, pheno = problem.get_data(atlas='cc200', kind='partial correlation',return_pheno=True)
X_aal = problem.get_data(atlas='aal', kind='partial correlation')
X_ho = problem.get_data(atlas='ho', kind='partial correlation')
y = pheno['DX'].values
#clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.))


clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))


#results = cross_validate(clf, X, y, scoring=['roc_auc', 'accuracy'], cv=5,
#                         verbose=1, return_train_score=True, n_jobs=3)

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 144)
for train, test in skf.split(X_cc, y):
    clf_cc = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    clf_aal = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    clf_ho = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    clf_cc.fit(X_cc[train], y[train])
    clf_aal.fit(X_aal[train], y[train])
    clf_ho.fit(X_ho[train], y[train])


sex_ = pheno['Sex']
age_ = pheno['Age']
iq_ = pheno['WISC_FSIQ']
hand_ = pheno['Edinburgh_Handedness']

scaler = StandardScaler()

sex = sex_converter(sex_).reshape(-1, 1)
age = scaler.fit_transform(age_.values.reshape(-1, 1))
iq = scaler.fit_transform(iq_.values.reshape(-1, 1))
hand = scaler.fit_transform(hand_.values.reshape(-1, 1))




#print(get_hsic(X, sex))
#print(get_hsic(X, age))
#print(get_hsic(X, iq))
#print(get_hsic(X, hand))

#A = np.concatenate((sex, hand), axis=1)

#from sklearn.model_selection import StratifiedKFold
#for i in range(10):
#    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 144 * i)
#    pred = np.zeros(y.shape)
#    dec = np.zeros(y.shape)
#    for train, test in skf.split(X, y):
#        y_temp = np.zeros(y.shape)
#        y_temp[train] = y[train]
#        clf=DISVM()
#        clf.fit(X, y_temp, A)
#        pred[test]=clf.predict(X[test])
#        dec[test]=clf.decision_function(X[test])
#    print('Acc',accuracy_score(y, pred))
#    print('AUC',roc_auc_score(y, dec))

        
    