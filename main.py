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
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    
kind= 'tangent'
#kind= 'covariance'
#kind= 'correlation'
Xcc, pheno = problem.get_data(atlas='cc200', kind=kind,return_pheno=True)
Xaal = problem.get_data(atlas='aal', kind=kind)
Xho = problem.get_data(atlas='ho', kind=kind)
y = pheno['DX'].values
#clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.))


clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))


#results = cross_validate(clf, X, y, scoring=['roc_auc', 'accuracy'], cv=5,
#                         verbose=1, return_train_score=True, n_jobs=3)

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 144)

#pred = np.zeros(y.shape)
#prob = np.zeros(y.shape)
#for train, test in skf.split(Xcc, y):
#    clf_cc = make_pipeline(StandardScaler(), LogisticRegression(C=1.))#SVC(kernel='linear', probability=True))
#    clf_aal = make_pipeline(StandardScaler(), LogisticRegression(C=1.))#SVC(kernel='linear', probability=True))
#    clf_ho = make_pipeline(StandardScaler(), LogisticRegression(C=1.))#SVC(kernel='linear', probability=True))
#    tr_idx, va_idx = train_test_split(range(y[train].size), test_size=0.33,
#                                      shuffle=True, random_state=42)
#    
#    
#    
#    clf_cc.fit(Xcc[train][tr_idx], y[train][tr_idx])
#    clf_aal.fit(Xaal[train][tr_idx], y[train][tr_idx])
#    clf_ho.fit(Xho[train][tr_idx], y[train][tr_idx])
#    
#    prob_cc_valid = clf_cc.predict_proba(Xcc[train][va_idx])
#    prob_aal_valid = clf_aal.predict_proba(Xaal[train][va_idx])
#    prob_ho_valid = clf_ho.predict_proba(Xho[train][va_idx])
#    
#    meta_clf = LogisticRegression(C=1.)
#    
#    meta_clf.fit(np.concatenate([prob_cc_valid, prob_aal_valid, prob_ho_valid], 
#                                axis=1), y[train][va_idx])
#    
#    prob_cc_test = clf_cc.predict_proba(Xcc[test])
#    prob_aal_test = clf_aal.predict_proba(Xaal[test])
#    prob_ho_test = clf_ho.predict_proba(Xho[test])
#    
#    pred[test]=meta_clf.predict(np.concatenate([prob_cc_test, prob_aal_test, prob_ho_test], axis=1))
#    prob[test]=meta_clf.predict_proba(np.concatenate([prob_cc_test, prob_aal_test, prob_ho_test], axis=1))[:,0]
    
    
# =============================================================================
#     clf_cc = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
#     clf_cc.fit(Xcc[train], y[train])
#     pred[test] = clf_cc.predict(Xcc[test])
#     prob[test] = clf_cc.predict_proba(Xcc[test])[:,0]
# =============================================================================
    
    
#print('Acc',accuracy_score(y, pred))
#print('AUC',roc_auc_score(y, prob))

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

A = np.concatenate((sex, hand), axis=1)
#A = sex
scaler = StandardScaler()
X = scaler.fit_transform(Xcc)
#X=Xho
acc = []
auc = []
#from sklearn.model_selection import StratifiedKFold
for i in range(10):
    skf = StratifiedKFold(n_splits = 2, shuffle = True, random_state = 144 * i)
    pred = np.zeros(y.shape)
    dec = np.zeros(y.shape)
    for train, test in skf.split(X, y):
        y_temp = np.zeros(y.shape)
        y_temp[train] = y[train]
        clf=DISVM()
        clf.fit(X, y_temp, A)
        pred[test]=clf.predict(X[test])
        dec[test]=clf.decision_function(X[test])
    acc.append(accuracy_score(y, pred))
    auc.append(roc_auc_score(y, dec))
    print('Acc',acc[-1])
    print('AUC',auc[-1])

        
    