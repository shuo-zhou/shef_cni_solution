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
from nilearn.connectome import ConnectivityMeasure
from TPy.did import DISVM
from get_adhd200_data import load_adhd200

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

# =============================================================================
# Xs_tc, pheno_src = load_adhd200(atlas='aal')
# ys = pheno_src['DX'].values
# sex_src = pheno_src['Gender'].values.reshape(-1, 1)

# measure = ConnectivityMeasure(kind=kind, vectorize=True)
# Xs = measure.fit_transform(Xs_tc) 
# 
# sites = pheno_src['Site'].values
# uni_sites = np.unique(sites)
# site_mat = np.zeros((pheno_src.shape[0], uni_sites.shape[0]))
# for i in range(uni_sites.shape[0]):
#     site_mat[np.where(sites==uni_sites[i]), i] = 1
# site_mat=np.concatenate((np.zeros((pheno_src.shape[0], 1)), site_mat), axis=1)
# =============================================================================

#site_ = np.zeros((yt.shape[0], site_mat.shape[1]))
#site_[:,0]=1


Xcc, pheno = problem.get_data(atlas='cc200', kind=kind,return_pheno=True)
Xaal = problem.get_data(atlas='aal', kind=kind)
Xho = problem.get_data(atlas='ho', kind=kind)
yt = pheno['DX'].values


#clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.))


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
#A = hand
#A = np.concatenate((site_mat, site_))
#A = np.concatenate((np.concatenate((site_mat, site_)), 
#                    np.concatenate((sex_src, sex))), axis=1)
#A = np.concatenate((sex_src, sex))
scaler = StandardScaler()
#X = scaler.fit_transform(np.concatenate((Xcc, Xaal, Xho), axis=1))
X = Xaal[:,232:]#scaler.fit_transform(Xho[:,400:])
#X = np.concatenate((Xs, Xaal))
#X = scaler.fit_transform(X)
#y = np.concatenate((ys, yt))
y=yt
#X=Xho
acc = []
auc = []
#from sklearn.model_selection import StratifiedKFold

n_splits=2

for i in range(10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=144*i)
    pred = np.zeros(y.shape)
    dec = np.zeros(y.shape)
    for train, test in skf.split(X, y):
# =============================================================================
        y_temp = y.copy()
        y_temp[test] = 0
#        temp = np.zeros(y.shape)
#        temp[train] = 1
#        temp[test] = -1
#        temp = temp.reshape(-1,1)
#        temp_A = np.concatenate((temp, A), axis=1)
        clf=DISVM(kernel='linear', C=1)
        clf.fit(X, y_temp, A)
# =============================================================================
#        clf=make_pipeline(StandardScaler(), SVC(kernel='linear', C=0.1, max_iter=1000))
        # clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000))
        # clf.fit(X[train], y[train])
        
        pred[test]=clf.predict(X[test])
        dec[test]=clf.decision_function(X[test])
    acc.append(accuracy_score(y, pred))
    auc.append(roc_auc_score(y, dec))
    print('Acc',acc[-1])
    print('AUC',auc[-1])
print('Mean Auc: ',np.mean(auc), 'AUC std: ',np.std(auc)) 
print('Mean Acc: ',np.mean(acc), 'Acc std: ',np.std(acc))
    