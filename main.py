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
    

X, pheno = problem.get_data(atlas='ho')
y = pheno['DX']
#clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.))
clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))

results = cross_validate(clf, X, y, scoring=['roc_auc', 'accuracy'], cv=5,
                         verbose=1, return_train_score=True, n_jobs=3)


sex_ = pheno['Sex']
age_ = pheno['Age']
iq_ = pheno['WISC_FSIQ']
hand_ = pheno['Edinburgh_Handedness']

scaler = StandardScaler()

sex = sex_converter(sex_).reshape(-1, 1)
age = scaler.fit_transform(age_.values.reshape(-1, 1))
iq = scaler.fit_transform(iq_.values.reshape(-1, 1))
hand = scaler.fit_transform(hand_.values.reshape(-1, 1))

print(get_hsic(X, sex))
print(get_hsic(X, age))
print(get_hsic(X, iq))
print(get_hsic(X, hand))