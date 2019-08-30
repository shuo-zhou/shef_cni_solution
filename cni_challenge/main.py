# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 11:41:51 2019

@author: sz144
"""
import scipy
import numpy as np
from numpy.linalg import multi_dot
import pandas as pd
import problem
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


def sex2onehot(sex_):
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
    Kx = pairwise_kernels(X, metric=kernel_x, **kwargs)
    Ky = pairwise_kernels(Y, metric=kernel_y, **kwargs)
    return 1/np.square(n-1) * np.trace(multi_dot([Kx, H, Ky, H])) 


kind = 'tangent'
# kind = 'covariance'
# kind = 'correlation'

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

# site_ = np.zeros((yt.shape[0], site_mat.shape[1]))
# site_[:, 0]=1

# Xcc, pheno = problem.get_data(atlas='cc200', kind=kind, return_pheno=True)
# Xaal = problem.get_data(atlas='aal', kind=kind)
# Xho = problem.get_data(atlas='ho', kind=kind)
Xcc_tan, pheno = problem.get_data(atlas='cc200', kind='tangent', return_pheno=True)
Xcc_cor = problem.get_data(atlas='cc200', kind='correlation')
Xcc_cov = problem.get_data(atlas='cc200', kind='covariance')

X = dict()
X['cc1'] = Xcc_tan[:, 400:]
X['cc2'] = Xcc_cor[:, 400:]
X['cc3'] = Xcc_cov[:, 400:]
X['cc4'] = Xcc_tan[:, :400]

Xaal_tan = problem.get_data(atlas='aal', kind='tangent')
Xaal_cor = problem.get_data(atlas='aal', kind='correlation')
Xaal_cov = problem.get_data(atlas='aal', kind='covariance')

X['aa1'] = Xaal_tan[:, 232:]
X['aa2'] = Xaal_cor[:, 232:]
X['aa3'] = Xaal_cov[:, 232:]
X['aa4'] = Xaal_tan[:, :232]

Xho_tan = problem.get_data(atlas='ho', kind='tangent')
Xho_cor = problem.get_data(atlas='ho', kind='correlation')
Xho_cov = problem.get_data(atlas='ho', kind='covariance')

X['ho1'] = Xho_tan[:, 220:]
X['ho2'] = Xho_cor[:, 220:]
X['ho3'] = Xho_cov[:, 220:]
X['ho4'] = Xho_tan[:, :220]

yt = pheno['DX'].values

scaler = StandardScaler()

for key in X:
    X[key] = scaler.fit_transform(X[key])

# clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.))

# results = cross_validate(clf, X, y, scoring=['roc_auc', 'accuracy'], cv=5,
#                         verbose=1, return_train_score=True, n_jobs=3)

y = yt

sex_ = pheno['Sex']
age_ = pheno['Age']
iq_ = pheno['WISC_FSIQ']
hand_ = pheno['Edinburgh_Handedness']

scaler = StandardScaler()
sex = sex2onehot(sex_).reshape(-1, 1)
age = scaler.fit_transform(age_.values.reshape(-1, 1))
iq = scaler.fit_transform(iq_.values.reshape(-1, 1))
hand = scaler.fit_transform(hand_.values.reshape(-1, 1))

D = np.concatenate((sex, hand), axis=1)

acc = []
auc = []
for i in range(10):
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=144*i)

    pred = np.zeros(y.shape)
    prob = np.zeros(y.shape)
    score = np.zeros(y.shape)

    for train, test in skf.split(Xcc_tan, yt):
        # tr_idx, va_idx = train_test_split(range(y[train].size), test_size=0.5,
        #                                   shuffle=True, random_state=42)
        # clf = dict()
        # scores = []
        # for key in X:
        #     Xdata = X[key]
        #     clf[key] = make_pipeline(StandardScaler(), SVC(kernel='linear', max_iter=10000))
        #     clf[key].fit(Xdata[train][tr_idx], y[train][tr_idx])
        #     scores.append(clf[key].decision_function(Xdata[train][va_idx]).reshape(-1, 1))
        #
        # meta_clf = LogisticRegression(C=1., solver='lbfgs', max_iter=10000)
        # meta_clf.fit(np.concatenate(scores, axis=1), y[train][va_idx])
        #
        # scores_ = []
        # for key in X:
        #     Xdata = X[key]
        #     scores_.append(clf[key].predict(Xdata[test]).reshape(-1, 1))
        #
        # pred[test] = meta_clf.predict(np.concatenate(scores_, axis=1))
        # prob[test] = meta_clf.predict_proba(np.concatenate(scores_, axis=1))[:, 0]

        clf = dict()
        score_list = []
        pred_list = []
        for key in X:
            Xdata = X[key]
            # clf[key] = make_pipeline(StandardScaler(), SVC(kernel='linear', max_iter=10000))
            clf[key] = DISVM(kernel='linear', C=0.1, lambda_=0.1)
            clf[key].fit(Xdata[train], Xdata[test], y[train], D[train], D[test])
            pred_list.append(clf[key].predict(Xdata[test]).reshape(-1, 1))
            score_list.append(clf[key].decision_function(Xdata[test]).reshape(-1, 1))

        preds = np.concatenate(pred_list, axis=1)
        scores = np.concatenate(pred_list, axis=1)
        # pred[test] = scipy.stats.mode(preds, axis=1)[0].reshape(pred[test].shape)
        score[test] = np.mean(scores, axis=1)
        pred[test] = np.sign(score[test])

    acc.append(accuracy_score(y, pred))
    auc.append(roc_auc_score(y, score))
    print('Acc', acc[-1])
    print('AUC', auc[-1])

print('Mean Auc: ', np.mean(auc), 'AUC std: ', np.std(auc))
print('Mean Acc: ', np.mean(acc), 'Acc std: ', np.std(acc))



# print(get_hsic(X, sex))
# print(get_hsic(X, age))
# print(get_hsic(X, iq))
# print(get_hsic(X, hand))


# A = hand
# A = np.concatenate((site_mat, site_))
# A = np.concatenate((np.concatenate((site_mat, site_)),
#                    np.concatenate((sex_src, sex))), axis=1)
# A = np.concatenate((sex_src, sex))

# X = scaler.fit_transform(np.concatenate((Xcc, Xaal, Xho), axis=1))
X = scaler.fit_transform(Xcc)
# X = np.concatenate((Xs, Xaal))
# X = scaler.fit_transform(X)
# y = np.concatenate((ys, yt))
y = yt
# X = Xho
acc = []
auc = []

n_splits = 2

for i in range(10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=144*i)
    pred = np.zeros(y.shape)
    dec = np.zeros(y.shape)
    for train, test in skf.split(X, y):
        clf = DISVM(kernel='linear', C=1, lambda_=1, solver='osqp')
        clf.fit(X[train], X[test], y[train], A[train], A[test])

        # clf=make_pipeline(StandardScaler(), SVC(kernel='linear', C=0.1, max_iter=1000))
        # clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000))
        # clf.fit(X[train], y[train])

        pred[test] = clf.predict(X[test])
        dec[test] = clf.decision_function(X[test])
    acc.append(accuracy_score(y, pred))
    auc.append(roc_auc_score(y, dec))
    print('Acc', acc[-1])
    print('AUC', auc[-1])
print('Mean Auc: ', np.mean(auc), 'AUC std: ', np.std(auc))
print('Mean Acc: ', np.mean(acc), 'Acc std: ', np.std(acc))
