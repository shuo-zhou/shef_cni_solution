#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:15:26 2019

"""
import numpy as np
from scipy.linalg import eig
from numpy.linalg import multi_dot, inv, solve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
#import cvxpy as cvx
#from cvxpy.error import SolverError
from cvxopt import matrix, solvers

def get_kernel(X, Y=None, kernel = 'linear', **kwargs):
    '''
    Generate kernel matrix
    Parameters:
        X: X matrix (n1,d)
        Y: Y matrix (n2,d)
    Return: 
        Kernel matrix
    '''

    return pairwise_kernels(X, Y=Y, metric = kernel, 
                            filter_params = True, **kwargs)

class DISVM(BaseEstimator, TransformerMixin):
    def __init__(self, C=1, kernel='linear', lambda_=1, **kwargs):
        '''
        Init function
        Parameters
            n_components: n_componentss after tca (n_components <= d)
            kernel_type: 'rbf' | 'linear' | 'poly' (default is 'linear')
            kernelparam: kernel param
            lambda_: regulization param
            gamma: label dependence param
        '''
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.C = C

    def fit(self, X, y, A, train_index, W=None):
        '''
        solve min_x x^TPx + q^Tx, s.t. Gx<=h
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (n_samples, )
            A: Domain auxiliary features, array-like, shape (n_samples, n_feautres)
        '''
        n = X.shape[0]
        X_train = X[train_index]
        n_train = X_train.shape[0]
        Ka = np.dot(A, A.T)
        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))
        K = get_kernel(X, kernel = self.kernel, **self.kwargs)
        K_train = get_kernel(X_train, X, kernel=self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0
        if W is None:
            W = np.eye(n)

        # self.scaler = StandardScaler()
        S = np.eye(n)/(n-1)
        P = np.zeros((n+n_train+1, n+n_train+1))
        P[:n, :n] = multi_dot([S, K, S]) + self.lambda_ * multi_dot([S, K, H, Ka, H, K, S])
        # P[n+1, n+1] = 1
        
        q = np.zeros((n+n_train+1, 1))
        q[n+1:, :] = self.C
        
#        y = y.reshape((n, 1))
        G = np.zeros((2*n_train, n+n_train+1))
        G[:n_train, :n] = -np.multiply(K_train, y.reshape((n_train, 1)))
        G[:n_train, n] = -y
        G[:n_train, n+1:] = -np.eye(n_train)
        G[n_train:, n+1:] = -np.eye(n_train)
        
        h = np.zeros((2*n_train, 1))
        h[:n_train, :] = -1

        # dual         
        # P = multi_dot([np.multiply(K, y).T, inv(K + self.lambda_ * multi_dot([K, H, Ka, H, K])), np.multiply(K, y)])
        # q = -1 * np.ones((n, 1))
        # G = np.zeros((2*n, n))
        # G[:n, :] = -np.eye(n)
        # G[n:, :] = np.eye(n)
        
        # h = np.zeros((2*n, 1))
        # h[n:, :] = self.C
        
        # convert numpy matrix to cvxopt matrix
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)

        # dual only
        # A = matrix(y.reshape(1, -1).astype('double'))
        # b = matrix(np.zeros(1).astype('double'))
        
        solvers.options['show_progress'] = False
        # sol = solvers.qp(P, q, G, h, A, b)
        sol = solvers.qp(P, q, G, h)

        # solve dual 
        # self.alpha = np.array(sol['x']).reshape(n)
        # if self.kernel == 'linear':
        #     self.coef_ = np.dot((y * self.alpha), X)
        # self.support_idx = (self.alpha > 1e-4).flatten()
        # self.support_ = X[self.support_idx]
        # self.intercept_ = y[self.support_idx] - np.dot(K[self.support_idx], (y * self.alpha))

        # solve primal
        self.coef_ = sol['x'][:n]
        self.coef_ = np.array(self.coef_).reshape(n)
        self.intercept_ = sol['x'][n]
        
# =============================================================================
#         beta = cvx.Variable(shape = (2 * n, 1))
#         objective = cvx.Minimize(cvx.quad_form(beta, P) + q.T * beta)
#         constraints = [G * beta <= h]
#         prob = cvx.Problem(objective, constraints)
#         try:
#             prob.solve()
#         except SolverError:
#             prob.solve(solver = 'SCS')
#         
#         self.coef_ = beta.value[:n]
# =============================================================================
        
#        a = np.dot(W + self.gamma * multi_dot([H, Ka, H]), self.lambda_*I)
#        b = np.dot(y, W)
#        beta = solve(a, b)
        
        self.beta = sol['x']
        self.X = X
        self.y = y
        
        return self

    def decision_function(self, X):
        check_is_fitted(self, 'X')
        check_is_fitted(self, 'y')

        X_fit = self.X

        # primal
        K = get_kernel(X, X_fit, kernel=self.kernel, **self.kwargs)
        return np.dot(K, self.coef_)+self.intercept_

        # dual
        # K = get_kernel(X, X_fit, kernel = self.kernel, **self.kwargs)
        # return np.dot(K, (self.y * self.alpha))+self.intercept_[0]
    
    def predict(self, X):
        '''
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples)
        '''
        
        return np.sign(self.decision_function(X))


    def fit_predict(self, X, y, A, train_index, W = None):
        '''
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (n_samples, )
            A: Domain auxiliary features, array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples)
        '''
        self.fit(X, y, A, train_index, W)
        y_pred = self.predict(X)
        return y_pred
