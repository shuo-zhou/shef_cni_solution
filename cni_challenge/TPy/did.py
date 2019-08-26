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

    def fit(self, X_train, X_test, y, A_train, A_test, W=None):
        '''
        solve min_x x^TPx + q^Tx, s.t. Gx<=h
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (n_samples, )
            A: Domain auxiliary features, array-like, shape (n_samples, n_feautres)
        '''

        n_train = X_train.shape[0]
        X = np.concatenate((X_train, X_test))
        n = X.shape[0]
        A = np.concatenate((A_train, A_test))
        Ka = np.dot(A, A.T)
        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))
        K = get_kernel(X, kernel=self.kernel, **self.kwargs)
        K_train = get_kernel(X_train, X, kernel=self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0
        if W is None:
            W = np.eye(n)

        # dual
        Y = np.diag(y)
        J = np.zeros((n_train, n))
        J[:n_train, :n_train] = np.eye(n_train)
        Q_ = np.eye(n) + self.lambda_ * multi_dot([H, Ka, H, K])
        Q = multi_dot([Y, J, K, inv(Q_), J.T, Y])
        q = -1 * np.ones((n_train, 1))
        G = np.zeros((2*n_train, n_train))
        G[:n_train, :] = -1 * np.eye(n_train)
        G[n_train:, :] = np.eye(n_train)
        
        h = np.zeros((2*n_train, 1))
        h[n_train:, :] = self.C
        
        # convert numpy matrix to cvxopt matrix
        P = matrix(Q)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)

        # dual only
        A = matrix(y.reshape(1, -1).astype('double'))
        b = matrix(np.zeros(1).astype('double'))
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        # sol = solvers.qp(P, q, G, h)

        # solve dual 
        self.alpha = np.array(sol['x']).reshape(n_train)
        self.coef_ = multi_dot([inv(Q_), J.T, Y, self.alpha])
        self.support_ = (self.alpha > 1e-4).flatten()
        self.support_vectors_ = X_train[self.support_]
        self.intercept_ = np.mean(y[self.support_] - y[self.support_] *
                                  np.dot(K_train[self.support_], self.coef_))

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

        self.X = X
        self.y = y
        
        return self

    def decision_function(self, X):
        check_is_fitted(self, 'X')
        check_is_fitted(self, 'y')

        X_fit = self.X

        # primal
        K = get_kernel(X, X_fit, kernel=self.kernel, **self.kwargs)
        return np.dot(K, self.coef_)#+self.intercept_

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
