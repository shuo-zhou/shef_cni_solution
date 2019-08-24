#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:15:26 2019

"""
import numpy as np
from scipy.linalg import eig
from numpy.linalg import multi_dot, solve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
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
    def __init__(self, C= 1, kernel='linear', lambda_=1, **kwargs):
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

    def fit(self, X, y, A, W=None):
        '''
        solve min_x x^TPx + q^Tx, s.t. Gx<=h
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (n_samples, )
            A: Domain auxiliary features, array-like, shape (n_samples, n_feautres)
        '''
        n = X.shape[0]
        Ka = np.dot(A, A.T)
        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))
        K = get_kernel(X, kernel = self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0
        if W is None:
            W = np.eye(n)
            
        P = np.zeros((2*n+1, 2*n+1))
        P[:n, :n] = K + self.lambda_ * 1/np.square(n-1) * multi_dot([K, H, Ka, H, K])
        
        q = np.zeros((2*n+1, 1))
        q[n:, :] = self.C
        
#        y = y.reshape((n, 1))
        G = np.zeros((2*n, 2*n+1))
        G[:n, :n] = -np.multiply(K, y.reshape((n, 1)))
        G[:n, n] = -y
        G[:n, n+1:] = -np.eye(n)
        G[n:, n+1:] = -np.eye(n)
        
        h = np.zeros((2*n, 1))
        h[:n, :] = -1
        
        # convert numpy matrix to cvxopt matrix
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(P,q,G,h)
        
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
        
        return self

    def decision_function(self, X):
        check_is_fitted(self, 'X')

        X_fit = self.X
        K = get_kernel(X, X_fit, kernel = self.kernel, **self.kwargs)
        return np.dot(K, self.coef_)+self.intercept_
    
    def predict(self, X):
        '''
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples)
        '''
        
        return np.sign(self.decision_function(X))


    def fit_predict(self, X, y, A, W = None):
        '''
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (n_samples, )
            A: Domain auxiliary features, array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples)
        '''
        self.fit(X, A, y, W)
        y_pred = self.predict(X)
        return y_pred
