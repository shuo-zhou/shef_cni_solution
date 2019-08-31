#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:15:26 2019

"""
import sys
import warnings
import numpy as np
from scipy.linalg import eig
import scipy.sparse as sparse
from numpy.linalg import multi_dot, inv, solve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
# import cvxpy as cvx
# from cvxpy.error import SolverError
from cvxopt import matrix, solvers
import osqp


def get_kernel(X, Y=None, kernel='linear', **kwargs):
    """
    Generate kernel matrix
    Parameters:
        X: X matrix (n1,d)
        Y: Y matrix (n2,d)
        kernel: 'linear'(default) | 'rbf' | 'poly'
    Return:
        Kernel matrix

    """

    return pairwise_kernels(X, Y=Y, metric=kernel,
                            filter_params=True, **kwargs)


class DISVM(BaseEstimator, TransformerMixin):
    def __init__(self, C=1, kernel='linear', lambda_=1, solver='cvxopt', **kwargs):
        """
        Init function
        Parameters
            n_components: n_componentss after tca (n_components <= d)
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            kernelparam: kernel param
            lambda_: regulization param
            gamma: label dependence param
            solver: cvxopt (default), osqp
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.C = C
        self.solver = solver

    def fit(self, X_train, X_test, y, D_train, D_test, W=None):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        Parameters:
            X_train: Training data, array-like, shape (n_train_samples, n_feautres)
            X_test: Testing data, array-like, shape (n_test_samples, n_feautres)
            y: Label, array-like, shape (n_train_samples, )
            D_train: Domain covariate matrix for training data, array-like, shape (n_train_samples, n_covariates)
            D_test: Domain covariate matrix for testing data, array-like, shape (n_test_samples, n_covariates)
        """

        n_train = X_train.shape[0]
        X = np.concatenate((X_train, X_test))
        n = X.shape[0]
        D = np.concatenate((D_train, D_test))
        Ka = np.dot(D, D.T)
        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))
        K = get_kernel(X, kernel=self.kernel, **self.kwargs)

        K[np.isnan(K)] = 0
        if W is None:
            W = np.eye(n)

        # dual
        Y = np.diag(y)
        J = np.zeros((n_train, n))
        J[:n_train, :n_train] = np.eye(n_train)
        Q_ = np.eye(n) + self.lambda_/np.square(n-1) * multi_dot([H, Ka, H, K])
        Q = multi_dot([Y, J, K, inv(Q_), J.T, Y])
        q = -1 * np.ones((n_train, 1))

        if self.solver == 'cvxopt':
            G = np.zeros((2 * n_train, n_train))
            G[:n_train, :] = -1 * np.eye(n_train)
            G[n_train:, :] = np.eye(n_train)
            h = np.zeros((2 * n_train, 1))
            h[n_train:, :] = self.C / n_train

            # convert numpy matrix to cvxopt matrix
            P = matrix(Q)
            q = matrix(q)
            G = matrix(G)
            h = matrix(h)
            A = matrix(y.reshape(1, -1).astype('float64'))
            b = matrix(np.zeros(1).astype('float64'))

            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h, A, b)

            self.alpha = np.array(sol['x']).reshape(n_train)

        elif self.solver == 'osqp':
            warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
            P = sparse.csc_matrix((n_train, n_train))
            P[:n_train, :n_train] = Q[:n_train, :n_train]
            G = sparse.vstack([sparse.eye(n_train), y.reshape(1, -1)]).tocsc()
            l = np.zeros((n_train+1, 1))
            u = np.zeros(l.shape)
            u[:n_train, 0] = self.C / n_train

            prob = osqp.OSQP()
            prob.setup(P, q, G, l, u, verbose=False)
            res = prob.solve()
            self.alpha = res.x

        else:
            print('Invalid QP solver')
            sys.exit()

        self.coef_ = multi_dot([inv(Q_), J.T, Y, self.alpha])
        self.support_ = np.where((self.alpha > 0) & (self.alpha < self.C))
        self.support_vectors_ = X_train[self.support_]
        self.n_support_ = self.support_vectors_.shape[0]
        # K_train = get_kernel(X_train, X, kernel=self.kernel, **self.kwargs)
        # self.intercept_ = np.mean(y[self.support_] - y[self.support_] *
        #                           np.dot(K_train[self.support_], self.coef_))/self.n_support_

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
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            prediction scores, array-like, shape (n_samples)
        """
        check_is_fitted(self, 'X')
        check_is_fitted(self, 'y')
        K = get_kernel(X, self.X, kernel=self.kernel, **self.kwargs)
        return np.dot(K, self.coef_)  # +self.intercept_

    def predict(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples)
        """
        
        return np.sign(self.decision_function(X))

    def fit_predict(self, X_train, X_test, y, D_train, D_test, W = None):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        Parameters:
            X_train: Training data, array-like, shape (n_train_samples, n_feautres)
            X_test: Testing data, array-like, shape (n_test_samples, n_feautres)
            y: Label, array-like, shape (n_train_samples, )
            D_train: Domain covariate matrix for training data, array-like, shape (n_train_samples, n_covariates)
            D_test: Domain covariate matrix for testing data, array-like, shape (n_test_samples, n_covariates)
        """
        self.fit(X_train, X_test, y, D_train, D_test, W)
        return self.predict(X_test)
