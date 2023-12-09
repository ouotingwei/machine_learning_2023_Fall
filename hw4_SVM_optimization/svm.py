# -*- coding: utf-8 -*-

"""
@author: OU,TING-WEI , M.S. in Robotics @ NYCU
date : 10-31-2023
Machien Learning HW4 ( NYCU FALL-2023 )
"""

import numpy as np
from qpsolvers import solve_qp
import scipy.io
import math

class SVM():
    def __init__(self,x_train, y_train, x_test, y_test, kernel_type='rbf', C=1, sigma_p=1):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
       
        # parameters
        self.alpha = None
        self.C = C
        self.sigma = sigma_p
        self.p = sigma_p
        self.kernel_type = kernel_type
        self.b = None
        self.alpha_sum = None

        self.predict_list = None

    def kernel_function(self, x, y):
        norm_sq_neg = -1 * ( np.linalg.norm(x - y) ** 2 )
        k = math.exp(norm_sq_neg / (2 * (self.p ** 2)))
        return k
    
    def solve_parameters(self):
        #  alpha parameters preprocessing
        n = len(self.x_train)
        lb = np.full(n, 0)
        ub = np.full(n, self.C)

        P = scipy.sparse.lil_matrix((n, n))
        for i in range(n):
            for j in range(n):
                k = self.kernel_function(self.x_train[i], self.x_train[j])
                P[i, j] = self.y_train[i] * self.y_train[j] * k

        P = P.tocsc()  # transfer into csc_matrix

        q = np.full(n, -1).T
        A = self.y_train
        A = scipy.sparse.csc_matrix(A)
        b = np.array([0])

        # use qpsolvers to solve alpha
        alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver="clarabel")

        # handling alpha values
        eps = 2.2204e-16
        for i in range(alpha.size):
            if alpha[i] >= self.C - np.sqrt(eps):
                alpha[i] = self.C
                alpha[i] = np.round(alpha[i], 6)
            elif alpha[i] <= 0 + np.sqrt(eps):
                alpha[i] = 0
                alpha[i] = np.round(alpha[i], 6)
            else:
                alpha[i] = np.round(alpha[i], 6)
                #print(f"support vector: alpha = {alpha[i]}")

        #print_alpha = np.round(alpha, 4)
        #print('alpha = ', print_alpha)

        self.alpha = alpha
        self.alpha_sum = np.round(np.sum(self.alpha), 4)

        # solve b*
        sum = 0
        b_list = []

        for i in range(len(self.alpha)):
            if self.alpha[i] > 0 and self.alpha[i] < self.C:
                sum = 0
                for j in range(len(self.alpha)):
                    k = self.kernel_function(self.x_train[j], self.x_train[i])
                    sum += self.alpha[j] * self.y_train[j] * k
                bias = 1.0 / self.y_train[i] - sum

                b_list.append(bias)
        
        if b_list:
            self.b = np.mean(np.array(b_list))
            print('b = ', np.round(self.b, 4))
        else:
            print('b_list is empty, cannot calculate self.b')


    def predict(self):
        if self.b is None:
            self.predict_list = np.full(len(self.x_test), np.nan)
            return self.predict_list

        self.predict_list = []
        for i in range(len(self.x_test)):
            sum_all = 0
            for j in range(len(self.x_train)):
                k = self.kernel_function(self.x_train[j], self.x_test[i])
                sum_all += (self.alpha[j] * self.y_train[j] * k)
            d = sum_all + self.b
            if d >= 0:
                self.predict_list.append(1)
            else:
                self.predict_list.append(-1)

        return self.predict_list

'''
    def auto_execute(self, x_train, y_train, x_test, y_test, kernel_type='linear', C=1, sigma_p=1):
        obj = SVM(x_train, y_train, x_test, y_test, kernel_type, C, sigma_p)
        obj.solve_parameters()
        predict_result = obj.predict()

        return predict_result
'''