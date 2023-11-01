# -*- coding: utf-8 -*-

"""
@author: OU,TING-WEI , M.S. in Robotics @ NYCU
date : 10-31-2023
Machien Learning HW3 ( NYCU FALL-2023 )
"""

import numpy as np
import pandas as pd
from qpsolvers import solve_qp
import scipy.io
import math

class SVM():
    def __init__(self,data, selected_features, positive_class, negative_class, kernel_type='linear', C=1, sigma_p=1):
        # data processing
        self.positive_data = selected_features[data['label'] == positive_class]
        self.negative_data = selected_features[data['label'] == negative_class]

        self.training_data = pd.concat([self.positive_data.head(25), self.negative_data.head(25)], axis=0)
        self.testing_data = pd.concat([self.positive_data.tail(25), self.negative_data.tail(25)], axis=0)

        self.x_train = self.training_data.values
        self.y_train = np.concatenate((np.ones(25), -np.ones(25)))

        self.x_test = self.testing_data.values
        self.y_test = np.concatenate((np.ones(25), -np.ones(25)))

        # parameters
        self.alpha = None
        self.C = C
        self.sigma = sigma_p
        self.p = sigma_p
        self.kernel_type = kernel_type
        self.b = None
        self.alpha_sum = None

    def kernel_function(self, x, y):
        if self.kernel_type == 'linear':
            k = x @ y
        elif self.kernel_type == 'rbf':
            ...
        elif self.kernel_type == 'polynomial':
            ...
        return k
    
    def solve_parameters(self):
        #  alpha parameters preprocessing
        n = len(self.x_train)
        lb = np.full(n, 0)
        ub = np.full(n, self.C)

        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                k = self.kernel_function(self.x_train[i], self.x_train[j])
                P[i][j] = self.y_train[i] * self.y_train[j] * k       
                
        q = np.full(n, -1).T
        A = self.y_train
        b = np.array([0])

        # use qpsolvers to solve alpha
        alpha = solve_qp(P, q,None,None,A , b, lb, ub, solver="clarabel")

        # handling alpha values
        eps =   2.2204e-16
        for i in range(alpha.size):
            if alpha[i] >= self.C - np.sqrt(eps):
                alpha[i] = self.C
                alpha[i] = np.round(alpha[i],6)
            elif alpha[i] <= 0 + np.sqrt(eps):
                alpha[i] = 0
                alpha[i] = np.round(alpha[i],6)
            else:
                alpha[i] = np.round(alpha[i],6)
                print(f"support vector: alpha = {alpha[i]}")
        
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
        self.b = np.mean(np.array(b_list))
        print(self.b)

    def CR(self):
        predict = []
        for i in range(len(self.x_test)):
            sum_all = 0
            for j in range(len(self.x_train)):
                k = self.kernel_function(self.x_train[j], self.x_test[i])
                sum_all += (self.alpha[j]*self.y_train[j]*k)
            d = sum_all + self.b
            if d >= 0:
                predict.append(1)
            else:
                predict.append(-1)

        correct = np.sum(predict == self.y_test)
        accuracy = correct / len(self.x_test)
        print(f"Classification Rate (CR): {accuracy * 100:.2f}%")


    def auto_execute(cls, data, selected_features, positive_class, negative_class, kernel_type='linear', C=1, sigma_p=1):
        obj = cls(data, selected_features, positive_class, negative_class, kernel_type, C, sigma_p)
        obj.solve_parameters()
        obj.CR()


def main():
    data = pd.read_csv('iris.txt', delim_whitespace=True, header=None, engine='python')
    data = data.rename(columns={
        0: "sepal_length",
        1: "sepal_width",
        2: "petal_length",
        3: "petal_width",
        4: "label"})
    
    # initialize
    positive_class = 2
    negative_class = 3
    selected_features = data[['petal_length', 'petal_width']]

    # linear svm with c = 1
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'linear', C=1)
    # linear svm with c = 10
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'linear', C=10)
    # linear svm with c = 100
    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'linear', C=100)

    
if __name__ == '__main__':
    main()