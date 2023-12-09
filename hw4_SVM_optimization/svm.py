# -*- coding: utf-8 -*-

"""
@author: OU,TING-WEI , M.S. in Robotics @ NYCU
date : 10-31-2023
Machien Learning HW4 ( NYCU FALL-2023 )
"""

import numpy as np
import pandas as pd
from qpsolvers import solve_qp
import scipy.io
import math
from collections import defaultdict

class SVM():
    def __init__(self,x_train, y_train, x_test, y_test, kernel_type='linear', C=1, sigma_p=1):

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

        self.predict_list = []

    def kernel_function(self, x, y):
        if self.kernel_type == 'linear':
            k = x @ y
        elif self.kernel_type == 'rbf':
            norm_sq_neg = -1 * ( np.linalg.norm(x - y) ** 2 )
            k = math.exp(norm_sq_neg / (2 * (self.p ** 2)))
        elif self.kernel_type == 'polynomial':
            k = pow(x @ y, self.sigma)
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

        print_alpha = np.round(alpha, 4)
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
        
        self.b = np.mean(np.array(b_list))

        #print('b = ', np.round(b_list, 4))


    def predict(self):
        self.predict_list = []
        for i in range(len(self.x_test)):
            sum_all = 0
            for j in range(len(self.x_train)):
                k = self.kernel_function(self.x_train[j], self.x_test[i])
                sum_all += (self.alpha[j]*self.y_train[j]*k)
            d = sum_all + self.b
            if d >= 0:
                self.predict_list.append(1)
            else:
                self.predict_list.append(-1)
        
        return self.predict_list


    def auto_execute(cls, x_train, y_train, x_test, y_test, kernel_type='linear', C=1, sigma_p=1):
        obj = cls(x_train, y_train, x_test, y_test, kernel_type, C, sigma_p)
        obj.solve_parameters()
        predict_result = obj.predict()

        return predict_result  

def vote(list_1, list_2, list_3, real):
    correct_prediction = 0
    
    for i in range(len(list_1)):
        counts = defaultdict(int)  # Create a dictionary to count occurrences
        counts[list_1[i]] += 1
        counts[list_2[i]] += 1
        counts[list_3[i]] += 1

        max_count = max(counts.values())  # Find the maximum count
        predicted_value = [key for key, value in counts.items() if value == max_count][0]

        if real[i] == predicted_value:  # Compare the entire array, not individual elements
            correct_prediction += 1
    
    return round(correct_prediction / len(real), 4)


def main():
    data = pd.read_csv('iris.txt', delim_whitespace=True, header=None, engine='python')
    data = data.rename(columns={
        0: "sepal_length",
        1: "sepal_width",
        2: "petal_length",
        3: "petal_width",
        4: "label"})
    
    c_values = []
    sigma_values = []
    accuracy_values = []

    best_accuracy = 0  # 用于记录最高分类率
    best_c = None
    best_sigma = None
    
    # search range
    C_list = [1, 5, 10, 50, 100, 500, 1000]
    sigma_list = [1.05**(-i) for i in range(100, -100, -5)]

    #C_list = [1, 5]
    #sigma_list = [1.05**-100, 1.05**-95]
    
    # choose all fratures 
    selected_features = data[['sepal_length', 'sepal_width','petal_length', 'petal_width']]

    class_1 = 1  # Setosa
    class_2 = 2  # Versicolor
    class_3 = 3  # Virginica]

    data_1 = selected_features[data['label'] == class_1]
    data_2 = selected_features[data['label'] == class_2]
    data_3 = selected_features[data['label'] == class_3]

    real = np.concatenate((np.ones(25), np.full(25, 2), np.full(25, 3)))

    for c in C_list:
        for sigma in sigma_list:
            # fold-1
            # svm12
            training_data = pd.concat([data_1.head(25), data_2.head(25)], axis=0)
            testing_data = pd.concat([data_1.tail(25), data_2.tail(25), data_3.tail(25)], axis=0)

            x_train = training_data.values
            y_train = np.concatenate((np.ones(25), -np.ones(25)))

            x_test = testing_data.values
            y_test = np.concatenate((np.ones(25), -np.ones(50)))

            predict_1 = SVM.auto_execute(SVM, x_train, y_train, x_test, y_test, 'rbf', C = c, sigma_p = sigma)

            # reform the list
            for i in range(len(predict_1)):
                if predict_1[i] == 1:
                    predict_1[i] = class_1
                if predict_1[i] == -1:
                    predict_1[i] = class_2

            # svm23
            training_data = pd.concat([data_2.head(25), data_3.head(25)], axis=0)
            testing_data = pd.concat([data_2.tail(25), data_3.tail(25), data_1.tail(25)], axis=0)

            x_train = training_data.values
            y_train = np.concatenate((np.ones(25), -np.ones(25)))

            x_test = testing_data.values
            y_test = np.concatenate((np.ones(25), -np.ones(50)))

            predict_2 = SVM.auto_execute(SVM, x_train, y_train, x_test, y_test, 'rbf', C = c, sigma_p = sigma)

            # reform the list
            for i in range(len(predict_2)):
                if predict_2[i] == 1:
                    predict_2[i] = class_2
                if predict_2[i] == -1:
                    predict_2[i] = class_3
                    
            # svm13
            training_data = pd.concat([data_1.head(25), data_3.head(25)], axis=0)
            testing_data = pd.concat([data_1.tail(25), data_3.tail(25), data_2.tail(25)], axis=0)

            x_train = training_data.values
            y_train = np.concatenate((np.ones(25), -np.ones(25)))

            x_test = testing_data.values
            y_test = np.concatenate((np.ones(25), -np.ones(50)))

            predict_3 = SVM.auto_execute(SVM, x_train, y_train, x_test, y_test, 'rbf', C = c, sigma_p = sigma)

            # reform the list
            for i in range(len(predict_3)):
                if predict_3[i] == 1:
                    predict_3[i] = class_1
                if predict_3[i] == -1:
                    predict_3[i] = class_3

            accuracy_fold1 = vote(predict_1, predict_2, predict_3, real)
            #print('fold-1 = ', 100*accuracy_fold1, '%')

            # fold 2
            # svm12
            training_data = pd.concat([data_1.tail(25), data_2.tail(25)], axis=0)
            testing_data = pd.concat([data_1.head(25), data_2.head(25), data_3.head(25)], axis=0)

            x_train = training_data.values
            y_train = np.concatenate((np.ones(25), -np.ones(25)))

            x_test = testing_data.values
            y_test = np.concatenate((np.ones(25), -np.ones(50)))

            predict_1 = SVM.auto_execute(SVM, x_train, y_train, x_test, y_test, 'rbf', C=c, sigma_p=sigma)

            # reform the list
            for i in range(len(predict_1)):
                if predict_1[i] == 1:
                    predict_1[i] = class_1
                if predict_1[i] == -1:
                    predict_1[i] = class_2

            # svm23
            training_data = pd.concat([data_2.tail(25), data_3.tail(25)], axis=0)
            testing_data = pd.concat([data_2.head(25), data_3.head(25), data_1.head(25)], axis=0)

            x_train = training_data.values
            y_train = np.concatenate((np.ones(25), -np.ones(25)))

            x_test = testing_data.values
            y_test = np.concatenate((np.ones(25), -np.ones(50)))

            predict_2 = SVM.auto_execute(SVM, x_train, y_train, x_test, y_test, 'rbf', C=c, sigma_p=sigma)

            # reform the list
            for i in range(len(predict_2)):
                if predict_2[i] == 1:
                    predict_2[i] = class_2
                if predict_2[i] == -1:
                    predict_2[i] = class_3

            # svm13
            training_data = pd.concat([data_1.tail(25), data_3.tail(25)], axis=0)
            testing_data = pd.concat([data_1.head(25), data_3.head(25), data_2.head(25)], axis=0)

            x_train = training_data.values
            y_train = np.concatenate((np.ones(25), -np.ones(25)))

            x_test = testing_data.values
            y_test = np.concatenate((np.ones(25), -np.ones(50)))

            predict_3 = SVM.auto_execute(SVM, x_train, y_train, x_test, y_test, 'rbf', C=c, sigma_p=sigma)

            # reform the list
            for i in range(len(predict_3)):
                if predict_3[i] == 1:
                    predict_3[i] = class_1
                if predict_3[i] == -1:
                    predict_3[i] = class_3

            accuracy_fold2 = vote(predict_1, predict_2, predict_3, real)
            #print('fold-2 = ', 100 * accuracy_fold2, '%')

            accuracy = (accuracy_fold1 + accuracy_fold2) / 2

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_c = c
                best_sigma = round(sigma, 4)

            c_values.append(c)
            sigma_values.append(round(sigma, 4))
            accuracy_values.append(f'{accuracy:.2%}')

    # 创建表格
    table_data = {'C': c_values, 'Sigma': sigma_values, 'Accuracy': accuracy_values}
    table_df = pd.DataFrame(table_data)

    # 将 DataFrame 存储为 CSV 文件
    table_df.to_csv('/home/weiwei-robotic/machine_learning_2023_Fall/hw4_SVM_optimization/result.csv', index=False)

    print(f"Best CR: {best_accuracy:.2%}")
    print(f"c: {best_c}")
    print(f"sigma: {best_sigma}")

                    

if __name__ == "__main__":
    main()