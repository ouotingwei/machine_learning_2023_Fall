"""
@author: OU,TING-WEI @ M.S. in Robotics 
date : 10-8-2023
Machien Learning HW2 ( NYCU FALL-2023 )
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from collections import defaultdict


class LDA():
    def __init__(self):
        self.mean_class_1 = None
        self.mean_class_2 = None
        self.w_T = None
        self.b = None
        self.TPR = None
        self.FPR = None
        self.predicted_list = None


    def fit(self, x, y, C=1):
        class_1 = x[y == 1]
        class_2 = x[y == 0]

        n1 = class_1.shape[0]
        n2 = class_2.shape[0]
        
        p1 = n1 / (n1 + n2)
        p2 = n2 / (n1 + n2)

        if x.shape[1] == 1:  # If there is only one feature
            mean_class_1 = np.mean(class_1)
            mean_class_2 = np.mean(class_2)

            covariance = np.var(class_1) * p1 + np.var(class_2) * p2

            w_T = (mean_class_1 - mean_class_2) / covariance
            b = -0.5 * w_T * (mean_class_1 + mean_class_2) - np.log(C * (p2 / p1))

            w_T = np.array([w_T])  # Ensure w_T is a vector
            w_T = np.round(w_T, 2)
            b = round(b, 2)

        else:
            self.mean_class_1 = np.mean(class_1, axis=0)
            self.mean_class_2 = np.mean(class_2, axis=0)

            covariance_1 = np.cov(class_1, rowvar=False)
            covariance_2 = np.cov(class_2, rowvar=False)
            covariance = covariance_1 * p1 + covariance_2 * p2

            w_T = (self.mean_class_1 - self.mean_class_2).T @ np.linalg.inv(covariance)
            b = -0.5 * w_T @ (self.mean_class_1 + self.mean_class_2) - np.log(C * (p2 / p1))

            w_T = np.round(w_T, 2)
            b = round(b, 2)

        self.w_T = np.array([w_T])  # Ensure w_T is a vector
        b = 0 if x.shape[1] == 1 else b  # Set a default value for b if using multiple features

        return w_T, b



        

    def LDA_decision_function(self, x, y_true):
        TP = FP = FN = TN = 0
        x = np.array(x)
        self.predicted_list = []

        if x.ndim == 1:  # If there is only one feature
            for i in range(len(x)):
                g = self.w_T * x[i] + self.b
                predicted_class = 1 if g > 0 else 0
                self.predicted_list.append(predicted_class)

                if predicted_class == y_true[i] and predicted_class == 1:
                    TP += 1
                if predicted_class == y_true[i] and predicted_class == 0:
                    TN += 1
                if predicted_class != y_true[i] and predicted_class == 1:
                    FP += 1
                if predicted_class != y_true[i] and predicted_class == 0:
                    FN += 1

        else:  # If there are multiple features
            for i in range(len(x)):
                x_col = x[i].T
                if self.b is not None:  # Check if self.b is not None
                    g = self.w_T @ x_col + self.b
                    predicted_class = 1 if g > 0 else 0
                    self.predicted_list.append(predicted_class)

                    if predicted_class == y_true[i] and predicted_class == 1:
                        TP += 1
                    if predicted_class == y_true[i] and predicted_class == 0:
                        TN += 1
                    if predicted_class != y_true[i] and predicted_class == 1:
                        FP += 1
                    if predicted_class != y_true[i] and predicted_class == 0:
                        FN += 1

        # Add a condition to check if the denominator is zero
        if TP + FN != 0:
            self.TPR = TP / (TP + FN)
        else:
            self.TPR = 0

        #self.FPR = FP / (FP + TN)

        accuracy = (TP + TN) / len(y_true)

        return accuracy



    
    def return_TPR_AND_FPR(self):
        return self.TPR, self.FPR
    
    def return_predicted_list(self):
        return self.predicted_list