"""
@author: OU,TING-WEI @ M.S. in Robotics 
date : 10-8-2023
Machien Learning HW2 ( NYCU FALL-2023 )
"""

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class LDA():
    def __init__(self):
        self.mean_class_1 = None
        self.mean_class_2 = None
        self.covariance = None
        self.C1 = 1
        self.C2 = 1
    
        self.w_T = None
        self.b = None
    

    def fit(self, x, y):
        class_1 = x[y == 1]
        class_2 = x[y == 0]

        n1 = class_1.shape[0]
        n2 = class_2.shape[0]
        
        p1 = n1/(n1+n2)
        p2 = n2/(n1+n2)

        self.mean_class_1 = np.mean(class_1, axis = 0)
        self.mean_class_2 = np.mean(class_2, axis = 0)

        covariance_1 = np.cov(class_1, rowvar=False)
        covariance_2 = np.cov(class_2, rowvar=False)
        covariance = covariance_1*p1 + covariance_2*p2

        self.w_T = (self.mean_class_1 - self.mean_class_2).T @ np.linalg.inv(covariance)
        self.b = -0.5 * self.w_T @ (self.mean_class_1 + self.mean_class_2) - np.log((self.C1 * p2) / (self.C2 * p1))

        self.w_T = np.round(self.w_T, 2)
        self.b = round(self.b, 2)

        print('[!] training weight vector : ', self.w_T, ' training bias : ', self.b)
        

    def LDA_decision_function(self, x, y_true):
        correct_predictions = 0

        x = np.array(x)

        for i in range(len(x)):
            x_col = np.array([[x[i][0], x[i][1]]]).T

            g = self.w_T @ x_col + self.b
            predicted_class = 1 if g > 0 else 0

            if predicted_class == y_true[i]:
                correct_predictions += 1
                print(i)

        accuracy = correct_predictions / len(y_true)

        print(accuracy)


def two_fold_cross_variation(data):

    # positive class = Versicolor  /  negative class = Virginica
    positive_class = 1
    negative_class = 2

    # adopt the third and forth typees of features
    selected_features = data[['petal_length', 'petal_width']]
    positive_data = selected_features[data['label'] == positive_class]
    negative_data = selected_features[data['label'] == negative_class]

    # split the data into training set & testing set
    training_data = pd.concat([positive_data.head(25), negative_data.head(25)], axis=0)
    testing_data = pd.concat([positive_data.tail(25), negative_data.tail(25)], axis=0)

    x_train = training_data
    y_train = np.concatenate((np.ones(25), np.zeros(25)))

    x_test = testing_data
    y_test = np.concatenate((np.ones(25), np.zeros(25)))

    lda = LDA()

    lda.fit(x_train, y_train)
    lda.LDA_decision_function(x_test, y_test)
    print('--------------------------------')
    
    lda.fit(x_test, y_test)
    lda.LDA_decision_function(x_train, y_train)

    
def main():
    data = pd.read_csv('iris.txt', delim_whitespace=True, header=None, engine='python')
    data = data.rename(columns={
        0: "sepal_length",
        1: "sepal_width",
        2: "petal_length",
        3: "petal_width",
        4: "label"})

    two_fold_cross_variation(data)


if __name__ == '__main__':
    main()