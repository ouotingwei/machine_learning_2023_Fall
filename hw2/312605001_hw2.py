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
    def __init__(self,):
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

        w_T = (self.mean_class_1 - self.mean_class_2).T @ np.linalg.inv(covariance)
        b = -0.5 * w_T @ (self.mean_class_1 + self.mean_class_2) - np.log((self.C1 * p2) / (self.C2 * p1))


    

        


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

