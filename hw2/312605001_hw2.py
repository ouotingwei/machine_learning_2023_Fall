"""
@author: OU,TING-WEI @ M.S. in Robotics 
date : 10-8-2023
Machien Learning HW2 ( NYCU FALL-2023 )
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


class LDA():
    def __init__(self):
        self.mean_class_1 = None
        self.mean_class_2 = None
        self.w_T = None
        self.b = None
        self.TPR = None
        self.FPR = None


    def fit(self, x, y, C):
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
        self.b = -0.5 * self.w_T @ (self.mean_class_1 + self.mean_class_2) - np.log( C*(p2 / p1) )

        self.w_T = np.round(self.w_T, 2)
        self.b = round(self.b, 2)

        return self.w_T, self.b
        

    def LDA_decision_function(self, x, y_true):
        TP = FP = FN = TN = 0
        x = np.array(x)

        for i in range(len(x)):
            x_col = x[i].T

            g = self.w_T @ x_col + self.b
            predicted_class = 1 if g > 0 else 0

            if predicted_class == y_true[i] and predicted_class == 1:
                TP += 1
            if predicted_class == y_true[i] and predicted_class == 0:
                TN += 1
            if predicted_class != y_true[i] and predicted_class == 1:
                FP += 1
            if predicted_class != y_true[i] and predicted_class == 0:
                FN += 1

        self.TPR = TP / (TP + FN)
        self.FPR = FP / (FP + TN)

        accuracy = (TP+TN) / len(y_true)

        return accuracy
    
    def return_TPR_AND_FPR(self):
        return self.TPR, self.FPR


def two_fold_cross_variation_lda(data, selected_features, positive_class, negative_class, C=1):

    # adopt the third and forth typees of features
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

    training_w, training_b = lda.fit(x_train, y_train, C)
    testing_accuracy = (lda.LDA_decision_function(x_test, y_test))*100

    testing_w, testing_b = lda.fit(x_test, y_test, C)
    training_accuracy = (lda.LDA_decision_function(x_train, y_train))*100

    average_accuracy = (testing_accuracy + training_accuracy) / 2

    print('training w = ', training_w, 'training b = ', training_b,'testing accuracy', testing_accuracy, '%')
    print('testing w = ', testing_w, 'testing b = ', testing_b,'training accuracy', training_accuracy, '%')
    print('average accuracy = ', average_accuracy, '%')


def ROC_AND_AUC(data, selected_features, positive_class, negative_class):
    # adopt the third and forth typees of features
    positive_data = selected_features[data['label'] == positive_class]
    negative_data = selected_features[data['label'] == negative_class]

    # split the data into training set & testing set
    training_data = pd.concat([positive_data.head(25), negative_data.head(25)], axis=0)
    testing_data = pd.concat([positive_data.tail(25), negative_data.tail(25)], axis=0)

    x_train = training_data
    y_train = np.concatenate((np.ones(25), np.zeros(25)))

    x_test = testing_data
    y_test = np.concatenate((np.ones(25), np.zeros(25)))

    FPR_X = []
    TPR_Y = []

    for exp in range(-1500, 1500, 1): 
        C = 10 ** (exp / 10.0)

        lda = LDA()
        lda.fit(x_train, y_train, C)
        lda.LDA_decision_function(x_test, y_test)
        TPR_test, FPR_test = lda.return_TPR_AND_FPR()

        lda.fit(x_test, y_test, C)
        lda.LDA_decision_function(x_train, y_train)
        TPR_train, FPR_train = lda.return_TPR_AND_FPR()

        TPR_Y.append((TPR_test + TPR_train)/2)
        FPR_X.append((FPR_test + FPR_train)/2)

        # print((TPR_test + TPR_train)/2, (FPR_test + FPR_train)/2)

    roc_auc = auc(FPR_X, TPR_Y)

    print("AUC = ", roc_auc)

    # Plot the ROC curve
    plt.scatter(FPR_X, TPR_Y)
    plt.xlabel('False Positive Rate (FPR_X)')
    plt.ylabel('True Positive Rate (TPR_Y)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()


def main():
    data = pd.read_csv('iris.txt', delim_whitespace=True, header=None, engine='python')
    data = data.rename(columns={
        0: "sepal_length",
        1: "sepal_width",
        2: "petal_length",
        3: "petal_width",
        4: "label"})

    # Q1 : Use 2-fold cross-validation to calculate the CR value with LDA Classifier
    # positive class = Versicolor  /  negative class = Virginica
    positive_class = 2
    negative_class = 3

    C = 1

    selected_features = data[['petal_length', 'petal_width']]
    two_fold_cross_variation_lda(data, selected_features, positive_class, negative_class, C)
    print('--------------------------------')

    # Q2 : Plot the ROC and calculate the AUC scores
    # positive class = Virginica  /  negative class = Versicolor
    positive_class = 3
    negative_class = 2

    print('Q2 Feature = sepal_length, sepal_width, petal_length, petal_width')
    # Used all features
    selected_features = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    ROC_AND_AUC(data, selected_features, positive_class, negative_class)

    # Use sepal_length & sepal_width
    print('Q2  Feature = sepal_length, sepal_width')
    selected_features = data[['sepal_length', 'sepal_width']]
    ROC_AND_AUC(data, selected_features, positive_class, negative_class)

    # Use petal_length & petal_width
    print('Q2 Feature = petal_length, petal_width')
    selected_features = data[['petal_length', 'petal_width']]
    ROC_AND_AUC(data, selected_features, positive_class, negative_class)

    # Q3 : One against one strategy


if __name__ == '__main__':
    main()