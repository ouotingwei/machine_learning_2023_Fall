from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from collections import defaultdict
import lda 

# SFS
def sequential_forward_selection_2fold(X, y):
    num_features = X.shape[1]
    selected_features = []
    best_accuracy = 0

    for _ in range(num_features):   # 30
        remaining_features = [feature for feature in range(num_features) if feature not in selected_features]
        local_best_feature = None

        for feature in remaining_features:
            current_features = selected_features + [feature]

            # Split the data into two folds for each iteration
            current_features = [feature for feature in current_features if feature is not None]
            fold1_X, fold2_X, fold1_y, fold2_y = train_test_split(X[:, np.array(current_features).astype(int)], y, test_size=0.5, random_state=0)


            # Evaluate the model on the first fold
            lda_ = lda.LDA()
            lda_.fit(fold1_X, fold1_y)
            fold_1_accuracy = lda_.LDA_decision_function(fold2_X, fold2_y)*100

            # Evaluate the model on the second fold
            lda_.fit(fold2_X, fold2_y)
            fold_2_accuracy = lda_.LDA_decision_function(fold1_X, fold1_y)*100

            # Calculate the average accuracy
            accuracy = (fold_1_accuracy + fold_2_accuracy)/2

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                local_best_feature = feature

        selected_features.append(local_best_feature)
        print(f"Step {_ + 1}: Selected Feature {local_best_feature}, Accuracy: {best_accuracy}")


    return selected_features

def main():
    cancer = load_breast_cancer()
    x = cancer.data
    y = cancer.target
    print(x.shape)

    sequential_forward_selection_2fold(x, y)


if __name__ == '__main__':
    main()
