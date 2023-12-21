from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
import lda 

# SFS
def sequential_forward_selection_2fold(X, y):

    print("------------------------SFS-----------------------------")

    num_features = X.shape[1]
    selected_features = []
    accuracies = []
    best_accuracy = 0
    best_loc = 0

    for i in range(num_features):   # 30
        remaining_features = [feature for feature in range(num_features) if feature not in selected_features]
        local_best_feature = None
        local_best_accuracy = 0

        for feature in remaining_features:
            current_features = selected_features + [feature]

            # Split the data into two folds for each iteration
            current_features = [feature for feature in current_features if feature is not None]
            fold1_X, fold2_X, fold1_y, fold2_y = train_test_split(X[:, np.array(current_features).astype(int)], y, test_size=0.5, random_state=0)

            # Evaluate the model on the first fold
            lda_ = lda.LDA()
            lda_.fit(fold1_X, fold1_y)
            fold_1_accuracy = lda_.LDA_decision_function(fold2_X, fold2_y) * 100

            # Evaluate the model on the second fold
            lda_.fit(fold2_X, fold2_y)
            fold_2_accuracy = lda_.LDA_decision_function(fold1_X, fold1_y) * 100

            # Calculate the average accuracy
            accuracy = (fold_1_accuracy + fold_2_accuracy) / 2

            if accuracy > local_best_accuracy:
                local_best_accuracy = accuracy
                local_best_feature = feature

        if local_best_feature is not None:
            selected_features.append(local_best_feature)
            accuracies.append(local_best_accuracy)
        
        if local_best_accuracy > best_accuracy:
            best_accuracy = local_best_accuracy
            best_loc = i

        print(f" Step {i + 1}: Selected Feature {local_best_feature}, Accuracy: {local_best_accuracy}")

    print(" [-] Best Accuracy:", best_accuracy)
    print(" [-] Features with the highest accuracy (indices):", selected_features[:best_loc])

    return selected_features, accuracies


def fisher_criterion_2fold(X, y):
    print("------------------------Fisher's Criterion-----------------------------")


def main():
    cancer = load_breast_cancer()
    x = cancer.data
    y = cancer.target

    sequential_forward_selection_2fold(x, y)
    fisher_criterion_2fold(x, y)


if __name__ == '__main__':
    main()
