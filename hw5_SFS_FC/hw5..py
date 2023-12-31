from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

            # Initialize LDA
            lda_ = LDA()

            # Fit and evaluate the model on the first fold
            lda_.fit(fold1_X, fold1_y)
            fold_1_accuracy = lda_.score(fold2_X, fold2_y) * 100

            # Fit and evaluate the model on the second fold
            lda_.fit(fold2_X, fold2_y)
            fold_2_accuracy = lda_.score(fold1_X, fold1_y) * 100

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


def fisher_criterion_2fold(X, y, k=30):

    print("------------------------Fisher's Criterion-----------------------------")

    pos_len = len(X[y == 1])
    neg_len = len(X[y == 0])
    data_len = (pos_len + neg_len)

    # find the mean vector for each class
    mean_vectors = []
    for i in np.unique(y):
        mean_vectors.append(np.mean(X[y == i], axis=0))

    mean = np.mean(mean_vectors, axis=0)

    # find the within-class matrix (Sw)
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for i in np.unique(y):
        class_samples = X[y == i]
        for j in range(len(class_samples)):
            diff = class_samples[j, :] - mean_vectors[i]
            Sw += np.outer(diff, diff)

    # find the Between-class matrix (Sb)
    Sb = np.zeros((X.shape[1], X.shape[1]))
    for i in np.unique(y):
        Sb += len(X[y == i]) * (mean_vectors[i] - mean) @ (mean_vectors[i] - mean).T

        # calculate Fisher's scores
        F_scores = np.diag(np.linalg.inv(Sw) @ Sb)

        # sort indices based on F-scores in descending order
        sorted_indices = np.argsort(F_scores)[::-1]

        # get top k indices and their corresponding F-scores
        top_k_indices = sorted_indices[:k]
        top_k_scores = F_scores[top_k_indices]

        # print top k indices and their corresponding F-scores
        # print(f"Top {k} indices: {top_k_indices}")

    # Initialize variables to store information about the best accuracy
    best_accuracy = 0
    best_selected_features = None

    # Loop through the top k features
    for i in range(1, min(k, len(top_k_indices)) + 1):
        selected_features = X[:, top_k_indices[:i]]

        # Split the data into two folds for each iteration
        fold1_X, fold2_X, fold1_y, fold2_y = train_test_split(selected_features, y, test_size=0.5, random_state=0)

        # Initialize LDA
        lda_ = LDA()

        # Fit and evaluate the model on the first fold
        lda_.fit(fold1_X, fold1_y)
        fold_1_accuracy = lda_.score(fold2_X, fold2_y) * 100

        # Fit and evaluate the model on the second fold
        lda_.fit(fold2_X, fold2_y)
        fold_2_accuracy = lda_.score(fold1_X, fold1_y) * 100

        accuracy = (fold_1_accuracy + fold_2_accuracy) / 2

        print(f" Top {i} features, Selected Feature = {top_k_indices[i-1]},  Accuracy: {accuracy}")

        # Update the best accuracy and selected features if a new best is found
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_selected_features = selected_features

    # Print the best accuracy and corresponding features
    print(" [-] Best Accuracy:", best_accuracy)
    print(" [-] Selected Features for Best Accuracy:", top_k_indices[:best_selected_features.shape[1]])


    return top_k_indices, top_k_scores

        
    
def main():
    cancer = load_breast_cancer()
    x = cancer.data
    y = cancer.target

    sequential_forward_selection_2fold(x, y)
    fisher_criterion_2fold(x, y)


if __name__ == '__main__':
    main()
