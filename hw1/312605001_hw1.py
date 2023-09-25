# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance


def scatter_plot(data):
    label_color_map = {1: ('blue', 'o'), 2: ('green', 'x'), 3: ('red', '^')}

    # plot 1. sepal_length v.s sepal_width
    x1 = data['sepal_length']
    y1 = data['sepal_width']
    for i in range(len(data)):
        label = data['label'][i]
        color, marker = label_color_map[label]
        plt.scatter(x1[i], y1[i], color=color, marker=marker, label=f'Label {label}')

    plt.title('1. sepal_width v.s sepal_length')
    plt.xlabel('sepal_width')
    plt.ylabel('sepal_length')
    plt.show()

    # plot 2. petal_width v.s. petal_length
    x2 = data['petal_width']
    y2 = data['petal_length']
    for i in range(len(data)):
        label = data['label'][i]
        color, marker = label_color_map[label]
        plt.scatter(x2[i], y2[i], color=color, marker=marker, label=f'Label {label}')

    plt.title('2. petal_width v.s. petal_length')
    plt.xlabel('petal_width')
    plt.ylabel('petal_length')
    plt.show()

    # plot 3. sepal_length v.s. petal_width
    x3 = data['sepal_length']
    y3 = data['petal_width']
    for i in range(len(data)):
        label = data['label'][i]
        color, marker = label_color_map[label]
        plt.scatter(x3[i], y3[i], color=color, marker=marker, label=f'Label {label}')

    plt.title('3. petal_width v.s sepal_length')
    plt.xlabel('sepal_length')
    plt.ylabel('petal_width')
    plt.show()

    # plot 4. sepal_width v.s. petal_length
    x4 = data['sepal_width']
    y4 = data['petal_length']
    for i in range(len(data)):
        label = data['label'][i]
        color, marker = label_color_map[label]
        plt.scatter(x4[i], y4[i], color=color, marker=marker, label=f'Label {label}')

    plt.title('4. sepal_width v.s. petal_length')
    plt.xlabel('sepal_width')
    plt.ylabel('petal_length')
    plt.show()

    # plot 5. sepal_length v.s. petal_length
    x4 = data['sepal_length']
    y4 = data['petal_length']
    for i in range(len(data)):
        label = data['label'][i]
        color, marker = label_color_map[label]
        plt.scatter(x4[i], y4[i], color=color, marker=marker, label=f'Label {label}')

    plt.title('5. sepal_length v.s. petal_length')
    plt.xlabel('sepal_length')
    plt.ylabel('petal_length')
    plt.show()

    # plot 6. sepal_width v.s. petal_width
    x4 = data['sepal_width']
    y4 = data['petal_width']
    for i in range(len(data)):
        label = data['label'][i]
        color, marker = label_color_map[label]
        plt.scatter(x4[i], y4[i], color=color, marker=marker, label=f'Label {label}')

    plt.title('6. sepal_width v.s. petal_width')
    plt.xlabel('sepal_width')
    plt.ylabel('petal_width')
    plt.show()


def knn_classifier(train_data, train_labels, test_data, k):
    predictions = []
    
    for test_point in test_data:
        distances = [np.sqrt(np.sum((train_point - test_point) ** 2)) for train_point in train_data]
        
        k_indices = np.argsort(distances)[:k]
        
        k_nearest_labels = [train_labels[i] for i in k_indices]
        
        prediction = np.bincount(k_nearest_labels).argmax()
        predictions.append(prediction)
    
    return predictions


def KNN(data, k=3):
    x = data[['sepal_length','sepal_width','petal_length','petal_width']]
    y = data['label']

    feature_combinations = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]]
    accuracy_list_1 = []
    accuracy_list_2 = []
    accuracy_avg = []

    for features in feature_combinations:
        selected_features = x.iloc[:, features] 
        
        x_train, x_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.5, random_state=87)
        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        predictions_1 = knn_classifier(x_train, y_train, x_test, k)
        prediction_2 = knn_classifier(x_test, y_test, x_train, k)

        accuracy_1 = 100*accuracy_score(y_test, predictions_1)
        accuracy_2 = 100*accuracy_score(y_train, prediction_2)

        accuracy_list_1.append(accuracy_1)
        accuracy_list_2.append(accuracy_2)
        accuracy_avg.append((accuracy_1+accuracy_2)/2)

    
    accuracy_format = ["{:.2f}".format(accuracy) for accuracy in accuracy_avg]

    CR_table = {"Combinations":['1. Sepal Length only', 
                          '2. Sepal Width only', 
                          '3. Petal Length only', 
                          '4. Petal Width only',
                          '5. Sepal Length + Sepal Width',
                          '6. Sepal Length + Petal Length',
                          '7. Sepal Length + Petal Width',
                          '8. Sepal Width + Petal Length',
                          '9. Sepal Width + Petal Width',
                          '10. petal Length + Petal Width',
                          '11. Sepal Length + Sepal Width + Petal Length',
                          '12. Sepal Length + Sepal Width + Petal Width',
                          '13. Sepal Length + Petal Length + Petal Width',
                          '14. sepal Width + Petal Length + Petal Width',
                          '15. Sepal Length + Sepal Width + Petal Length + Petal Width'],
                "Classification Rate": accuracy_format,}
    
    df = pd.DataFrame(CR_table)

    fig, ax = plt.subplots()

    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    plt.title('K = {}'.format(k))
    plt.show()
           

def main():
    data = pd.read_csv('iris.txt', delim_whitespace=True, header=None, engine='python')
    data = data.rename(columns={
        0: "sepal_length",
        1: "sepal_width",
        2: "petal_length",
        3: "petal_width",
        4: "label"})

    #print(data)
    #scatter_plot(data)
    KNN(data, k=3)


if __name__ == '__main__':
    main()

"""
@author: OU,TING-WEI @ M.S. in Robotics 
Machien Learning HW1 ( NYCU FALL-2023 )
"""