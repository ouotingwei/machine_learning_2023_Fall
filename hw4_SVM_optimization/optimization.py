import svm
import numpy as np
import pandas as pd

def main():
    data = pd.read_csv('iris.txt', delim_whitespace=True, header=None, engine='python')
    data = data.rename(columns={
        0: "sepal_length",
        1: "sepal_width",
        2: "petal_length",
        3: "petal_width",
        4: "label"})
    
    positive_class = 2
    negative_class = 3
    selected_features = data[['petal_length', 'petal_width']]

    SVM.auto_execute(SVM, data, selected_features, positive_class, negative_class, 'linear', C=1)
    
if __name__ == "__main__":
    main()