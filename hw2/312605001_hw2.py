# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def main():
    data = pd.read_csv('iris.txt', delim_whitespace=True, header=None, engine='python')
    data = data.rename(columns={
        0: "sepal_length",
        1: "sepal_width",
        2: "petal_length",
        3: "petal_width",
        4: "label"})

if __name__ == '__main__':
    main()

"""
@author: OU,TING-WEI @ M.S. in Robotics 
Machien Learning HW2 ( NYCU FALL-2023 )
"""