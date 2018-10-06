"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from plot import plot_boundary
from data import make_dataset1, make_dataset2

# (Question 2)

# Put your funtions here
# ...


if __name__ == "__main__":
    pass 
    n_table = [1,5,25,125,625,1200]
    data = make_dataset2(1500,2048)
    scores = {}
    with open('knn_score', 'w') as f:
        for n in n_table:

            # part 1
            estimator = KNeighborsClassifier(n_neighbors = n).fit(data[0],data[1])
            print("computing" + str(n))
            plot_boundary("knn_" + str(n),estimator,data[0],data[1])

            #part 2
            scores[n] = cross_val_score(estimator,data[0],data[1],cv = 10)
            f.write("data of " + str(n) + "\n")
            f.write(str(scores[n]) + "\n")
            f.write(" mean: ")
            f.write(str(np.mean(scores[n])) + "\n")
            f.write("var: ")
            f.write(str(np.var(scores[n]) ) + "\n")
