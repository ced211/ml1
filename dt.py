"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset1, make_dataset2
from plot import plot_boundary
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    pass # Make your experiments here
    print("create sample")
    train_set = make_dataset2(1200,565354)
    test_set = make_dataset2(5000,156)

    #create tree and figure
    depth_test = [1,2,4,8]
    for depth in depth_test :
        print("create tree" + str(depth))
        estimator = DecisionTreeClassifier (max_depth = depth). fit(train_set[0],train_set[1])
        plot_boundary("test_tree_" + str(depth),estimator,test_set[0],test_set[1])
        plot_boundary("train_tree_" + str(depth),estimator,train_set[0],train_set[1]) 
        prediction = estimator.predict(test_set[0])
        score = accuracy_score(test_set[1],prediction)
        print("accuracy test of depth "+ str(depth) + " is " + str(score))

    estimator = DecisionTreeClassifier().fit(train_set[0],train_set[1])
    plot_boundary("inf_train_tree",estimator,train_set[0],train_set[1])
    plot_boundary("inf_test_tree",estimator,test_set[0],test_set[1])
    prediction = estimator.predict(test_set[0])
    score = accuracy_score(test_set[1],prediction)
    print("accuracy test of unconstrined depth is " + str(score))
