"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

from sklearn.base import BaseEstimator, ClassifierMixin



class LinearDiscriminantAnalysis(BaseEstimator, ClassifierMixin):


    def fit(self, X, y):
        """Fit a linear discriminant analysis model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        # ====================
        # TODO your code here.
        # ====================

        prop = [0, 0] # Binary output supposed
        mean = np.zeros(len(X[0]))

        for sample in y:
        	if sample == 0.0:
        		prop[0]+=1
        	else:
        		prop[1]+=1

        for sample in X:
        	i = 0
        	for feature in sample:
        		mean[i] += feature
        		i+=1

        mean /= len(y)

        cov = np.cov(X, None, False)

        # Statement formula
        self.f = lambda x, k : (2* math.pi * math.sqrt(np.linalg.det(cov)))**-1 * np.exp(-0.5 * (x - mean[k]) * np.linalg.inv(cov) * np.matrix.transpose(x - mean[k]) ) 
        self.mean = mean

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # ====================
        # TODO your code here.
        # ====================
        n_features = len(X[0])
        y = []
	    densities = np.zeros(n_features)

	    for sample in X:    
	        
	        for k in range(n_features):
	        	densities[k] = self.f(sample[k], k)

	        den = np.dot(self.mean, densities)
	        if (densities[0]*mean[0]/den) > (densities[1]*mean[1]/den) : # Search of the Maximum
	        	y.append(0.0)
	        else:
	        	y.append(1.0)

	    return y

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # ====================
        # TODO your code here.
        # ====================
        n_features = len(X[0])
        p = []
        num = []
		densities = np.zeros(n_features)

	    for sample in X:    
	        
	        for k in range(n_features):
	        	densities[k] = self.f(sample[k], k)
	        	num[k] = densities[k]*mean[k]

	        den = np.dot(self.mean, densities)
	        p.append( num/den )


        return p

if __name__ == "__main__":
    from data import make_data
    from plot import plot_boundary
