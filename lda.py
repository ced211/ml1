"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from plot import plot_boundary
from data import make_dataset1, make_dataset2

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

		nb_samples = len(y)
		prop = np.zeros(2) # Binary output supposed (Marginal probability for both class)
		mean = np.zeros((2,len(X[0]))) # Mean attrubute values for both class

		i = 0
		for sample in y:
			if sample == 0.: # First class
				prop[0]+=1
				mean[0]+=X[i]
			else:			 # Second class
				prop[1]+=1
				mean[1]+=X[i]
			i+=1

		for i in range(2):
			mean[i]/=prop[i]
			prop[i]/=nb_samples

		cov = np.cov(X, None, False)

		# Statement formula
		self.f = lambda x, k : (2* math.pi * math.sqrt(np.linalg.det(cov)))**-1 * np.exp(-0.5 * np.dot( np.dot( (x - mean[k]), np.linalg.inv(cov)), (x - mean[k]) )) 
		self.prop = prop

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

			den = np.dot(self.prop, densities)
			if (densities[0]*self.prop[0]/den) > (densities[1]*self.prop[1]/den) : # Search of the Maximum
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
		(n_sample, n_features) = X.shape
		p = np.empty([n_sample,2])
		num = np.zeros(n_features)
		densities = np.zeros(n_features)

		i=0
		for sample in X:    
			
			for k in range(n_features):
				densities[k] = self.f(sample[k], k)
				num[k] = densities[k]*self.prop[k]

			den = np.dot(self.prop, densities)
			p[i] = num/den
			i+=1
			
		return p

if __name__ == "__main__":
	from data import make_dataset1, make_dataset2
	from plot import plot_boundary

	# 1st dataset
	train_set = make_dataset1(1200, 565354)
	test_set = make_dataset1(300, 156)
	lda = LinearDiscriminantAnalysis()
	lda.fit(train_set[0],train_set[1])
	plot_boundary('lda_trainDataset1', lda, train_set[0], train_set[1])
	plot_boundary('lda_testDataset1', lda, test_set[0], test_set[1])
	# p1 = lda.predict_proba(test_set[0])

	#2nd dataset
	train_set = make_dataset2(1200, 565354)
	test_set = make_dataset2(300, 156)
	lda = LinearDiscriminantAnalysis()
	lda.fit(train_set[0],train_set[1])
	plot_boundary('lda_trainDataset2', lda, train_set[0], train_set[1])
	plot_boundary('lda_testDataset2', lda, test_set[0], test_set[1])
	# p2 = lda.predict_proba(test_set[0])
