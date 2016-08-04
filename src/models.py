#!/usr/bin/env python
"""
Models used in project
"""


def linear_regression(X_train, y_train):
    """
    INPUT: Pandas dataframe, Pandas series
    OUTPUT: model score
    """
    lr = LinearRegression()
	lr.fit(X_train, y_train)
	# cPickle.dump(lr, open('../models/lr.pkl', 'wb'))
	return lr
