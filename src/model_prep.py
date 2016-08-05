#!/usr/bin/env python
"""
Prepare data from "data_prep.py" for use in models.  Perform train_test_split,
set up labels by binarizing.

There is a model_prep function for each type of model being run, described in docstrings below.
"""
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

def prep_text_data_TFIDF(df):
    """
    Prep data for use in text analysis using TF-IDF
    """




    # return X_train, X_test, y_train, y_test
    print "AWESOME"


def prep_text_data_W2V(df):
    """
    Prep data for use in text analysis using Word2Vec
    """




    # return X_train, X_test, y_train, y_test
    print "RADICAL"

def prep_non_text_data_RF(df):
    """
    Prep data for use in non-text analysis using OneVRest Classifier and Random Forest
    """



    # return X_train, X_test, y_train, y_test
    print "BODACIOUS"

def prep_non_text_data_LR(df):
    """
    Prep data for use in non-text analysis using Logistic Regression
    """



    # return X_train, X_test, y_train, y_test
    print "TUBULAR"
