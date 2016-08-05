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
    print "Creating test/train data for TFIDF\n"
    print df.head()



    # return X_train, X_test, y_train, y_test
    print "Test/Train data for TF-IDF has been created!!!\n"


def prep_text_data_W2V(df):
    """
    Prep data for use in text analysis using Word2Vec
    """
    print "Creating test/train data for Word2Vec\n"




    # return X_train, X_test, y_train, y_test
    print "Test/Train data for Word2Vec has been created!!!\n"

def prep_non_text_data_RF(df):
    """
    Prep data for use in non-text analysis using OneVRest Classifier and Random Forest
    """
    print "Creating test/train data for OneVRest Classifier\n"



    # return X_train, X_test, y_train, y_test
    print "Test/Train data for OneVRest Classifier has been created!!!\n"

def prep_non_text_data_LR(df):
    """
    Prep data for use in non-text analysis using Logistic Regression
    """
    print "Creating test/train data for Logistic Regression\n"




    # return X_train, X_test, y_train, y_test
    print "Test/Train data for Logistic Regression has been created!!!\n"
