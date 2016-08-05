#!/usr/bin/env python
"""
Run this script with the following in the terminal:

python run_ComplaintModel.py <'filename'>

"""
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split

from data_prep import create_df_no_text
from data_prep import create_df_text
from model_prep import prep_text_data_TFIDF
from model_prep import prep_text_data_W2V
from model_prep import prep_non_text_data_RF
from model_prep import prep_non_text_data_LR


if __name__ == '__main__':
    df = pd.read_csv('../data/Consumer_Complaints_with_Consumer_Complaint_Narratives.csv')
    # create_df_no_text(df)
    # # print create_df_no_text(df).head()
    # create_df_text(df)
    # print create_df_text(df).head()

    prep_text_data_TFIDF(create_df_text(df))
    print prep_text_data_TFIDF(create_df_text(df))
    # prep_text_data_W2V(df_text)
    # print prep_text_data_W2V(df_text)
    # prep_non_text_data_LR(df_no_text)
    # print prep_non_text_data_LR(df_no_text)
    # prep_non_text_data_RF(df_no_text)
    # print prep_non_text_data_RF(df_no_text)
