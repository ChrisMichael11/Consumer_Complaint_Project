#!/usr/bin/env python
"""
Build analytics base table (ABT) for project by filling missing data
"""
import pandas as pd
import numpy as np

def fill_missing_data(df):
    """
    Take input df and fill all missing/NaN or non-sensical data with something
    that is useful.
    INPUT - dataframe
    OUTPUT - dataframe
    """
    df['Product'].fillna('Not Provided', inplace=True)
    df['Sub-product'].fillna('Not Provided', inplace=True)
    df['Sub-issue'].fillna('Not Provided', inplace=True)
    df['Issue'].fillna('Not Provided', inplace=True)
    df['Consumer complaint narrative'].fillna('Not Provided', inplace=True)
    df['Company public response'].fillna('Not Provided', inplace=True)
    df['Company'].fillna('Not Provided', inplace=True)
    # df['State'].fillna('Not Provided', inplace=True)
    ## Modified by "zipcode_sorting.py"
    # df['ZIP code'].fillna('Not Provided', inplace=True)
    ## Modified by "zipcode_sorting.py"
    df['Tags'].fillna('Not Provided', inplace=True)
    df['Consumer consent provided?'].fillna('Not Provided', inplace=True)
    df['Submitted via'].fillna('Not Provided',inplace=True)
    df['Consumer disputed?'].fillna('Not Provided', inplace=True)

    ## 'Consumer consent provided?' has multiple responses:
    ## "O"
    df['Consumer consent provided?'] = df['Consumer consent provided?'].apply(lambda x:
        'Consent not provided' if x=='Other' or x=='Consent withdrawn'
         or x=='Not Provided' else x)

    return df
