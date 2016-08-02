#!/usr/bin/env python
"""
Sorting out missing state information using pyzipcode.  pyzipcode folder should
be in same folder where you are running from (https://pypi.python.org/pypi/pyzipcode)
INPUT - dataframe with columns "State" and "ZIP code"
OUTPUT - dataframe with update "State" and "ZIP code" columns
"""

import pandas as pd
from pyzipcode import ZipCodeDatabase

def find_state_by_zip(df):
    zipcode = ZipCodeDatabase
    for item in df[pd.isnull(df['State']) & pd.notnull(df['ZIP code'])].index:
    try:
        df['State'][i] = str(zip[df['ZIP code'][i]].state)
    except:
        continue

    #  Fill in empties that can't be filled with pyzipcode
    df['State'].fillna('Not provided', inplace=True)
    df['ZIP code'].fillna('Not provided', inplace=True)

    return df  
