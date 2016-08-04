#!/usr/bin/env python
"""
Build analytics base table (ABT) for project

Data cleaning and other preparation steps in functions below.
"""
import pandas as pd
import numpy as np

def create_df_no_text(df):
    """
    Take input df and fill all missing/NaN or non-sensical data with something
    that is useful in a model.  Also create labels column 'Company response to consumer', with 3 categories.
    INPUT - dataframe
    OUTPUT - 'df_no_text' dataframe for use in modeling non-text features
    """
    print "Creating df_no_text for non-text feature modeling"

    df['Product'].fillna('Not Provided', inplace=True)
    df['Sub-product'].fillna('Not Provided', inplace=True)
    df['Sub-issue'].fillna('Not Provided', inplace=True)
    df['Issue'].fillna('Not Provided', inplace=True)
    df['Consumer complaint narrative'].fillna('Not Provided', inplace=True)
    df['Company public response'].fillna('Not Provided', inplace=True)
    df['Company'].fillna('Not Provided', inplace=True)
    df['State'].fillna('Not Provided', inplace=True)
    df['ZIP code'].fillna('Not Provided', inplace=True)
    df['Tags'].fillna('Not Provided', inplace=True)
    df['Consumer consent provided?'].fillna('Not Provided', inplace=True)
    df['Submitted via'].fillna('Not Provided',inplace=True)
    df['Consumer disputed?'].fillna('Not Provided', inplace=True)

    column=['Product', 'Sub-product','Issue','Sub-issue', 'Company', 'Tags', 'State']
    for name in column:
        repl={}
        i=0
        for value in df[name].unique():
            repl[value] = i
            i+=1

        df[name] = df[name].apply(lambda x: repl[x])
        df_no_text[name] = df[name].astype('category')

    #  Create numerical values for 'Company response to consumer' and map to df
    cust_resp_dict ={'Closed':0,
                 'Untimely response':0,
                 'Closed with explanation':1,
                 'Closed with non-monetary relief':2,
                 'Closed with monetary relief':2}

    df_no_text['Company response to consumer'] = df['Company response to consumer'].apply(lambda x: cust_resp_dict[x])

    print "Successfully created df_no_text for non-text feature modeling!!!"

    return df_no_text


def create_df_text(df):
    """
    Take input df and get 'Consumer complaint narrative' (text features) for text
    modeling.  Also create labels column 'Company response to consumer', with 3
    categories.

    INPUT - dataframe
    OUTPUT - 'df_text' dataframe for use in modeling text features
    """

    print "Creating df_text for text feature modeling"

    df_text = pd.DataFrame()  # Create empty df to fill

    df_text['Consumer complaint narrative'] = df['Consumer complaint narrative']

    #  Create numerical values for 'Company response to consumer' and map to df
    cust_resp_dict ={'Closed':0,
                 'Untimely response':0,
                 'Closed with explanation':1,
                 'Closed with non-monetary relief':2,
                 'Closed with monetary relief':2}

    df_text['Company response to consumer'] = df['Company response to consumer'].apply(lambda x: cust_resp_dict[x])

    print "Successfully created df_text for non-text feature modeling!!!"

    return df_text

if __name__ == '__main__':
    df = pd.read_csv('../data/Consumer_Complaints_with_Consumer_Complaint_Narratives.csv')
    df_no_text = pd.DataFrame()
    create_df_no_text(df)
    # print create_df_no_text(df).head()
    create_df_text(df)
    # print create_df_text(df).head()
