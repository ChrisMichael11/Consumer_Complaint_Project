#!/usr/bin/env python
"""
Build analytics base table (ABT) for project

Data cleaning and other preparations steps in functions below.

fill_missing_data(df)
date_cleaning(df)
modify_categoricals(df)
create_numerical_features(df)

"""
import pandas as pd
import numpy as np
# from pyzipcode import ZipCodeDatabase

# def fill_missing_data(df):
#     """
#     Take input df and fill all missing/NaN or non-sensical data with something
#     that is useful
#     INPUT - dataframe
#     OUTPUT - dataframe
#     """
#     df['Product'].fillna('Not Provided', inplace=True)
#     df['Sub-product'].fillna('Not Provided', inplace=True)
#     df['Sub-issue'].fillna('Not Provided', inplace=True)
#     df['Issue'].fillna('Not Provided', inplace=True)
#     df['Consumer complaint narrative'].fillna('Not Provided', inplace=True)
#     df['Company public response'].fillna('Not Provided', inplace=True)
#     df['Company'].fillna('Not Provided', inplace=True)
#     # df['State'].fillna('Not Provided', inplace=True)
#     ## Modified by "find_state_by_zip" function
#     # df['ZIP code'].fillna('Not Provided', inplace=True)
#     ## Modified by "find_state_by_zip" function
#     df['Tags'].fillna('Not Provided', inplace=True)
#     df['Consumer consent provided?'].fillna('Not Provided', inplace=True)
#     df['Submitted via'].fillna('Not Provided',inplace=True)
#     df['Consumer disputed?'].fillna('Not Provided', inplace=True)
#
#     return df
#
# def date_cleaning(df):
#     """
#     Clean dates, transform to datetime
#     """
#     df['Recieved Year'] = df['Date received'].apply(lambda x: x.year)
#     df['Recieved Month'] = df['Date received'].apply(lambda x: x.month)
#     df['Recieved Day'] = df['Date received'].apply(lambda x: x.day)
#
#     df['Submitted Year'] = df['Date sent to company'].apply(lambda x: x.year)
#     df['Submitted Month'] = df['Date sent to company'].apply(lambda x: x.month)
#     df['Submitted Day'] = df['Date sent to company'].apply(lambda x: x.day)
#
#     return df
#
# def modify_categoricals(df):
#     """
#     Turn categorical variables into Yes/No, True/False, or 1/0 for input into models.
#     INPUT - dataframe
#     OUTPUT - dataframe
#     """
#
#     # MODIFY 'Consumer consent provided?' to be T/F
#     df['Consumer consent provided?'] = df['Consumer consent provided?'].apply(lambda x:
#         'Consent not provided' if x=='Other' or x=='Consent withdrawn'
#          or x=='Not Provided' else x)
#     replace_consent = {'Consent provided': True, "Consent not provided": False}
#     df['Consumer consent provided?'] = df['Consumer consent provided?'].apply(lambda x:
#         replace_consent[x])
#
#     # MODIFY 'Consumer disputed?' to be yes/no
#     df['Consumer disputed?'] = df['Consumer disputed?'].apply(lambda x: 'No' if x=='Not Provided' else x)
#
#     replace_Y_N_to_TF = {'Yes': True, 'No':False}
#
#     df['Consumer disputed?'] = df['Consumer disputed?'].apply(lambda x: replace_Y_N_to_TF[x])
#
#     # MODIFY 'Timely response?'
#     df['Timely response?'] = df['Timely response?'].apply(lambda x: replace[x])
#
#     return df
#
# def create_numerical_features(df):
#     """
#     Create numerical values for columns with many different values.
#     ['Product', 'Sub-product','Issue','Sub-issue','Tags', 'State']
#     Count unique items (see EDA for more info) and assign number
#     INPUT - dataframe
#     OUTPUT - dataframe
#     """
#     column=['Product', 'Sub-product','Issue','Sub-issue','Tags', 'State']
#     for name in column:
#         repl={}
#         i=0
#         for value in df[name].unique():
#             repl[value] = i
#             i+=1
#
#     df[name] = df[name].apply(lambda x: repl[x])
#     df_model[name] = df[name].astype('category')
#
#     return df
#
# def count_company_complaints(df):
#     """
#     Create count of complaints for each company, add column
#     """
#     count_company_complaints = df['Company'].value_counts()
#     df['Count of Company Complaints'] = df['Company'].apply(lambda x: count_company_complaints[x])
#
#     return df
#
# def find_state_by_zip(df):
#     zipcode = ZipCodeDatabase
#     for item in df[pd.isnull(df['State']) & pd.notnull(df['ZIP code'])].index:
#     try:
#         df['State'][i] = str(zip[df['ZIP code'][i]].state)
#     except:
#         continue
#
#     #  Fill in empties that can't be filled with pyzipcode
#     df['State'].fillna('Not provided', inplace=True)
#     df['ZIP code'].fillna('Not provided', inplace=True)
#
#     return df

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
