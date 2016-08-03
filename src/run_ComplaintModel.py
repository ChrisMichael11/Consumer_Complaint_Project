#!/usr/bin/env python
"""
Run this script with the following in the terminal:

python run_ComplaintModel.py <'filename'>

"""
import numpy as np
import pandas as pd

from pyzipcode import ZipCodeDatabase

from data_prep import fill_missing_data
from data_prep import date_cleaning
from data_prep import modify_categoricals
from data_prep import create_numerical_features
from data_prep import count_company_complaints
from data_prep import find_state_by_zip




if __name__ == '__main__':
    main()
    fill_missing_data
    date_cleaning
    modify_categoricals
    create_numerical_features
    count_company_complaints
    find_state_by_zip
