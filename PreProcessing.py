"""
    Author: Alessandro Tenaglia
    Project: ProvaMOBD
    File: PreProcessing.py
    Date created: 13/06/2020
    Description: 
    
"""


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import sklearn.preprocessing as prep


# Count missing values
def get_na_count(df):
    na_mask = df.isna()
    return na_mask.sum().sum()


# Compute IQR bounds
def iqr_bounds(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    return q1 - (1.5*iqr), q3 + (1.5*iqr)
