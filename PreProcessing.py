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


# Get boolean mask for outliers
def get_iqr_mask(df):
    # Compute iqr range
    lower_bound, upper_bound = iqr_bounds(df)
    return (df < lower_bound) | (df > upper_bound)


# Replace outliers with nan
def iqr_outliers_to_nan(df):
    outliers_mask = get_iqr_mask(df)
    return df.where(outliers_mask, np.nan)


# Get boolean mask for outliers
def get_zscore_mask(df):
    # Compute values on training set
    mean = df.mean()
    std = df.std()
    return ((df - mean) / std).abs() > 3


#
def scale_data(trainset, testset):
    std_scaler = prep.StandardScaler()
    std_scaler.fit(trainset)
    return pd.DataFrame(std_scaler.transform(trainset)), pd.DataFrame(std_scaler.transform(testset))
