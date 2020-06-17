"""
    Author: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: Main.py
    Date created: 15/06/2020
    Description:

"""


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import sklearn.preprocessing as prep


# Count missing values
def get_na_count(df):
    na_mask = df.isna()
    return na_mask.sum().sum()


# Count missing values for each features.
def get_na_count_cols(df):
    na_mask = df.isna()
    return na_mask.sum(axis=0)


# Compute IQR bounds
def iqr_bounds(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    return q1 - (1.5*iqr), q3 + (1.5*iqr)


def data_preparation(dataset):
    # 0. Separate features and target labels
    x = dataset.drop('CLASS', axis=1)
    y = dataset[['CLASS']]
    features_list = x.columns.values.tolist()

    # 1. Fill missing values with KNN method
    imputer = KNNImputer(n_neighbors=10)
    dataset = pd.DataFrame(imputer.fit_transform(dataset))

    # 2. Detect outliers with IQR and replace with KNN method
    lower_bound, upper_bound = iqr_bounds(dataset)
    dataset.where(~((dataset < lower_bound) | (dataset > upper_bound)), np.nan, inplace=True)

    # 3. Scaling data with standard scaler
    scaler = prep.StandardScaler()
    dataset = pd.DataFrame(scaler.fit_transform(dataset))
    dataset.columns = features_list

    return x, y
