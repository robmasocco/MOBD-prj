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

from DataVisualization import *
from Outliers.KNNReplacerIQR import KNNReplacerIQR


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


def old_data_preparation(train_x, test_x):
    # Missing values
    print('\nMissing values')
    print('Train nan: ', get_na_count(train_x))
    print('Test nan: ', get_na_count(test_x))
    # Mean
    # train_mean = train_x.mean()
    # train_x = train_x.fillna(train_mean)
    # test_x = test_x.fillna(train_mean)
    # KNN
    imputer = KNNImputer(n_neighbors=10)
    train_x = pd.DataFrame(imputer.fit_transform(train_x))
    test_x = pd.DataFrame(imputer.transform(test_x))
    if get_na_count(train_x) != 0 or get_na_count(test_x) != 0:
        print('Error: missing values')
        return -1

    # Outliers
    print('\nOutliers')
    show_boxplot_features(train_x, 'Test set features')
    # IQR
    replacer = KNNReplacerIQR(n_neighbors=10)
    train_x = pd.DataFrame(replacer.fit_transform(train_x))
    test_x = pd.DataFrame(replacer.transform(test_x))
    # Z Score
    # train_mean = train_x.mean()
    # train_std = train_x.std()
    # train_x.where(~(((train_x - train_mean) / train_std).abs() > 3), np.nan, inplace=True)
    # test_x.where(~(((test_x - train_mean) / train_std).abs() > 3), np.nan, inplace=True)
    print('Train outliers: ', get_na_count(train_x))
    print('Test outliers: ', get_na_count(test_x))
    train_x = pd.DataFrame(imputer.fit_transform(train_x))
    test_x = pd.DataFrame(imputer.transform(test_x))
    if get_na_count(train_x) != 0 or get_na_count(test_x) != 0:
        print('Error: outliers')
        return -1
    show_boxplot_features(train_x, 'Test set features')

    # Scaling
    print('\nScaling')
    scaler = prep.StandardScaler()
    train_x = pd.DataFrame(scaler.fit_transform(train_x))
    test_x = pd.DataFrame(scaler.transform(test_x))
    print(train_x.describe())
    show_boxplot_features(train_x, 'Test set features')

    return train_x, test_x
