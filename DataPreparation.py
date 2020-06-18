"""
    Authors: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: Main.py
    Date created: 15/06/2020
    Description: Auxiliary routines to generate descriptive data.
"""


import sys
import pandas as pd

from sklearn.decomposition import PCA

from DataVisualization import *


def get_na_count(df):
    """Count missing values."""
    na_mask = df.isna()
    return na_mask.sum().sum()


def data_preparation_info(train_x, feats_names, pipeline):
    """Prints information on the dataset."""
    # Missing values.
    print('\nMissing values')
    print('Train NaNs: ', get_na_count(train_x))

    # KNN-based NaN filling.
    imputer = pipeline.named_steps['imputer']
    train_x = pd.DataFrame(imputer.transform(train_x))
    if get_na_count(train_x) != 0:
        print('ERROR: Missing values filling failed.')
        sys.exit(1)

    # Outliers processing.
    print('\nOutliers')
    show_boxplot_features(train_x, 'Training set with outliers')
    replacer = pipeline.named_steps['replacer']
    train_x = pd.DataFrame(replacer.transform(train_x))
    if get_na_count(train_x) != 0:
        print('ERROR: Outliers processing failed.')
        sys.exit(1)
    show_boxplot_features(train_x, 'Training set with replaced outliers')

    # Scale the training set.
    print('\nScaling')
    scaler = pipeline.named_steps['scaler']
    train_x = pd.DataFrame(scaler.transform(train_x))
    print("Training set features properties after scaling:")
    print(train_x.describe())
    show_boxplot_features(train_x, 'Training set after scaling')

    # Feature selection information using PCA.
    pca = PCA(random_state=42)
    pca = pca.fit(train_x)
    show_histogram_features(pca.explained_variance_ratio_,
                            feats_names,
                            'Feature importance by variance ratios')
