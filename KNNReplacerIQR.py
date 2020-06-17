"""
    Author: Alessandro Tenaglia
    Project: MOBD-prj
    File: KNNReplacerIQR.py
    Date created: 17/06/2020
    Description: 
    
"""


import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.impute import KNNImputer


class KNNReplacerIQR(TransformerMixin):

    def __init__(self, n_neighbors=2):
        self.lower_bound = None
        self.upper_bound = None
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, x, y=None):
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        self.imputer.fit(x.where(~((x < self.lower_bound) | (x > self.upper_bound)), np.nan))
        return self

    def transform(self, x, y=None):
        x.where(~((x < self.lower_bound) | (x > self.upper_bound)), np.nan, inplace=True)
        return self.imputer.transform(x)
