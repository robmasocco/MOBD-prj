"""
    Author: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: KNNReplacerIQR.py
    Date created: 17/06/2020
    Description:
"""


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


class KNNReplacerIQR(KNNImputer):

    def __init__(self, n_neighbors=5):
        super().__init__(n_neighbors=n_neighbors)
        self.lower_bound = None
        self.upper_bound = None
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, x, y=None):
        """Computes IQR bound and fits the imputer on the data."""
        x = pd.DataFrame(x)
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        self.imputer.fit(x.where(~((x < self.lower_bound) | (x > self.upper_bound)), np.nan))
        return self

    def transform(self, x, y=None):
        """Detects outliers and replaces them with the imputer."""
        x = pd.DataFrame(x)
        x.where(~((x < self.lower_bound) | (x > self.upper_bound)), np.nan, inplace=True)
        return self.imputer.transform(x)

