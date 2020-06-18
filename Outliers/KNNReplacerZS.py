"""
    Author: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: KNNReplacerZS.py
    Date created: 17/06/2020
    Description: Class to replace the outliers detected with Z-Score method through KNNImputer.
"""


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


class KNNReplacerZS(KNNImputer):

    def __init__(self, n_neighbors=5):
        super().__init__(n_neighbors=n_neighbors)
        self.mean = None
        self.std = None
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, x, y=None):
        """Computes Z-Score and fits the imputer on the data."""
        x = pd.DataFrame(x)
        self.mean = x.mean()
        self.std = x.std()
        self.imputer.fit(~(((x - self.mean) / self.std).abs() > 3), np.nan)
        return self

    def transform(self, x, y=None):
        """Detects outliers and replaces them with the imputer."""
        x = pd.DataFrame(x)
        x.where(~(((x - self.mean) / self.std).abs() > 3), np.nan, inplace=True)
        return self.imputer.transform(x)
