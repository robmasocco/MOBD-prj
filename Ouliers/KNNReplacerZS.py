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


class KNNReplacerZS(KNNImputer):

    def __init__(self, n_neighbors=2):
        super().__init__(n_neighbors=n_neighbors)
        self.mean = None
        self.std = None
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, x, y=None):
        x = pd.DataFrame(x)
        self.mean = x.mean()
        self.std = x.std()
        self.imputer.fit(~(((x - self.mean) / self.std).abs() > 3), np.nan)
        return self

    def transform(self, x, y=None):
        x = pd.DataFrame(x)
        x.where(~(((x - self.mean) / self.std).abs() > 3), np.nan, inplace=True)
        return self.imputer.transform(x)
