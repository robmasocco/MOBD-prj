"""
    Authors: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: SVM_rbf_GridSearch.py
    Date created: 15/06/2020
    Description: Grid searches for best preprocessing pipeline and classifier.
                 K Nearest Neighbors.
"""

import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.model_selection as model_select

import numpy as np

from DataPreparation import *

from DataVisualization import *

from Outliers.KNNReplacerIQR import KNNReplacerIQR
from Outliers.KNNReplacerZS import KNNReplacerZS
from Outliers.MeanReplacerIQR import MeanReplacerIQR
from Outliers.MeanReplacerZS import MeanReplacerZS

from DataEvaluation import evaluate_classifier

# Output data column.
target = 'CLASS'


def get_knn_grids():
    """Define pipelines and grids for processing data."""
    pipe_knn_knn_iqr = Pipeline([('imputer', KNNImputer()),
                                 ('replacer', KNNReplacerIQR()),
                                 ('scaler', StandardScaler()),
                                 ('classifier', KNeighborsClassifier(n_jobs=-1))
                                 ])

    pipe_knn_knn_zs = Pipeline([('imputer', KNNImputer()),
                                ('replacer', KNNReplacerZS()),
                                ('scaler', StandardScaler()),
                                ('classifier', KNeighborsClassifier(n_jobs=-1))
                                ])

    pipe_knn_mean_iqr = Pipeline([('imputer', SimpleImputer()),
                                  ('replacer', MeanReplacerIQR()),
                                  ('scaler', StandardScaler()),
                                  ('classifier',
                                   KNeighborsClassifier(n_jobs=-1))
                                  ])

    pipe_knn_mean_zs = Pipeline([('imputer', SimpleImputer()),
                                 ('replacer', MeanReplacerZS()),
                                 ('scaler', StandardScaler()),
                                 ('classifier', KNeighborsClassifier(n_jobs=-1))
                                 ])

    # Set the parameters grids.
    grid_pipe_knn_knn = {'imputer__n_neighbors': [2, 5, 10],
                         'replacer__n_neighbors': [2, 5, 10],
                         'classifier__n_neighbors': [2, 5, 10],
                         'classifier__weights': ['uniform', 'distance'],
                         'classifier__p': [1, 2]
                         }

    grid_pipe_mean_knn = {'classifier__n_neighbors': [2, 5, 10],
                          'classifier__weights': ['uniform', 'distance'],
                          'classifier__p': [1, 2]
                          }

    # Define grid searches for each pipeline.
    gs_knn_knn_iqr = model_select.GridSearchCV(pipe_knn_knn_iqr,
                                               param_grid=grid_pipe_knn_knn,
                                               scoring='f1_macro',
                                               cv=5,
                                               refit=True,
                                               n_jobs=-1)

    gs_knn_knn_zs = model_select.GridSearchCV(pipe_knn_knn_zs,
                                              param_grid=grid_pipe_knn_knn,
                                              scoring='f1_macro',
                                              cv=5,
                                              refit=True,
                                              n_jobs=-1)

    gs_knn_mean_iqr = model_select.GridSearchCV(pipe_knn_mean_iqr,
                                                param_grid=grid_pipe_mean_knn,
                                                scoring='f1_macro',
                                                cv=5,
                                                refit=True,
                                                n_jobs=-1)

    gs_knn_mean_zs = model_select.GridSearchCV(pipe_knn_mean_zs,
                                               param_grid=grid_pipe_mean_knn,
                                               scoring='f1_macro',
                                               cv=5,
                                               refit=True,
                                               n_jobs=-1)

    # List of pipeline grids for ease of iteration.
    grids = [gs_knn_knn_iqr,
             gs_knn_knn_zs,
             gs_knn_mean_iqr,
             gs_knn_mean_zs]

    # Dictionary of pipelines and classifier types for ease of reference.
    grids_dict = {0: 'KNN_KNN-IQR',
                  1: 'KNN_KNN-ZS',
                  2: 'KNN_MEAN-IQR',
                  3: 'KNN_MEAN-ZS'}

    return grids, grids_dict
