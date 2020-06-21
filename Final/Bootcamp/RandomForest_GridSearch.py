"""
    Authors: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: SVM_rbf_GridSearch.py
    Date created: 15/06/2020
    Description: Grid searches for best preprocessing pipeline and classifier.
                 Random Forests.
"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.model_selection as model_select

from Outliers.KNNReplacerIQR import KNNReplacerIQR
from Outliers.KNNReplacerZS import KNNReplacerZS
from Outliers.MeanReplacerIQR import MeanReplacerIQR
from Outliers.MeanReplacerZS import MeanReplacerZS

# Output data column.
target = 'CLASS'


def get_rf_grids():
    """Define pipelines and grids for processing data."""
    pipe_rf_knn_iqr = Pipeline([('imputer', KNNImputer()),
                                ('replacer', KNNReplacerIQR()),
                                ('scaler', StandardScaler()),
                                ('classifier',
                                 RandomForestClassifier(random_state=42,
                                                        n_jobs=-1))
                                ])

    pipe_rf_knn_zs = Pipeline([('imputer', KNNImputer()),
                               ('replacer', KNNReplacerZS()),
                               ('scaler', StandardScaler()),
                               ('classifier',
                                RandomForestClassifier(random_state=42,
                                                       n_jobs=-1))
                               ])

    pipe_rf_mean_iqr = Pipeline([('imputer', SimpleImputer()),
                                 ('replacer', MeanReplacerIQR()),
                                 ('scaler', StandardScaler()),
                                 ('classifier',
                                  RandomForestClassifier(random_state=42,
                                                         n_jobs=-1))
                                 ])

    pipe_rf_mean_zs = Pipeline([('imputer', SimpleImputer()),
                                ('replacer', MeanReplacerZS()),
                                ('scaler', StandardScaler()),
                                ('classifier',
                                 RandomForestClassifier(random_state=42,
                                                        n_jobs=-1))
                                ])

    # Set the parameters grids.
    grid_pipe_knn_rf = {'imputer__n_neighbors': [2, 5],
                        'replacer__n_neighbors': [2, 5],
                        'classifier__max_depth': [10, 50, 100, None],
                        'classifier__min_samples_leaf': [1, 2, 4],
                        'classifier__min_samples_split': [2, 5, 10],
                        'classifier__n_estimators': [100, 500, 1000]
                        }

    grid_pipe_mean_rf = {'classifier__max_depth': [10, 50, 100, None],
                         'classifier__min_samples_leaf': [1, 2, 4],
                         'classifier__min_samples_split': [2, 5, 10],
                         'classifier__n_estimators': [100, 500, 1000]
                         }

    # Define grid searches for each pipeline.
    gs_rf_knn_iqr = model_select.GridSearchCV(pipe_rf_knn_iqr,
                                              param_grid=grid_pipe_knn_rf,
                                              scoring='f1_macro',
                                              cv=5,
                                              refit=True,
                                              n_jobs=-1)

    gs_rf_knn_zs = model_select.GridSearchCV(pipe_rf_knn_zs,
                                             param_grid=grid_pipe_knn_rf,
                                             scoring='f1_macro',
                                             cv=5,
                                             refit=True,
                                             n_jobs=-1)

    gs_rf_mean_iqr = model_select.GridSearchCV(pipe_rf_mean_iqr,
                                               param_grid=grid_pipe_mean_rf,
                                               scoring='f1_macro',
                                               cv=5,
                                               refit=True,
                                               n_jobs=-1)

    gs_rf_mean_zs = model_select.GridSearchCV(pipe_rf_mean_zs,
                                              param_grid=grid_pipe_mean_rf,
                                              scoring='f1_macro',
                                              cv=5,
                                              refit=True,
                                              n_jobs=-1)

    # List of pipeline grids for ease of iteration.
    grids = [gs_rf_knn_iqr,
             gs_rf_knn_zs,
             gs_rf_mean_iqr,
             gs_rf_mean_zs]

    # Dictionary of pipelines and classifier types for ease of reference.
    grids_dict = {0: 'RAND-FOREST_KNN-IQR',
                  1: 'RAND-FOREST_KNN-ZS',
                  2: 'RAND-FOREST_MEAN-IQR',
                  3: 'RAND-FOREST_MEAN-ZS'}

    return grids, grids_dict
