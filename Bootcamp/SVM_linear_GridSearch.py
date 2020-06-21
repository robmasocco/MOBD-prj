"""
    Authors: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: SVM_rbf_GridSearch.py
    Date created: 15/06/2020
    Description: Grid searches for best preprocessing pipeline and classifier.
                 SVM with linear kernel.
"""

import numpy as np

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import sklearn.model_selection as model_select

from DataVisualization import *

from Outliers.KNNReplacerIQR import KNNReplacerIQR
from Outliers.KNNReplacerZS import KNNReplacerZS
from Outliers.MeanReplacerIQR import MeanReplacerIQR
from Outliers.MeanReplacerZS import MeanReplacerZS

# Output data column.
target = 'CLASS'


def get_svm_linear_grids():
    """Define pipelines and grids for processing data."""
    pipe_linear_knn_iqr = Pipeline([('imputer', KNNImputer()),
                                    ('replacer', KNNReplacerIQR()),
                                    ('scaler', StandardScaler()),
                                    ('classifier',
                                     SVC(kernel='linear',
                                         decision_function_shape='ovo',
                                         random_state=42,
                                         cache_size=3000))
                                    ])

    pipe_linear_knn_zs = Pipeline([('imputer', KNNImputer()),
                                   ('replacer', KNNReplacerZS()),
                                   ('scaler', StandardScaler()),
                                   ('classifier',
                                    SVC(kernel='linear',
                                        decision_function_shape='ovo',
                                        random_state=42,
                                        cache_size=3000))
                                   ])

    pipe_linear_mean_iqr = Pipeline([('imputer', SimpleImputer()),
                                     ('replacer', MeanReplacerIQR()),
                                     ('scaler', StandardScaler()),
                                     ('classifier',
                                      SVC(kernel='linear',
                                          decision_function_shape='ovo',
                                          random_state=42,
                                          cache_size=3000))
                                     ])

    pipe_linear_mean_zs = Pipeline([('imputer', SimpleImputer()),
                                    ('replacer', MeanReplacerZS()),
                                    ('scaler', StandardScaler()),
                                    ('classifier',
                                     SVC(kernel='linear',
                                         decision_function_shape='ovo',
                                         random_state=42,
                                         cache_size=3000))
                                    ])

    # Set the parameters grids.
    c_range_svc_log10 = 10. ** np.arange(-3, 3)

    grid_pipe_knn_lin = {'imputer__n_neighbors': [2, 5, 10],
                         'replacer__n_neighbors': [2, 5, 10],
                         'classifier__C': c_range_svc_log10,
                         'classifier__class_weight': [None, 'balanced']
                         }

    grid_pipe_mean_lin = {'classifier__C': c_range_svc_log10,
                          'classifier__class_weight': [None, 'balanced']
                          }

    # Define grid searches for each pipeline.
    gs_lin_knn_iqr = model_select.GridSearchCV(pipe_linear_knn_iqr,
                                               param_grid=grid_pipe_knn_lin,
                                               scoring='f1_macro',
                                               cv=5,
                                               refit=True,
                                               n_jobs=-1)

    gs_lin_knn_zs = model_select.GridSearchCV(pipe_linear_knn_zs,
                                              param_grid=grid_pipe_knn_lin,
                                              scoring='f1_macro',
                                              cv=5,
                                              refit=True,
                                              n_jobs=-1)

    gs_lin_mean_iqr = model_select.GridSearchCV(pipe_linear_mean_iqr,
                                                param_grid=grid_pipe_mean_lin,
                                                scoring='f1_macro',
                                                cv=5,
                                                refit=True,
                                                n_jobs=-1)

    gs_lin_mean_zs = model_select.GridSearchCV(pipe_linear_mean_zs,
                                               param_grid=grid_pipe_mean_lin,
                                               scoring='f1_macro',
                                               cv=5,
                                               refit=True,
                                               n_jobs=-1)

    # List of pipeline grids for ease of iteration.
    grids = [gs_lin_knn_iqr,
             gs_lin_knn_zs,
             gs_lin_mean_iqr,
             gs_lin_mean_zs]

    # Dictionary of pipelines and classifier types for ease of reference.
    grids_dict = {0: 'SVM-LINEAR_KNN-IQR',
                  1: 'SVM-LINEAR_KNN-ZS',
                  2: 'SVM-LINEAR_MEAN-IQR',
                  3: 'SVM-LINEAR_MEAN-ZS'}

    return grids, grids_dict
