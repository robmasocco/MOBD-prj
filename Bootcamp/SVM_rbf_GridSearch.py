"""
    Authors: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: SVM_rbf_GridSearch.py
    Date created: 15/06/2020
    Description: Grid searches for best preprocessing pipeline and classifier.
                 SVM with RBF kernel.
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


def get_svm_rbf_grids():
    """Define pipelines and grids for processing data."""
    pipe_rbf_knn_iqr = Pipeline([('imputer', KNNImputer()),
                                 ('replacer', KNNReplacerIQR()),
                                 ('scaler', StandardScaler()),
                                 ('classifier',
                                  SVC(kernel='rbf',
                                      decision_function_shape='ovo',
                                      random_state=42,
                                      cache_size=3000))
                                 ])

    pipe_rbf_knn_zs = Pipeline([('imputer', KNNImputer()),
                                ('replacer', KNNReplacerZS()),
                                ('scaler', StandardScaler()),
                                ('classifier', SVC(kernel='rbf',
                                                   decision_function_shape='ovo',
                                                   random_state=42,
                                                   cache_size=3000))
                                ])

    pipe_rbf_mean_iqr = Pipeline([('imputer', SimpleImputer()),
                                  ('replacer', MeanReplacerIQR()),
                                  ('scaler', StandardScaler()),
                                  ('classifier',
                                   SVC(kernel='rbf',
                                       decision_function_shape='ovo',
                                       random_state=42,
                                       cache_size=3000))
                                  ])

    pipe_rbf_mean_zs = Pipeline([('imputer', SimpleImputer()),
                                 ('replacer', MeanReplacerZS()),
                                 ('scaler', StandardScaler()),
                                 ('classifier',
                                  SVC(kernel='rbf',
                                      decision_function_shape='ovo',
                                      random_state=42,
                                      cache_size=3000))
                                 ])

    # Set the parameters grids.
    c_range_svc = [1, 1.5, 2, 2.5, 2.75, 3, 3.5, 5, 10]
    gamma_range_svc = [0.03, 0.05, 0.07, 0.1, 0.5]
    c_range_svc_log10 = 10. ** np.arange(-3, 3)
    gamma_range_svc_log10 = 10. ** np.arange(-5, 4)
    c_range_svc_log2 = 2. ** np.arange(-5, 5)
    gamma_range_svc_log2 = 2. ** np.arange(-3, 3)

    grid_pipe_knn_rbf = {'imputer__n_neighbors': [2],
                         'replacer__n_neighbors': [2],
                         'classifier__C': [50],
                         'classifier__gamma': [0.01],
                         'classifier__class_weight': [None]
                         }

    grid_pipe_mean_rbf = {'classifier__C': c_range_svc_log2,
                          'classifier__gamma': gamma_range_svc_log2,
                          'classifier__class_weight': [None, 'balanced']
                          }

    # Define grid searches for each pipeline.
    gs_rbf_knn_iqr = model_select.GridSearchCV(pipe_rbf_knn_iqr,
                                               param_grid=grid_pipe_knn_rbf,
                                               scoring='f1_macro',
                                               cv=5,
                                               refit=True,
                                               n_jobs=-1)

    gs_rbf_knn_zs = model_select.GridSearchCV(pipe_rbf_knn_zs,
                                              param_grid=grid_pipe_knn_rbf,
                                              scoring='f1_macro',
                                              cv=5,
                                              refit=True,
                                              n_jobs=-1)

    gs_rbf_mean_iqr = model_select.GridSearchCV(pipe_rbf_mean_iqr,
                                                param_grid=grid_pipe_mean_rbf,
                                                scoring='f1_macro',
                                                cv=5,
                                                refit=True,
                                                n_jobs=-1)

    gs_rbf_mean_zs = model_select.GridSearchCV(pipe_rbf_mean_zs,
                                               param_grid=grid_pipe_mean_rbf,
                                               scoring='f1_macro',
                                               cv=5,
                                               refit=True,
                                               n_jobs=-1)

    # List of pipeline grids for ease of iteration.
    grids = [gs_rbf_knn_iqr,
             gs_rbf_knn_zs,
             gs_rbf_mean_iqr,
             gs_rbf_mean_zs]
    grids = [gs_rbf_knn_iqr]

    # Dictionary of pipelines and classifier types for ease of reference.
    grids_dict = {0: 'SVM-RBF_KNN-IQR',
                  1: 'SVM-RBF_KNN-ZS',
                  2: 'SVM-RBF_MEAN-IQR',
                  3: 'SVM-RBF_MEAN-ZS'}

    return grids, grids_dict
