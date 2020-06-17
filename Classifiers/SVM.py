"""
    Author: Alessandro Tenaglia
    Project: MOBD-prj
    File: SVM.py
    Date created: 15/06/2020
    Description: 
    
"""

import sklearn.model_selection as model_select
from sklearn import svm
import numpy as np


def svm_param_selection(train_x, train_y, n_folds, metric, verbose=False):

    # griglia degli iperparametri
    param_grid = [{'kernel': ['rbf'], 'C': [0.5, 1, 2, 2.55, 2.5, 3, 5], 'gamma': [0.1, 0.08, 0.075, 0.09]}]
                  # {'kernel': ['linear'], 'C': [0.1, 1, 10]},
                  # {'kernel': ['poly'], 'C': [10, 12, 15], 'degree': [2]}]
    param_grid_auto = [{'kernel': ['rbf'], 'C': np.arange(0.1, 5, 0.005), 'gamma': np.arange(0.1, 1, 0.005)},
                       {'kernel': ['linear'], 'C': np.arange(1, 10, 0.05)},
                       {'kernel': ['poly'], 'C': np.arange(1, 10, 0.05), 'degree': [2, 3]}]
    param_grid_poly = [{'kernel': ['linear'], 'C': np.arange(1, 10, 0.05)},
                       {'kernel': ['poly'], 'C': np.arange(1, 10, 0.05), 'degree': [2, 3]}]
    param_grid_rbf = [{'kernel': ['rbf'], 'C': np.arange(2.25, 2.75, 0.05), 'gamma': np.arange(0.01, 0.1, 0.05)}]

    param_grid_tenny = [{'kernel': ['rbf'], 'C': [3], 'gamma': [0.05], 'class_weight': [None, 'balanced']}]

    clf = model_select.GridSearchCV(svm.SVC(decision_function_shape='ovo',
                                            cache_size=3000),
                                    param_grid_tenny, scoring=metric, cv=n_folds, refit=True, n_jobs=-1)
    clf.fit(train_x, train_y)

    # Print best parameters.
    print("\nBest parameters:")
    print(clf.best_params_)

    # Print time required to refit on the whole training set.
    print("\nTime to refit on training set: %f second(s)." % clf.refit_time_)

    # Print grid search results (avoiding I/O buffers destruction).
    if verbose:
        print("\nGrid scores:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))

    return clf
