"""
    Author: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: SVM.py
    Date created: 17/06/2020
    Description:

"""

import sklearn.model_selection as model_select
from sklearn import svm
import numpy as np


def knn_param_selection(train_x, train_y, n_folds, metric, verbose=False):
    # Hyperparameters grid to search.
    param_grid_knn = [{'kernel': ['rbf'], 'C': [3], 'gamma': [0.05],
                         'class_weight': [None, 'balanced']}]

    clf = model_select.GridSearchCV(svm.SVC(decision_function_shape='ovo',
                                            cache_size=3000),
                                    param_grid_tenny, scoring=metric,
                                    cv=n_folds, refit=True, n_jobs=-1)
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
