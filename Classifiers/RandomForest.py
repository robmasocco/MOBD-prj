"""
    Author: Alessandro Tenaglia
    Project: MOBD-prj
    File: RandomForest.py
    Date created: 16/06/2020
    Description: 
    
"""


import sklearn.model_selection as model_select
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np


def random_forest_param_selection(train_x, train_y, n_folds, metric):
    # griglia degli iperparametri
    param_grid = {
        'max_depth': [80, 90],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4],
        'n_estimators': [100, 200, 300]
    }

    clf = model_select.GridSearchCV(RandomForestClassifier(), param_grid, scoring=metric, cv=n_folds, refit=True)
    clf.fit(train_x, train_y)

    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf
