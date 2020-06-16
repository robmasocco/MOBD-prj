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


def random_forest_param_selection(train_x, train_y, n_folds, metric, features_list):
    # griglia degli iperparametri
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [2, 3],
        'n_estimators': [range(130, 140, 5)]
    }
    # 2, 130, gini

    clf = model_select.GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid, scoring=metric, cv=n_folds, refit=True, n_jobs=-1)
    clf.fit(train_x, train_y)

    print("Best parameters:")
    print(clf.best_params_)
    print("\nGrid scores:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r\n" % (mean, std * 2, params))

    importances = list(clf.best_estimator_.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    return clf
