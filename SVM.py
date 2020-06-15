"""
    Author: Alessandro Tenaglia
    Project: MOBD-prj
    File: SVM.py
    Date created: 15/06/2020
    Description: 
    
"""

import sklearn.model_selection as model_select
from sklearn import svm


def svm_param_selection(train_x, train_y, n_folds, metric):

    # griglia degli iperparametri
    param_grid = [{'kernel': ['rbf'], 'C': [0.1, 0.5, 1, 5, 10, 100], 'gamma': [1, 0.1, 0.5]},
                  #{'kernel': ['linear'], 'C': [0.1, 1, 10]},
                  #{'kernel': ['linear'], 'C': [0.1, 1, 10]},
                  {'kernel': ['poly'], 'C': [0.1, 0.5, 1, 5, 10], 'degree': [2]}]

    clf = model_select.GridSearchCV(svm.SVC(class_weight='balanced',
                                            decision_function_shape='ovo',
                                            probability=True,
                                            cache_size=3000),
                                    param_grid, scoring=metric, cv=n_folds, refit=True, n_jobs=-1)
    clf.fit(train_x, train_y)

    print("Best parameters:\n")
    print(clf.best_params_)
    print("\nGrid scores:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf
