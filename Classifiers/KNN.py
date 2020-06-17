"""
    Author: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: Main.py
    Date created: 15/06/2020
    Description:

"""

import sklearn.model_selection as model_select
from sklearn.neighbors import KNeighborsClassifier


def knn_param_selection(train_x, train_y, n_folds, metric, verbose=False):
    # Hyperparameters grid to search.
    param_grid_knn = [{'n_neighbors': [5, 2, 10, 15],
                       'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                       'weights': ['uniform', 'distance'],
                       'p': [2, 1]}]

    # Search and cross-validate over the grid.
    clf = model_select.GridSearchCV(KNeighborsClassifier(
                                        n_jobs=-1),
                                    param_grid_knn,
                                    scoring=metric,
                                    cv=n_folds,
                                    refit=True,
                                    n_jobs=-1)
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
