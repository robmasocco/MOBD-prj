"""
    Authors: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: rbf_GridSearch.py
    Date created: 15/06/2020
    Description: Grid searches for best preprocessing pipeline and classifier.
                 Random Forests.
"""

import pickle

from sklearn.ensemble import RandomForestClassifier
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


def main():
    """Performs analysis and determines the best model for this problem."""
    # Read dataset.
    dataset_path = '../Dataset/training_set.csv'
    dataset = pd.read_csv(dataset_path)
    print("DATASET IMPORTED")
    print('\nDataset shape:', dataset.shape)
    print(dataset.describe())
    print('\nLast dataset entries:', dataset.tail())

    # Separate features and target labels.
    x = dataset.drop(target, axis=1)
    y = dataset[[target]]
    features_list = x.columns.values.tolist()

    # Split dataset in training set and test set.
    train_x, test_x, train_y, test_y = \
        model_select.train_test_split(x, y,
                                      test_size=0.2,
                                      random_state=42,
                                      stratify=y)
    print('\nTraining set shape:', train_x.shape, train_y.shape)
    print('Test set shape:', test_x.shape, test_y.shape)

    # Display data proportions after splitting.
    show_classes_proportions(y, 'Dataset classes proportions')
    show_classes_proportions(train_y, 'Training set classes proportions')
    show_classes_proportions(test_y, 'Test set classes proportions')

    # Define pipelines for preprocessing with Random Forests.
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
    grid_dict_pipe = {0: 'RAND-FOREST_KNN-IQR',
                      1: 'RAND-FOREST_KNN-ZS',
                      2: 'RAND-FOREST_MEAN-IQR',
                      3: 'RAND-FOREST_MEAN-ZS'}

    # Fit the grid search objects and look for the best model.
    print("\nMODEL OPTIMIZATIONS STARTED")
    best_f1 = 0.0
    best_idx = 0
    best_pipe = None
    for idx, pipe_gs in enumerate(grids):
        print('Currently trying model: %s' % grid_dict_pipe[idx])

        # Perform grid search.
        pipe_gs.fit(train_x, train_y[target])

        # Dump detailed scores on a file.
        results_file = open(grid_dict_pipe[idx] + '_results.txt', 'w')

        # Print scores and update bests.
        print("\nGrid scores:")
        means = pipe_gs.cv_results_['mean_test_score']
        stds = pipe_gs.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds,
                                     pipe_gs.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
            results_file.write("%0.4f (+/-%0.03f) for %r\n"
                               % (mean, std * 2, params))
        print("\nBest parameters:")
        print(pipe_gs.best_params_)
        print("\nBest score: %0.4f" % pipe_gs.best_score_)
        if pipe_gs.best_score_ > best_f1:
            best_f1 = pipe_gs.best_score_
            best_idx = idx
            best_pipe = pipe_gs.best_estimator_
        results_file.write("\nBest parameters:\n%r\n" % pipe_gs.best_params_)
        results_file.write("\nBest score: %0.4f\n" % pipe_gs.best_score_)

        results_file.close()

    print('\nPipeline with best training set F1 macro score: %s'
          % grid_dict_pipe[best_idx])

    # Show information and plots about best preprocessing pipeline.
    data_preparation_info(train_x, features_list, best_pipe)

    # Evaluates the pipeline on the test set.
    print('\nTest set F1 macro: %0.4f'
          % evaluate_classifier(best_pipe,
                                test_x,
                                test_y[target],
                                'Test Set Confusion matrix'))

    # Refit the best pipeline on the whole dataset.
    print("\nRE-FITTING BEST PIPELINE ON WHOLE DATASET")
    best_pipe = best_pipe.fit(x, y[target])
    print('\n(Pre-save) Dataset F1 macro: %0.4f'
          % evaluate_classifier(best_pipe,
                                x,
                                y[target],
                                'Dataset Confusion matrix'))

    # Serialize and dump the best model.
    pipeline_path = 'best_pipeline.sav'
    with open(pipeline_path, 'wb') as model_file:
        pickle.dump(best_pipe, model_file)

    # Reload best model and check if the save went well.
    with open(pipeline_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print('\n(Post-save) Dataset F1 macro: %0.4f'
          % evaluate_classifier(model,
                                x,
                                y[target],
                                show=False))


# Start the script.
if __name__ == '__main__':
    main()
