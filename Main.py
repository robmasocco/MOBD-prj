"""
    Authors: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: Main.py
    Date created: 15/06/2020
    Description: Grid searches for best preprocessing pipeline and classifier.
"""


import pickle

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import sklearn.model_selection as model_select

import numpy as np

from DataPreparation import *

from DataVisualization import *

from Outliers.KNNReplacerIQR import KNNReplacerIQR
from Outliers.KNNReplacerZS import KNNReplacerZS

from DataEvaluation import evaluate_classifier

# Output data column.
target = 'CLASS'


def main():
    """Performs analysis and determines the best model for this problem."""
    # Read dataset.
    dataset_path = 'Dataset/training_set.csv'
    dataset = pd.read_csv(dataset_path)
    print("DATASET IMPORTED")
    print('\nDataset shape:', dataset.shape)
    print(dataset.describe())
    print('\nLast dataset entries:', dataset.tail())

    # Separate features and target labels.
    x = dataset.drop(target, axis=1)
    y = dataset[[target]]

    # Split dataset in training set and test set.
    train_x, test_x, train_y, test_y =\
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

    # Define pipelines for preprocessing with SVMs (RBF kernel).
    pipeline_iqr = Pipeline([('imputer', KNNImputer()),
                             ('replacer', KNNReplacerIQR()),
                             ('scaler', StandardScaler()),
                             ('classifier', SVC(kernel='rbf',
                                                decision_function_shape='ovo',
                                                random_state=42,
                                                cache_size=3000))
                             ])

    pipeline_zs = Pipeline([('imputer', KNNImputer()),
                            ('replacer', KNNReplacerZS()),
                            ('scaler', StandardScaler()),
                            ('classifier', SVC(kernel='rbf',
                                               decision_function_shape='ovo',
                                               random_state=42,
                                               cache_size=3000))
                            ])

    # Define pipelines for preprocessing with SVMs (linear kernel). TODO

    # Define pipelines for preprocessing with SVMs (polynomial kernel). TODO

    # Define pipelines for preprocessing with Random Forests. TODO

    # Define pipelines for preprocessing with KNN. TODO

    # Set the parameters grids. TODO others too!
    # c_range_svc = [1, 1.5, 2, 2.5, 2.75, 3, 3.5, 5, 10]
    # gamma_range_svc = [0.03, 0.05, 0.07, 0.1, 0.5]
    # c_range_svc_log10 = 10. ** np.arange(-3, 3)
    # g_range_svc_log10 = 10. ** np.arange(-5, 4)
    # c_range_svc_log2 = 2. ** np.arange(-5, 5)
    # gamma_range_svc_log2 = 2. ** np.arange(-3, 3)
    c_range_svc = [2.75, 3]
    gamma_range_svc = [0.05]
    grid_pipeline_svc = {'imputer__n_neighbors': [2, 5, 10],
                         'replacer__n_neighbors': [2, 5, 10],
                         'classifier__C': c_range_svc,
                         'classifier__gamma': gamma_range_svc,
                         'classifier__class_weight': [None, 'balanced']
                         }

    # Define grid searches for each pipeline.
    pipe_gs_iqr = model_select.GridSearchCV(pipeline_iqr,
                                            param_grid=grid_pipeline_svc,
                                            scoring='f1_macro',
                                            cv=5,
                                            refit=True,
                                            n_jobs=-1)

    pipe_gs_zs = model_select.GridSearchCV(pipeline_zs,
                                           param_grid=grid_pipeline_svc,
                                           scoring='f1_macro',
                                           cv=5,
                                           refit=True,
                                           n_jobs=-1)

    # List of pipeline grids for ease of iteration.
    grids = [pipe_gs_iqr, pipe_gs_zs]

    # Dictionary of pipelines and classifier types for ease of reference.
    # TODO Now this must be extended.
    grid_dict_pipe = {0: 'IQR_SVM-RBF',
                      1: 'Z SCORE_SVM-RBF'}

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
        results_file = open(grid_dict_pipe[idx]+'_results.txt', 'w')

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
        if pipe_gs.best_score_ > best_f1:
            best_f1 = pipe_gs.best_score_
            best_idx = idx
            best_pipe = pipe_gs.best_estimator_

        results_file.close()

    print('\nPipeline with best training set F1 macro score: %s'
          % grid_dict_pipe[best_idx])

    # Show information and plots about best preprocessing pipeline.
    data_preparation_info(train_x, best_pipe)

    # Evaluates the pipeline on the test set.
    print('\nTest set F1 macro: %0.4f'
          % evaluate_classifier(best_pipe,
                                test_x,
                                test_y[target],
                                'Test Set Confusion matrix'))

    # Refit the best pipeline on the whole dataset.
    print("\nRE-FITTING BEST PIPELINE ON WHOLE DATASET")
    best_pipe = best_pipe.fit(x, y[target])
    print('\n(Post-save) Dataset F1 macro: %0.4f'
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
