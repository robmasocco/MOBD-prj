"""
    Authors: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: PipelineTraining.py
    Date created: 15/06/2020
    Description: Script to try pipelines.
"""

import pickle

import sklearn.model_selection as model_select

from Bootcamp.SVM_linear_GridSearch import get_svm_linear_grids
from Bootcamp.SVM_poly_GridSearch import get_svm_poly_grids
from Bootcamp.SVM_rbf_GridSearch import get_svm_rbf_grids
from Bootcamp.RandomForest_GridSearch import get_rf_grids
from Bootcamp.KNN_GridSearch import get_knn_grids

from DataEvaluation import evaluate_classifier

from DataPreparation import *

from DataVisualization import *

# Output data column.
target = 'CLASS'


def main():
    """Performs analysis and determines the best model for this problem."""
    # Read dataset.
    dataset_path = '../Dataset/training_set.csv'
    dataset = pd.read_csv(dataset_path)
    print("\nDATASET IMPORTED")
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

    # Define pipelines for processing data.
    # grids, grids_dict = get_svm_linear_grids()
    # grids, grids_dict = get_svm_poly_grids()
    grids, grids_dict = get_svm_rbf_grids()
    # grids, grids_dict = get_rf_grids()
    # grids, grids_dict = get_knn_grids()

    # Fit the grid search objects and look for the best model.
    print("\nMODEL OPTIMIZATIONS STARTED")
    best_f1 = 0.0
    best_idx = 0
    best_pipe = None
    for idx, pipe_gs in enumerate(grids):
        print('Currently trying model: %s' % grids_dict[idx])

        # Perform grid search.
        pipe_gs.fit(train_x, train_y[target])

        # Dump detailed scores on a file.
        results_file = open(grids_dict[idx] + '_results.txt', 'w')

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
          % grids_dict[best_idx])

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
