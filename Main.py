"""
    Author: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: Main.py
    Date created: 15/06/2020
    Description: Grid searches for best preprocessing pipeline and classifier.
"""

# TODO Which of these are really needed?
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import sklearn.model_selection as model_select

from DataEvaluation import evaluate_classifier
from DataPreparation import *
from Classifiers.SVM import svm_param_selection
from DataVisualization import *
from Outliers.KNNReplacerIQR import KNNReplacerIQR
from Outliers.KNNReplacerZS import KNNReplacerZS

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
    features_list = x.columns.values.tolist()  # TODO What for?

    # Split dataset in training set and test set.
    train_x, test_x, train_y, test_y = model_select.train_test_split(x, y,
                                                                     test_size=0.2,
                                                                     random_state=0,
                                                                     stratify=y)
    print('\nTraining set shape:', train_x.shape, train_y.shape)
    print('Test set shape:', test_x.shape, test_y.shape)

    # Display the data.
    show_classes_proportions(y, 'Dataset classes proportions')
    show_boxplot_features(train_x, 'Training set features boxplot')
    show_classes_proportions(train_y, 'Training set classes proportions')
    show_classes_proportions(test_y, 'Test set classes proportions')

    # TODO Redo other plots!

    # Define pipelines for preprocessing with SVC.
    pipeline_iqr = Pipeline([('imputer', KNNImputer()),
                             ('replacer', KNNReplacerIQR()),
                             ('scaler', prep.StandardScaler()),
                             ('classifier', SVC(kernel='rbf',
                                                decision_function_shape='ovo',
                                                random_state=0,
                                                cache_size=3000))
                             ])

    pipeline_zs = Pipeline([('imputer', KNNImputer()),
                            ('replacer', KNNReplacerZS()),
                            ('scaler', prep.StandardScaler()),
                            ('classifier', SVC(kernel='rbf',
                                               decision_function_shape='ovo',
                                               random_state=0,
                                               cache_size=3000))
                            ])

    # Define pipelines for preprocessing with Random Forests. TODO

    # Define pipelines for preprocessing with KNN. TODO

    # Set the parameters grids. TODO others too!
    c_range_svc = [1, 1.5, 2, 2.5, 2.75, 3, 3.5, 5, 10]
    gamma_range_svc = [0.03, 0.05, 0.07, 0.1, 0.5]
    grid_pipeline_svc = {'imputer__n_neighbors': [2, 5, 10],
                         'replacer__n_neighbors': [2, 5, 10],
                         'classifier__C': c_range_svc,
                         'classifier__gamma': gamma_range_svc,
                         'classifier__class_weight': [None, 'balanced']
                         }

    # Define grid searches for each pipeline.
    gs_iqr = model_select.GridSearchCV(pipeline_iqr,
                                       param_grid=grid_pipeline_svc,
                                       scoring='f1_macro',
                                       cv=5,
                                       refit=True,
                                       n_jobs=-1)

    gs_zs = model_select.GridSearchCV(pipeline_zs,
                                      param_grid=grid_pipeline_svc,
                                      scoring='f1_macro',
                                      cv=5,
                                      refit=True,
                                      n_jobs=-1)

    # List of pipeline grids for ease of iteration.
    grids = [gs_iqr, gs_zs]

    # Dictionary of pipelines and classifier types for ease of reference.
    # TODO Now this must be extended.
    grid_dict = {0: 'IQR', 1: 'Z SCORE'}

    # TODO A CSV report for each grid search must be generated here, too much data all at once.
    # Fit the grid search objects and look for the best model.
    print("\nMODEL OPTIMIZATIONS STARTED")
    best_f1 = 0.0
    best_clf = 0
    best_gs = ''
    for idx, gs in enumerate(grids):
        print('Currently trying model: %s' % grid_dict[idx])

        # Perform grid search.
        gs.fit(train_x, train_y[target])

        # Print scores and update bests.
        print("\nGrid scores:")
        means = gs.cv_results_['mean_test_score']
        stds = gs.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gs.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("\nBest parameters:")
        print(gs.best_params_)
        # TODO Best on train set or test set??
        f1_temp = evaluate_classifier(gs, test_x, test_y[target])  # TODO MUST PREPROCESS ACCORDINGLY!
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_gs = gs
            best_clf = idx

    print('\nClassifier with best test set F1 macro: %s' % grid_dict[best_clf])

    # Evaluates the pipeline on the test set
    # TODO Simply use Pipeline score
    # Pipeline.score(test_x, test_y[target])

    # Preprocess the whole dataset using the best pipeline.
    # TODO Replace all with Pipeline.fit(x, y[target])
    print("\nRE-PREPROCESSING DATASET WITH BEST PIPELINE")
    imputer = KNNImputer(n_neighbors=best_gs.best_params_['imputer__n_neighbors'])
    x = imputer.fit_transform(x)
    replacer = KNNReplacerIQR(n_neighbors=best_gs.best_params_['replacer__n_neighbors'])
    x = replacer.fit_transform(x)
    scaler = prep.StandardScaler()
    x = scaler.fit_transform(x)
    # TODO Add plots now to show effects of the preprocessing pipeline!

    # Fit the best classifier on the whole dataset.
    # TODO Must set best model, could also not be an SVM! Very unlikely though.
    #  Use above _best_shit data.
    print("\nFITTING BEST CLASSIFIER")
    final_clf = SVC(kernel='rbf',
                    decision_function_shape='ovo',
                    random_state=0,
                    cache_size=3000,
                    C=best_gs.best_params_['classifier__C'],
                    gamma=best_gs.best_params_['classifier__gamma'],
                    class_weight=best_gs.best_params_['classifier__class_weight'],
                    )
    final_clf.fit(x, y[target])

    # TODO Pickle the pipeline fitted on the whole dataset
    # pipeline_path = 'best_pipeline.sav'
    # pickle.dump(model, open(pipeline_path, 'wb'))

    # Get an idea of the error by evaluating the model on the dataset.
    print("\nFINAL SCORE ON WHOLE DATASET")
    evaluate_classifier(final_clf, x, y[target])


# Start the script.
if __name__ == '__main__':
    main()
