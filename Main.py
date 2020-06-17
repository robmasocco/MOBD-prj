"""
    Author: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: Main.py
    Date created: 15/06/2020
    Description: 
    
"""
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import sklearn.model_selection as model_select

from DataEvaluation import evaluate_classifier
from DataPreparation import *
from Classifiers.SVM import svm_param_selection
from DataVisualization import *
from Ouliers import *
from Ouliers.KNNReplacerIQR import KNNReplacerIQR
from Ouliers.KNNReplacerZS import KNNReplacerZS

"""
The works.
"""


# Target column
target = 'CLASS'


def main():
    # Read dataset
    dataset_path = 'Dataset/training_set.csv'
    dataset = pd.read_csv(dataset_path)
    print('\nDataset shape:', dataset.shape)
    print(dataset.describe())
    print('\n', dataset.tail())

    # Separate features and target labels
    x = dataset.drop(target, axis=1)
    y = dataset[[target]]
    features_list = x.columns.values.tolist()

    # Analyze dataset classes proportions
    show_classes_proportions(y, 'Dataset classes proportions')

    # Split dataset in train set and test test
    train_x, test_x, train_y, test_y = model_select.train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    print('\nTraining set shape:', train_x.shape, train_y.shape)
    print('Test set shape:', test_x.shape, test_y.shape)

    # Analyze train set and test set classes proportions
    show_classes_proportions(train_y, 'Training set classes proportions')
    show_classes_proportions(test_y, 'Test set classes proportions')

    """
    # Missing values
    print('\nMissing values')
    print('Train nan: ', get_na_count(train_x))
    print('Test nan: ', get_na_count(test_x))
    # Mean
    # train_mean = train_x.mean()
    # train_x = train_x.fillna(train_mean)
    # test_x = test_x.fillna(train_mean)
    # KNN
    imputer = KNNImputer(n_neighbors=10)
    train_x = pd.DataFrame(imputer.fit_transform(train_x))
    test_x = pd.DataFrame(imputer.transform(test_x))
    if get_na_count(train_x) != 0 or get_na_count(test_x) != 0:
        print('Error: missing values')
        return -1
        
    # Outliers
    print('\nOutliers')
    show_boxplot_featrues(train_x, 'Test set features')
    # IQR
    replacer = KNNReplacerIQR(n_neighbors=10)
    train_x = pd.DataFrame(replacer.fit_transform(train_x))
    test_x = pd.DataFrame(replacer.transform(test_x))
    # Z Score
    # train_mean = train_x.mean()
    # train_std = train_x.std()
    # train_x.where(~(((train_x - train_mean) / train_std).abs() > 3), np.nan, inplace=True)
    # test_x.where(~(((test_x - train_mean) / train_std).abs() > 3), np.nan, inplace=True)
    print('Train outliers: ', get_na_count(train_x))
    print('Test outliers: ', get_na_count(test_x))
    train_x = pd.DataFrame(imputer.fit_transform(train_x))
    test_x = pd.DataFrame(imputer.transform(test_x))
    if get_na_count(train_x) != 0 or get_na_count(test_x) != 0:
        print('Error: outliers')
        return -1
    show_boxplot_featrues(train_x, 'Test set features')
    
    # Scaling
    print('\nScaling')
    scaler = prep.StandardScaler()
    train_x = pd.DataFrame(scaler.fit_transform(train_x))
    train_x.columns = features_list
    test_x = pd.DataFrame(scaler.transform(test_x))
    test_x.columns = features_list
    print(train_x.describe())
    show_boxplot_featrues(train_x, 'Test set features')
    """

    # Make pipeline
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

    # Set the parameters
    grid_pipeline_svc = {'imputer__n_neighbors': [2, 5, 10],
                         'replacer__n_neighbors': [2, 5, 10],
                         'classifier__C': [0.1, 1, 2.75, 5, 10],
                         'classifier__gamma': [0.01, 0.05, 0.1],
                         'classifier__class_weight': [None, 'balanced']
                         }

    gs_iqr = model_select.GridSearchCV(pipeline_iqr, param_grid=grid_pipeline_svc, scoring='f1_macro', cv=5, refit=True,
                                        n_jobs=-1)
    gs_zs = model_select.GridSearchCV(pipeline_zs, param_grid=grid_pipeline_svc, scoring='f1_macro', cv=5, refit=True,
                                        n_jobs=-1)

    # List of pipelines for ease of iteration
    grids = [gs_iqr, gs_zs]

    # Dictionary of pipelines and classifier types for ease of reference
    grid_dict = {1: 'IQR', 2: 'Z SCORE'}

    # Fit the grid search objects
    print('Performing model optimizations...')
    best_f1 = 0.0
    best_clf = 0
    best_gs = ''
    for idx, gs in enumerate(grids):
        print('\nEstimator: %s' % grid_dict[idx])
        # Fit grid search
        gs.fit(train_x, train_y[target])
        #
        print("\nGrid scores:\n")
        means = gs.cv_results_['mean_test_score']
        stds = gs.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gs.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("\nBest parameters:")
        print(gs.best_params_)
        # Track best (highest test accuracy) model
        f1_temp = evaluate_classifier(gs, test_x, test_y[target])
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_gs = gs
            best_clf = idx
    print('\nClassifier with best test set f1 macro: %s' % grid_dict[best_clf])

    # Preprocessing
    imputer = KNNImputer(n_neighbors=best_gs.best_params_['imputer__n_neighbors'])
    x = imputer.fit_transform(x)
    replacer = KNNReplacerIQR(n_neighbors=best_gs.best_params_['replacer__n_neighbors'])
    x = replacer.fit_transform(x)
    scaler = prep.StandardScaler()
    x = scaler.fit_transform(x)

    final_clf = SVC(kernel='rbf', C=best_gs.best_params_['classifier__C'], gamma=best_gs.best_params_['classifier__gamma'],
                   class_weight=best_gs.best_params_['classifier__class_weight'], decision_function_shape='ovo',
                   random_state=0, cache_size=3000)
    final_clf.fit(x, y[target])

    print("F1 score final:")
    evaluate_classifier(best_clf, x, y[target])


# Start the script.
if __name__ == '__main__':
    main()
