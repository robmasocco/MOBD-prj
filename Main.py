"""
    Author: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: Main.py
    Date created: 15/06/2020
    Description: 
    
"""


import seaborn as sns
import sklearn.model_selection as model_select
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

from Classifiers.RandomForest import random_forest_param_selection
from DataEvaluation import evaluate_classifier
from DataPreparation import *
from Classifiers.SVM import svm_param_selection
from Classifiers.KNN import knn_param_selection
from DataVisualization import *
from KNNReplacerIQR import KNNReplacerIQR

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

    """"# Make pipeline
    pipeline = Pipeline([('nan_filler', KNNImputer()),
                         ('outliners_replacer', ),
                         ('scaler', prep.StandardScaler()),
                         ('classifier', )
                         ])"""

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
    train_lower, train_upper = iqr_bounds(train_x)
    train_x.where(~((train_x < train_lower) | (train_x > train_upper)), np.nan, inplace=True)
    test_x.where(~((test_x < train_lower) | (test_x > train_upper)), np.nan, inplace=True)
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
    """replacer = KNNReplacerIQR(n_neighbors=10)
    train_x = pd.DataFrame(replacer.fit_transform(train_x))
    test_x = pd.DataFrame(replacer.transform(test_x))"""
    show_boxplot_featrues(train_x, 'Test set features')

    # Scaling
    print('\nScaling')
    scaler = prep.StandardScaler()
    # scaler = prep.MinMaxScaler(feature_range=(-1, 1))
    train_x = pd.DataFrame(scaler.fit_transform(train_x))
    train_x.columns = features_list
    test_x = pd.DataFrame(scaler.transform(test_x))
    test_x.columns = features_list
    print(train_x.describe())
    show_boxplot_featrues(train_x, 'Test set features')

    # Resampling
    # train_x, train_y = OneSidedSelection(sampling_strategy='majority', random_state=0).fit_resample(train_x, train_y[target])
    # train_x, train_y = SVMSMOTE(sampling_strategy='all', random_state=0, n_jobs=-1).fit_resample(train_x, train_y[target])
    # train_x, train_y = TomekLinks(sampling_strategy='majority', n_jobs=-1).fit_resample(train_x, train_y[target])
    # train_y = pd.DataFrame(train_y)
    # train_y.columns = [target]

    # Features selection
    # pca = PCA(n_components=17)
    # train_x = pca.fit_transform(train_x)
    # test_x = pca.transform(test_x)
    # print(pca.explained_variance_ratio_)

    svm_classifier = svm_param_selection(train_x, train_y[target], n_folds=5, metric='f1_macro', verbose=True)
    # rf_classifier = random_forest_param_selection(train_x, train_y[target], n_folds=5, metric='f1_macro', features_list=features_list)
    # knn_classifier = knn_param_selection(train_x, train_y[target], 5, 'f1_macro', True)

    print("\nSVM GRID SEARCH")
    evaluate_classifier(svm_classifier, test_x, test_y[target])

    # print("\nKNN GRID SEARCH")
    # evaluate_classifier(knn_classifier, test_x, test_y[target])

    # print("RANDOM FORESTS GRID SEARCH")
    # evaluate_classifier(svm_classifier, test_x, test_y[target])

    # Save cross-validation results locally if called from console.
    # if __name__ != '__main__':
    #   return rf_classifier


# Start the script.
if __name__ == '__main__':
    main()
