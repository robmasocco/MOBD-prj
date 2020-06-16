"""
    Author: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD Classifier
    File: Main.py
    Date created: 15/06/2020
    Description: 
    
"""


import seaborn as sns
import sklearn.model_selection as model_select
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

from Classifiers.RandomForest import random_forest_param_selection
from Evaluate import evaluate_classifier
from PreProcessing import *
from Classifiers.SVM import svm_param_selection


"""
The works.
"""


# Target columns with classes
target = 'CLASS'


def main():
    # Read dataset
    dataset_path = './training_set.csv'
    dataset = pd.read_csv(dataset_path)
    print('\nDataset shape:', dataset.shape)
    print(dataset.describe())
    print('\n', dataset.tail())

    # Analyze dataset classes proportions
    pre_counts = dataset[target].value_counts()
    sns.countplot(x=target, data=dataset).set(title='Dataset classes proportions')
    plt.show()
    print('\nDataset classes proportions:')
    print(pre_counts)

    # Separate features and target labels
    x = dataset.drop(target, axis=1)
    y = dataset[[target]]
    features_list = x.columns.values.tolist()

    # Split dataset in train set and test test
    train_x, test_x, train_y, test_y = model_select.train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    print('\nTraining set shape:', train_x.shape, train_y.shape)
    print('Test set shape:', test_x.shape, test_y.shape)

    # Analyze dataset classes proportions
    post_counts = train_y[target].value_counts(normalize=True)
    sns.countplot(x=target, data=train_y).set(title='Training set classes proportions')
    plt.show()
    print('\nTraining set classes proportions:')
    print(post_counts)

    # Missing values
    print('\nMissing values')
    print('Train nan: ', get_na_count_cols(train_x))
    print('Test nan: ', get_na_count_cols(test_x))
    # train_mean = train_x.mean()
    # train_x = train_x.fillna(train_mean)
    # test_x = test_x.fillna(train_mean)
    imputer = KNNImputer()
    train_x = pd.DataFrame(imputer.fit_transform(train_x))
    test_x = pd.DataFrame(imputer.fit_transform(test_x))
    if get_na_count(train_x) != 0 or get_na_count(test_x) != 0:
        print('Error: missing values')
        return -1

    # Outliers
    print('\nOutliers')
    sns.boxplot(data=train_x)
    plt.show()
    train_lower, train_upper = iqr_bounds(train_x)
    train_x.where(~((train_x < train_lower) | (train_x > train_upper)), np.nan, inplace=True)
    test_x.where(~((test_x < train_lower) | (test_x > train_upper)), np.nan, inplace=True)
    train_mean = train_x.mean()
    train_std = train_x.std()
    # train_x.where(~(((train_x - train_mean) / train_std).abs() > 3), np.nan, inplace=True)
    # test_x.where(~(((test_x - train_mean) / train_std).abs() > 3), np.nan, inplace=True)
    print('Train outliers: ', get_na_count(train_x))
    print('Test outliers: ', get_na_count(test_x))
    train_x = pd.DataFrame(imputer.fit_transform(train_x))
    test_x = pd.DataFrame(imputer.fit_transform(test_x))
    if get_na_count(train_x) != 0 or get_na_count(test_x != 0):
        print('Error: outliers')
        return -1
    sns.boxplot(data=train_x)
    plt.show()

    # Scaling
    print('\nScaling')
    scaler = prep.StandardScaler()
    # scaler = prep.MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_x)
    train_x = pd.DataFrame(scaler.transform(train_x))
    test_x = pd.DataFrame(scaler.transform(test_x))
    print('\nTraining set shape:', train_x.shape)
    train_x.columns = features_list
    test_x.columns = features_list
    print(train_x.describe())
    sns.boxplot(data=train_x)
    plt.show()

    np_train_x = np.float64(train_x.values)
    np_train_y = np.float64(train_y.values)
    np_train_y = np_train_y.reshape((len(np_train_y), 1))
    np_test_x = np.float64(test_x.values)
    np_test_y = np.float64(test_y.values)
    np_test_y = np_test_y.reshape((len(np_test_y), 1))

    svm_classifier = svm_param_selection(train_x, train_y[target], n_folds=5, metric='f1_macro', verbose=True)
    # rf_classifier = random_forest_param_selection(train_x, train_y[target], n_folds=5, metric='f1_macro', features_list=features_list)

    print("SVM GRID SEARCH")
    evaluate_classifier(svm_classifier, test_x, test_y[target])

    # print("RANDOM FORESTS GRID SEARCH")
    # evaluate_classifier(rf_classifier, test_x, test_y[target])

    # Save cross-validation results locally if called from console.
    if __name__ != '__main__':
        return svm_classifier


# Start the script.
if __name__ == '__main__':
    main()
