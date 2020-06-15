"""
    Author: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD Classifier
    File: Main.py
    Date created: 15/06/2020
    Description: 
    
"""


import seaborn as sns
import sklearn.model_selection as model_select
import matplotlib.pyplot as plt

from PreProcessing import *


target = 'CLASS'

def main():
    # Read dataset
    dataset_path = './training_set.csv'
    dataset = pd.read_csv(dataset_path)
    print('\nDataset shape:', dataset.shape)
    print(dataset.describe())
    print('\n', dataset.tail())

    # Separate features and target labels
    x = dataset.drop(target, axis=1)
    y = dataset[[target]]
    features_list = x.columns.values.tolist()

    # Analyze dataset classes proportions
    pre_counts = y[target].value_counts(normalize=True)
    sns.countplot(x=target, data=dataset).set(title='Dataset classes proportions')
    plt.show()
    print('\nDataset classes proportions:')
    print(pre_counts)

    # Split dataset in train set and test test
    train_x, test_x, train_y, test_y = model_select.train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
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
    print('Train nan: ', get_na_count(train_x))
    print('Test nan: ', get_na_count(test_x))
    # train_x = train_x.dropna()
    # test_x = test_x.dropna()
    train_mean = train_x.mean()
    train_x = train_x.fillna(train_mean)
    test_x = test_x.fillna(train_mean)
    if get_na_count(train_x) != 0 or get_na_count(test_x) != 0:
        print('Error: missing values')
        return -1

    # Outliers
    print('\nOutliers')
    sns.boxplot(data=train_x)
    plt.show()
    train_mean = train_x.mean()
    train_std = train_x.std()
    train_lower, train_upper = iqr_bounds(train_x)
    train_x.where(~((train_x < train_lower) | (train_x > train_upper)), np.nan, inplace=True)
    test_x.where(~((test_x < train_lower) | (test_x > train_upper)), np.nan, inplace=True)
    # train_x.where(~(((train_x - train_mean) / train_std).abs() > 3), np.nan, inplace=True)
    # test_x.where(~(((test_x - train_mean) / train_std).abs() > 3), np.nan, inplace=True)
    print('Train outliers: ', get_na_count(train_x))
    print('Test outliers: ', get_na_count(test_x))
    train_x = train_x.fillna(train_mean)
    test_x = test_x.fillna(train_mean)
    if get_na_count(train_x) != 0 or get_na_count(test_x != 0):
        print('Error: outliers')
        return -1
    sns.boxplot(data=train_x)
    plt.show()

    # Scaling
    print('\nScaling')
    # scaler = prep.StandardScaler()
    scaler = prep.MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_x)
    train_x = pd.DataFrame(scaler.transform(train_x))
    test_x = pd.DataFrame(scaler.transform(test_x))
    print('\nTraining set shape:', train_x.shape)
    train_x.columns = features_list
    test_x.columns = features_list
    print(train_x.describe())
    sns.boxplot(data=train_x)
    plt.show()


if __name__ == '__main__':
    main()
