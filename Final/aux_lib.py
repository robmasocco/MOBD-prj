"""
    Authors: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: evaluation.py
    Date created: 19/06/2020
    Description: Auxiliary library for evaluation script.
"""

import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.impute import KNNImputer


class KNNReplacerIQR(KNNImputer):
    """Pipeline-compliant KNNReplacer, based on IQR."""

    def __init__(self, n_neighbors=5):
        super().__init__(n_neighbors=n_neighbors)
        self.lower_bound = None
        self.upper_bound = None
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, x, y=None):
        """Computes IQR bound and fits the imputer on the data."""
        x = pd.DataFrame(x)
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        self.imputer.fit(
            x.where(~((x < self.lower_bound) | (x > self.upper_bound)), np.nan)
        )
        return self

    def transform(self, x, y=None):
        """Detects outliers and replaces them with the imputer."""
        x = pd.DataFrame(x)
        x.where(~((x < self.lower_bound) | (x > self.upper_bound)),
                np.nan,
                inplace=True)
        return self.imputer.transform(x)


def show_confusion_matrix(cm, f1_score, title):
    """Displays confusion matrix with annotations."""
    # Create annotations label.
    group_counts = ["{0:0.0f}\n".format(value) for value in cm.flatten()]
    group_percentages =\
        ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    box_labels =\
        [f"{v1}{v2}".strip() for v1, v2 in zip(group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cm.shape[0], cm.shape[1])
    # Show confusion matrix with heat map.
    sns.heatmap(cm,
                annot=box_labels,
                fmt="",
                cmap="YlGnBu",
                cbar=False,
                linewidths=1.0)\
        .set(title=title,
             xlabel='Predicted class\nF1 macro: %0.4f' % f1_score,
             ylabel='Actual class')
    plt.show()


def evaluate_classifier(classifier, data_x, data_y, matrix_title='', show=True):
    """Preprocesses test set and evaluates classifiers."""
    pred_y = classifier.predict(data_x)
    confusion_matrix = metrics.confusion_matrix(data_y, pred_y)
    f1_score = metrics.f1_score(data_y, pred_y, average='macro')
    print('\nTest set F1 macro score: %0.4f .\n' % f1_score)
    if show:
        show_confusion_matrix(confusion_matrix, f1_score, matrix_title)
    return f1_score
