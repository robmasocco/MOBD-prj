"""
    Authors: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: DataEvaluation.py
    Date created: 15/06/2020
    Description: Routine to evaluate classifiers during grid search.
"""


import sklearn.metrics as metrics

from DataVisualization import show_confusion_matrix


def evaluate_classifier(classifier, data_x, data_y, matrix_title='', show=True):
    """Preprocesses test set and evaluates classifiers."""
    pred_y = classifier.predict(data_x)
    confusion_matrix = metrics.confusion_matrix(data_y, pred_y)
    f1_score = metrics.f1_score(data_y, pred_y, average='macro')
    if show:
        show_confusion_matrix(confusion_matrix, f1_score, matrix_title)
    return f1_score
