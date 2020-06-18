"""
    Authors: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: DataEvaluation.py
    Date created: 15/06/2020
    Description: Routine to evaluate classifiers during grid search.
"""


import sklearn.metrics as metrics

from DataVisualization import show_confusion_matrix


def evaluate_classifier(classifier, test_x, test_y):
    """Preprocesses test set and evaluates classifiers."""
    pred_y = classifier.predict(test_x)
    confusion_matrix = metrics.confusion_matrix(test_y, pred_y)
    f1_score = metrics.f1_score(test_y, pred_y, average='macro')
    show_confusion_matrix(confusion_matrix, f1_score)
    return f1_score
