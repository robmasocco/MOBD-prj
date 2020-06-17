"""
    Author: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: Main.py
    Date created: 15/06/2020
    Description: 
    
"""


import sklearn.metrics as metrics

from DataVisualization import show_confusion_matrix


# Evaluate classifier with F1 macro
def evaluate_classifier(classifier, test_x, test_y):
    pred_y = classifier.predict(test_x)
    confusion_matrix = metrics.confusion_matrix(test_y, pred_y)
    f1_score = metrics.f1_score(test_y, pred_y, average='macro')
    show_confusion_matrix(confusion_matrix, f1_score)
    print('F1 macro: %0.4f' % f1_score)
