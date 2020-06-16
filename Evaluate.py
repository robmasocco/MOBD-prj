"""
    Author: Alessandro Tenaglia
    Project: MOBD-prj
    File: Evaluate.py
    Date created: 15/06/2020
    Description: 
    
"""


import numpy as np
import sklearn.metrics as metrics
import seaborn as sns
import matplotlib.pyplot as plt


# Evaluate classifier with F1 macro
def evaluate_classifier(classifier, test_x, test_y):
    pred_y = classifier.predict(test_x)
    confusion_matrix = metrics.confusion_matrix(test_y, pred_y)
    f1_score = metrics.f1_score(test_y, pred_y, average='macro')
    show_confusion_matrix(confusion_matrix, f1_score)
    print('F1 macro: %0.4f' % f1_score)


# Show confusion matrix with annotations
def show_confusion_matrix(cm, f1_score):
    group_counts = ["{0:0.0f}\n".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cm.shape[0], cm.shape[1])
    sns.heatmap(cm, annot=box_labels, fmt="", cmap="YlGnBu", cbar=False, linewidths=1.0)\
        .set(title='Confusion matrix', xlabel='F1 macro: %0.4f' % f1_score)
    plt.show()
