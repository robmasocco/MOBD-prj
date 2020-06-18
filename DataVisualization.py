"""
    Author: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: DataVisualization.py
    Date created: 15/06/2020
    Description:
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Analyze dataset classes proportions
def show_classes_proportions(dataset, title):
    data_counts = dataset['CLASS'].value_counts(normalize=True)
    sns.countplot(x='CLASS', data=dataset).set(title=title)
    plt.show()
    print('\n' + title)
    print(data_counts)
    return data_counts


# Analyze dataset classes proportions
def show_boxplot_features(dataset, title):
    sns.boxplot(data=dataset).set(title=title)
    plt.show()


# Show confusion matrix with annotations.
def show_confusion_matrix(cm, f1_score):
    # Create annotations label.
    group_counts = ["{0:0.0f}\n".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cm.shape[0], cm.shape[1])
    # Show confusion matrix with heat map.
    sns.heatmap(cm, annot=box_labels, fmt="", cmap="YlGnBu", cbar=False, linewidths=1.0)\
        .set(title='Confusion matrix', xlabel='Predicted class\n\nF1 macro: %0.4f' % f1_score, ylabel='Actual class')
    plt.show()
