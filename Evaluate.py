"""
    Author: Alessandro Tenaglia
    Project: MOBD-prj
    File: Evaluate.py
    Date created: 15/06/2020
    Description: 
    
"""

import sklearn.metrics as metrics

# utilizziamo ora il miglior modello ottenuto al termine della cross-validation per fare previsioni sui dati di test
def evaluate_classifier(classifier, test_x, test_y):
    pred_y = classifier.predict(test_x)
    confusion_matrix = metrics.confusion_matrix(test_y, pred_y)
    print(confusion_matrix)
    f1_score = metrics.f1_score(test_y, pred_y, average='macro')
    print('F1: %0.4f' % f1_score)
