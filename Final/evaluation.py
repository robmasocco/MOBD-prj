"""
    Authors: Alessandro Tenaglia, Roberto Masocco
    Project: MOBD-prj
    File: evaluation.py
    Date created: 19/06/2020
    Description: Final script for classifier evaluation on the test set.
"""

import pickle

from aux_lib import *


def evaluation():
    """Evaluates our classifier on the test set."""
    # Load our classifier.
    with open('best_pipeline.sav', 'rb') as model_file:
        best_pipeline = pickle.load(model_file)

    # Load test set.
    testset_path = str(input("Enter test set file name: "))
    testset = pd.read_csv(testset_path)
    print("TEST SET IMPORTED")

    # Separate features and labels.
    x = testset.drop('CLASS', axis=1)
    y = testset['CLASS']

    evaluate_classifier(best_pipeline, x, y)


# Start the script.
if __name__ == '__main__':
    evaluation()
