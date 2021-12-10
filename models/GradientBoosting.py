import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from data import _load_model, _save_model, get_splitted_data
from evaluate_model import evaluate_model_simple


def load_model():
    return _load_model('./trained/gradient-boosting.pickle')


def save_model(classifier):
    _save_model(classifier, './trained/gradient-boosting.pickle')


def get_trained_model():
    NUM_CLASSIFIERS = int(sys.argv[1]) if len(sys.argv) >= 2 else 20
    MAX_DEPTH = int(sys.argv[2]) if len(sys.argv) >= 3 else 10

    classifier = load_model()

    if classifier is not None:
        return classifier
    else:
        classifier = GradientBoostingClassifier(
            n_estimators=NUM_CLASSIFIERS, max_depth=MAX_DEPTH)

    X_train, X_test, y_train, y_test = get_splitted_data()

    classifier.fit(np.array(X_train), np.array(y_train.values.ravel()))

    evaluate_model_simple(X_train, X_test, y_train, y_test,
                          lambda x: classifier.predict(x))

    # sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=7)
    # for train_index, test_index in sss.split(X, y):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]

    # print(
    #     NUM_CLASSIFIERS,
    #     MAX_DEPTH,
    #     accuracy_score(classifier.predict(X_train), y_train),
    #     accuracy_score(classifier.predict(X_test), y_test)
    # )

    save_model(classifier)

    return classifier
