import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.svm import SVC

from data_loader import _load_model, _save_model, get_X_y, get_splitted_data, load_data, columns, split_data
from evaluate_model import evaluate_model_simple


def load_model():
    return _load_model('./trained/svm.pickle')


def save_model(classifier):
    return _save_model(classifier, './trained/svm.pickle')


def get_trained_model():
    KERNEL = 'rbf'

    classifier = load_model()
    if classifier is not None:
        return classifier
    else:
        # linear, poly, rbf (≈ gaussian), ...
        classifier = SVC(kernel=KERNEL)

    X_train, X_test, y_train, y_test = get_splitted_data()

    print("Training...")
    classifier.fit(np.array(X_train), np.array(y_train.values.ravel()))

    evaluate_model_simple(X_train, X_test, y_train, y_test,
                          lambda x: classifier.predict(x))

    save_model(classifier)

    return classifier
