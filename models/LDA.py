import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from data_loader import _load_model, _save_model, get_splitted_data
from evaluate_model import evaluate_model_simple


def load_model():
    return _load_model('./trained/lda.pickle')


def save_model(classifier):
    return _save_model(classifier, './trained/lda.pickle')


def get_trained_model():
    classifier = load_model()
    if classifier is not None:
        return classifier
    else:
        classifier = LinearDiscriminantAnalysis(n_components=3)

    X_train, X_test, y_train, y_test = get_splitted_data()

    # print("Training...")
    classifier.fit(np.array(X_train), np.array(y_train.values.ravel()))

    evaluate_model_simple(X_train, X_test, y_train, y_test,
                          lambda x: classifier.predict(x))

    save_model(classifier)

    return classifier
