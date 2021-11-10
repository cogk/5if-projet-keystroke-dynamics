import sys
import numpy as np
import pandas as pd
from data import get_splitted_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
import pickle

columns = 'H.period,H.a,H.e,H.five,H.i,H.l,H.n,H.o,H.Shift.r,H.t,H.Return' .split(
    ',')


def load_model(model_name='./gradient-boosting-model.pickle'):
    print('load model')
    try:
        with open(model_name, 'rb') as f:
            print('-> loaded')
            return pickle.load(f)
    except:
        print('-> no saved model')
        return None


def save_model(classifier, model_name='./gradient-boosting-model.pickle'):
    with open(model_name, 'wb') as f:
        pickle.dump(classifier, f, pickle.HIGHEST_PROTOCOL)


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

    print("Accuracy on training data : {}".format(
        accuracy_score(y_train, classifier.predict(X_train))))
    print("Accuracy on testing data : {}".format(
        accuracy_score(y_test, classifier.predict(X_test))))

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
