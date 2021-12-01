import sys
import numpy as np
import pandas as pd
from data import get_splitted_data
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
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

    y_test_pred = classifier.predict(X_test)

    acc_test = metrics.accuracy_score(y_test.values, y_test_pred)
    f1_score = metrics.f1_score(y_test, y_test_pred, average='micro')
    fbeta_score = metrics.fbeta_score(
        y_test, y_test_pred, average='micro', beta=0.5)
    hamming_loss = metrics.hamming_loss(y_test, y_test_pred)
    jaccard_score = metrics.jaccard_score(y_test, y_test_pred, average='micro')
    mcc = metrics.matthews_corrcoef(y_test.values, y_test_pred)
    precision_score = metrics.precision_score(
        y_test, y_test_pred, average='micro')
    recall_score = metrics.recall_score(y_test, y_test_pred, average='micro')
    zero_one_loss = metrics.zero_one_loss(y_test, y_test_pred)

    print()
    print("Accuracy on testing data : {}".format(acc_test))
    print("F1 Score on testing data : {}".format(f1_score))
    print("Fbeta Score on testing data : {}".format(fbeta_score))
    print("Hamming Loss on testing data : {}".format(hamming_loss))
    print("Jaccard Score on testing data : {}".format(jaccard_score))
    print("MCC on testing data : {}".format(mcc))
    print("Precision on testing data : {}".format(precision_score))
    print("Recall on testing data : {}".format(recall_score))
    print("Zero One Loss on testing data : {}".format(zero_one_loss))

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
