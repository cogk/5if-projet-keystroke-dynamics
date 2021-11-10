import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.svm import SVC

from data import _load_model, _save_model, get_X_y, get_splitted_data, load_data, columns, split_data


def load_model():
    return _load_model('./svm-model.pickle')


def save_model(classifier):
    return _save_model(classifier, './svm-model.pickle')


def get_trained_model():
    KERNEL = 'rbf'

    classifier = None  # load_model()
    if classifier is not None:
        return classifier
    else:
        # linear, poly, rbf (≈ gaussian), ...
        classifier = SVC(kernel=KERNEL)

    X_train, X_test, y_train, y_test = get_splitted_data()

    print("Training...")
    classifier.fit(np.array(X_train), np.array(y_train.values.ravel()))

    y_test_pred = classifier.predict(X_test)

    acc_train = metrics.accuracy_score(
        y_train.values, classifier.predict(X_train))

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

    print("Accuracy on training data : {}".format(acc_train))

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

    print("Sample:", KERNEL, acc_train, acc_test, f1_score, fbeta_score,
          hamming_loss, jaccard_score, mcc, precision_score, recall_score, zero_one_loss)

    save_model(classifier)

    return classifier
