import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from data import _load_model, _save_model, get_X_y, get_splitted_data, load_data, columns, split_data


def load_model():
    return _load_model('./manhattan-model.pickle')


def save_model(classifier):
    return _save_model(classifier, './manhattan-model.pickle')


def predict(model, x):
    x = x.values.squeeze()
    y = np.absolute(model - x).sum(axis=1)\
        .sort_values(ascending=True)\
        .index[0]
    return y


def predict_n(model, X):
    Y = []
    for _, row in X.iterrows():
        x = row.to_frame().T
        y = predict(model, x)
        Y.append(y)
    return Y


def get_trained_model():
    model = load_model()
    if model is not None:
        return model
    else:
        model = None

    df = load_data()

    X = df[columns + ["subject"]]
    y = df[["subject"]]

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train = X_train.groupby('subject').mean()
    X_test = X_test.drop(['subject'], axis=1)

    model = X_train

    y_test_pred = predict_n(model, X_test)

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

    save_model(model)
    return model
