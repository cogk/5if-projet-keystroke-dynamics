import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from data import _load_model, _save_model, get_X_y, get_splitted_data, load_data, columns, split_data
from evaluate_model import evaluate_model_simple


def load_model():
    return _load_model('./trained/manhattan.pickle')


def save_model(classifier):
    return _save_model(classifier, './trained/manhattan.pickle')


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

    model = X_train.groupby('subject').mean()
    X_test = X_test.drop(['subject'], axis=1)

    evaluate_model_simple(X_train.drop(
        ['subject'], axis=1), X_test, y_train, y_test, lambda x: predict_n(model, x))

    save_model(model)
    return model
