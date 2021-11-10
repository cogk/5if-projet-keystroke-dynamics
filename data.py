import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

columns = 'H.period,DD.period.t,UD.period.t,H.t,DD.t.i,UD.t.i,H.i,DD.i.e,UD.i.e,H.e,DD.e.five,UD.e.five,H.five,DD.five.Shift.r,UD.five.Shift.r,H.Shift.r,DD.Shift.r.o,UD.Shift.r.o,H.o,DD.o.a,UD.o.a,H.a,DD.a.n,UD.a.n,H.n,DD.n.l,UD.n.l,H.l,DD.l.Return,UD.l.Return,H.Return'.split(
    ',')

columns_dropped = ['sessionIndex', 'rep']


def load_data():
    df = pd.read_csv("./cmu-data.csv")
    df.drop(columns=columns_dropped, inplace=True)
    return df


def get_X_y(df=None):
    if df is None:
        df = load_data()

    X = df[columns]
    y = df[["subject"]]
    return X, y


def get_splitted_data(df=None):
    return split_data(*get_X_y(df))


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7)
    return X_train, X_test, y_train, y_test


def _load_model(model_name):
    print('load model')
    try:
        with open(model_name, 'rb') as f:
            print('-> loaded')
            return pickle.load(f)
    except:
        print('-> no saved model')
        return None


def _save_model(classifier, model_name):
    with open(model_name, 'wb') as f:
        pickle.dump(classifier, f, 4)
