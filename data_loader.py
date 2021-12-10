import os
import pickle
import random

import pandas as pd
from sklearn.model_selection import train_test_split

# Least useful columns
# DD/UD .i.e
# DD/UD .a.n
# DD/UD .five.Shift.r
# DD/UD .Shift.r.o
# DD/UD .n.l
# DD/UD .t.i
# DD/UD .period.t
# DD/UD .o.a
# DD/UD .l.Return
# DD/UD .e.five

# Most useful columns
# H.l
# H.e
# H.five
# H.o
# H.n
# H.Return
# H.a
# H.t
# H.period
# H.Shift.r
# H.i

all_columns = [
    'DD.a.n',
    'DD.e.five',
    'DD.five.Shift.r',
    'DD.i.e',
    'DD.l.Return',
    'DD.n.l',
    'DD.o.a',
    'DD.period.t',
    'DD.Shift.r.o',
    'DD.t.i',

    'H.a',
    'H.e',
    'H.five',
    'H.i',
    'H.l',
    'H.n',
    'H.o',
    'H.period',
    'H.Return',
    'H.Shift.r',
    'H.t',

    'UD.a.n',
    'UD.e.five',
    'UD.five.Shift.r',
    'UD.i.e',
    'UD.l.Return',
    'UD.n.l',
    'UD.o.a',
    'UD.period.t',
    'UD.Shift.r.o',
    'UD.t.i',
]

columns_selected_features = [
    'H.a',
    'H.i',
    'H.n',
    'H.period',
    'H.Return',
    'H.Shift.r',
    'H.t',
    'UD.e.five',
    'UD.i.e',
    'UD.l.Return',
    'UD.Shift.r.o',
    'UD.t.i',
]


# # randomly remove some columns
# def _read_int(file):
#     return int.from_bytes(file.read(4), byteorder='big')
# def _write_int(file, value):
#     file.write(value.to_bytes(4, byteorder='big'))
# if not os.path.exists('j.bin'):
#     with open('j.bin', 'wb') as f:
#         _write_int(f, 0)
# with open('j.bin', 'rb') as f:
#     j = _read_int(f)
# dropped = []
# dropped.append(columns[j])
# columns.pop(j)
# j += 1
# with open('j.bin', 'wb') as f:
#     _write_int(f, j)
# # dropped = []
# # for i in range(1):
# #     j = random.randint(0, len(columns) - 1)
# #     dropped.append(columns[j])
# #     columns.pop(j)
# print(f'\t{1}\tH + UD only but without {" ".join(dropped)}')


columns_dropped = ['sessionIndex', 'rep']


def load_data():
    df = pd.read_csv("./cmu-data.csv")
    df.drop(columns=columns_dropped, inplace=True)
    return df


def get_X_y(df=None, subset=''):
    if df is None:
        df = load_data()

    selected_columns = []

    if 'H' in subset:
        selected_columns += filter(lambda c: c.startswith('H.'), all_columns)
    if 'DD' in subset:
        selected_columns += filter(lambda c: c.startswith('DD.'), all_columns)
    if 'UD' in subset:
        selected_columns += filter(lambda c: c.startswith('UD.'), all_columns)

    if subset == 'selected features':
        selected_columns = columns_selected_features

    if len(selected_columns) == 0:
        selected_columns = all_columns

    X = df[selected_columns]

    X = X.fillna(0)
    X = X.where(X > 0, 0.000000001)

    # X_H = df[filter(lambda x: x.startswith('H.'), columns)].sum(axis=1)
    # X_DD = df[filter(lambda x: x.startswith('DD.'), columns)].sum(axis=1)
    # X = pd.concat([X_H, X_DD], axis=1)

    y = df[["subject"]]
    return X, y


def get_splitted_data(df=None, subset=''):
    return split_data(*get_X_y(df, subset))


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7)
    return X_train, X_test, y_train, y_test


def _load_model(model_name):
    # print('load model')
    try:
        with open(model_name, 'rb') as f:
            print(f'-> loading pre-trained model {model_name}')
            return pickle.load(f)
    except:
        # print(f'-> no saved model')
        return None


def _save_model(classifier, model_name):
    with open(model_name, 'wb') as f:
        pickle.dump(classifier, f, 4)
