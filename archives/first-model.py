import pandas as pd
import scipy.stats
import math
import numpy as np
from functools import reduce
import operator
import random


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s < m]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def random_range():
    offset, length = random.randrange(0, 380), random.randrange(1, 10)
    return range(offset, offset + length)


csv = pd.read_csv('cmu-data.csv')


def main():
    # verif()
    # exit()

    H_columns = list(filter(lambda n: n.startswith('H.'), csv.columns))
    H_cols = random.sample(H_columns, 7)

    subject = random.choice(csv.subject.unique())
    print('Subject:', subject)
    print('Columns:', H_cols)

    samples = csv[csv.subject == subject]

    mesures = []
    for c in H_cols:
        s = samples[c]
        mesures += [(c, s.iloc[i]) for i in random_range()]
        #mesures += [(c, 0.1 + 0.01 * random.random()) for i in random_range()]

    scores = [score_mesure_all(*m) for m in mesures]

    scores = reduce(operator.add, scores) / len(scores)
    scores = scores[scores > 0.5]

    score_diff = np.max(scores) - scores
    score_diff = score_diff[score_diff < 0.05]

    # print(score_sum.sort_values(ascending=False).head(5))
    candidates = list(score_diff.sort_values().index.values)
    print(candidates)


def verif():
    import matplotlib.pyplot as plt
    plt.show()

    H_columns = list(filter(lambda n: n.startswith('H.'), csv.columns))
    subjects = [26]  # list(csv.subject.unique())

    print(subjects)
    print(H_columns)

    alpha = 0.01
    fails = []
    for s in subjects:
        for h in H_columns:
            x = csv[csv.subject == s][h]
            x = reject_outliers(x)
            k2, p = scipy.stats.normaltest(x)
            if p < alpha:
                fails.append((p, s, h))
                # print(f'\x1b[31;1mFail:\x1b[m s={s}, h={h}, p={p}')

    fails.sort(key=lambda x: x[0], reverse=False)
    for f in fails:
        p, s, h = f
        print(f'plt.hist(csv[csv.subject == {s}]["{h}"], bins=40)  # {p}')

    # print()
    # print(x.describe())
    # plt.hist(x, bins=40)
    # plt.show()


def score_mesure_all(col, hold_duration):
    distribution = csv.groupby('subject')[col]
    mean = distribution.mean()
    std = distribution.std()

    distance = sigmoid((hold_duration - mean) / std)
    # [0; 1], 0.5 is best

    score = 1 - 2 * abs(distance - 0.5)
    # [0; 1], 1 is best

    return score


if __name__ == '__main__':
    main()
