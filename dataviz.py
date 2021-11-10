import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cmu-data.csv')

sns.set_theme(color_codes=True, style="ticks")

idx = df.groupby("subject").agg({'H.period': np.median}).sort_values(
    by='H.period', ascending=False).index

sns.boxplot(
    x="subject",
    y="H.period",
    data=df,
    order=idx,
)

plt.show()

data = df[['subject', 'H.period', 'H.t', 'H.e']]

sns.pairplot(data, hue="subject")

plt.show()
