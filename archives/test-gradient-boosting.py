import pandas as pd
import numpy as np

from GradientBoosting import get_trained_model, columns

# Loading
model = get_trained_model()
df = pd.read_csv("./cmu-data.csv")

# Sampling
all_subjects = df.subject.unique()
subj = np.random.choice(all_subjects)
df_sample = df[df.subject == subj].iloc[0:3]

X = df_sample[columns]
y = df_sample[["subject"]]

# Prediction
y_pred = model.predict_proba(X)
y_pred = np.prod(np.vstack(y_pred), axis=0)
final_prediction = np.argmax(y_pred)

# Results
final_prediction = all_subjects[final_prediction]
print('subject =', final_prediction)
print('expected =', subj)
