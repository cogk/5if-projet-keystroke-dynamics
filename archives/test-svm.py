from SVM import get_trained_model
from data_loader import load_data, all_columns

model = get_trained_model()

df = load_data()

X = df[all_columns]
y = df[["subject"]]

a, b = 0, 0

for i in range(0, 400):
    ax = X.iloc[i]
    ay = y.iloc[i]

    y_pred = model.predict([ax])
    if (ay.values[0] == y_pred):
        a += 1
    else:
        b += 1

print("Correct: ", a)
print("Incorrect: ", b)
print("Accuracy: ", a / (a + b))
