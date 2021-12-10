from Manhattan import get_trained_model, predict
from data_loader import load_data, columns

model = get_trained_model()

df = load_data()

X = df[columns]
y = df[["subject"]]

a, b = 0, 0

for i in range(0, 400):
    ax = X.iloc[i]
    ay = y.iloc[i]

    y_pred = predict(model, ax)
    if (ay.values[0] == y_pred):
        a += 1
    else:
        b += 1

print("Correct: ", a)
print("Incorrect: ", b)
