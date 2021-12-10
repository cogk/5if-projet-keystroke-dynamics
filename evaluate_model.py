from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics


def evaluate_model_simple(X_train, X_test, y_train, y_test, predict):
    print("Accuracy on training data : {}".format(
        accuracy_score(y_train, predict(X_train))))
    y_test_pred = predict(X_test)
    acc_test = metrics.accuracy_score(y_test.values, y_test_pred)
    print("Accuracy on testing data : {}".format(acc_test))


def evaluate_model(X_train, X_test, y_train, y_test, predict):
    print("Accuracy on training data : {}".format(
        accuracy_score(y_train, predict(X_train))))

    y_test_pred = predict(X_test)

    acc_test = metrics.accuracy_score(y_test.values, y_test_pred)
    print("Accuracy on testing data : {}".format(acc_test))
    print()

    f1_score = metrics.f1_score(y_test, y_test_pred, average='micro')
    print("F1 Score on testing data : {}".format(f1_score))

    fbeta_score = metrics.fbeta_score(
        y_test, y_test_pred, average='micro', beta=0.5)
    print("Fbeta Score on testing data : {}".format(fbeta_score))

    hamming_loss = metrics.hamming_loss(y_test, y_test_pred)
    print("Hamming Loss on testing data : {}".format(hamming_loss))

    jaccard_score = metrics.jaccard_score(y_test, y_test_pred, average='micro')
    print("Jaccard Score on testing data : {}".format(jaccard_score))

    mcc = metrics.matthews_corrcoef(y_test.values, y_test_pred)
    print("MCC on testing data : {}".format(mcc))

    precision_score = metrics.precision_score(
        y_test, y_test_pred, average='micro')
    print("Precision on testing data : {}".format(precision_score))

    recall_score = metrics.recall_score(y_test, y_test_pred, average='micro')
    print("Recall on testing data : {}".format(recall_score))

    zero_one_loss = metrics.zero_one_loss(y_test, y_test_pred)
    print("Zero One Loss on testing data : {}".format(zero_one_loss))
