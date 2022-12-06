###SOME CODE PROVIDED AS BOILERPLATE CODE FOR EIGHT BIT BIAS BOUNTY###
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


def class_accuracies(pred, actual):
    ypred = [[np.argmax(pred[i][j, :]) for j in range(pred[i].shape[0])] for i in range(len(pred))]
    ytrue = [[np.argmax(actual[i][j, :]) for j in range(actual[i].shape[0])] for i in range(len(pred))]

    accs = []
    for i in range(len(ypred)):
        cm = confusion_matrix(ytrue[i], ypred[i])
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        accs.append(list(cm.diagonal()))

    return accs

def model_accuracy(pred, actual):
    ypred = [[np.argmax(pred[i][j, :]) for j in range(pred[i].shape[0])] for i in range(len(pred))]
    ytrue = [[np.argmax(actual[i][j, :]) for j in range(actual[i].shape[0])] for i in range(len(pred))]

    accs = []
    for i in range(len(ypred)):
        acc = accuracy_score(ytrue[i], ypred[i])
        accs.append(acc)

    return accs

def min_accuracy(accs):
    min = accs[0]
    arg_min = 0

    for i in range(len(accs)):
        if accs[i] < min:
            min = accs[i]
            arg_min = i

    return arg_min

def disparities(class_accuracies):
    disparities = []
    for i in range(3):
        disparities.append(max(class_accuracies[i]) - min(class_accuracies[i]))

    return disparities


def failure_positions(pred, actual, target_labels):
    ypred = [[np.argmax(pred[i][j, :]) for j in range(pred[i].shape[0])] for i in range(len(pred))]
    ytrue = [[np.argmax(actual[i][j, :]) for j in range(actual[i].shape[0])] for i in range(len(pred))]

    mislabeled = []
    for i in range(len(ypred)):
        mislabeled.append([])
        for j in range(len(ypred[i])):
            if ytrue[i][j] == target_labels[i] and ypred[i][j] != ytrue[i][j]:
                mislabeled[i].append(j)

    return mislabeled
