import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def calculate_metrics(y_true, y_pred): 
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] >= 0.5:
            TP += 1
        if y_true[i] == 0 and y_pred[i] < 0.5:
            TN += 1
        if y_true[i] == 0 and y_pred[i] >= 0.5:
            FP += 1
        if y_true[i] == 1 and y_pred[i] < 0.5:
            FN += 1
    recall = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    F1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    numerator = TP * TN - FP * FN
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = numerator / (denominator + 1e-10)
    return recall, precision, F1_score, accuracy, specificity, mcc


def get_result(loader, model):
    pred, target = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.float(), y.float()
            y_hat = model(x)
            pred += list(y_hat.cpu().numpy())
            target += list(y.cpu().numpy())
    auc = roc_auc_score(target, pred)
    rec, pre, F1, acc, spe, mcc = calculate_metrics(target, pred)
    return auc, rec, pre, F1, acc, spe, mcc


def get_result1(y_hat, y):
    y_hat_np = y_hat.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy()

    auc = roc_auc_score(y_np, y_hat_np)
    rec, pre, F1, acc, spe, mcc = calculate_metrics(y_np, y_hat_np)
    return auc, rec, pre, F1, acc, spe, mcc
