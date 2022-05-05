
# IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVR


# Get training and testing dataframes
# INPUT
# df = dataframe to split
# test_size = float, size of test split, default = 0.25
# OUTPUT
# X_train, X_test, y_train, y_test = training and
# testing dataframes
def get_train_test(df, test_size=0.25):

    # taking absolute value of z-scores
    df["n1Zscore"] = abs(df["n1Zscore"])
    df["n2Zscore"] = abs(df["n2Zscore"])
    df["p2Zscore"] = abs(df["p2Zscore"])

    X = df[["n1Zscore", "n2Zscore", "p2Zscore", "n1Latency", "n2Latency", "p2Latency"]]
    y = df[["outcome"]]["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    return X_train, X_test, y_train, y_test

# Linear regression model
# INPUT
# df = dataframe
# OUTPUT
# lr = trained linear regression model
# tpr
# fpr
# precision
# recall
def linear_regression(df, plot_roc = False, plot_pr = False):

    X_train, X_test, y_train, y_test = get_train_test(df)

    lr = linear_model.LinearRegression()
    y_pred = lr.fit(X_train, y_train).predict(X_test)
    y_rounded = np.array(pd.Series(y_pred).round())
    print("Linear Regression - Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_rounded).sum()))

    if plot_roc:
        tpr, fpr = roc(y_test, y_pred)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

    if plot_pr:
        precision, recall = pr(y_test, y_pred)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

    return lr, tpr, fpr, precision, recall

# Naive Bayes model
# INPUT
# df = dataframe
# OUTPUT
# gnb = trained gaussian naive bayes model
# tpr
# fpr
# precision
# recall
def naive_bayes(df, plot_roc = False, plot_pr = False):

    X_train, X_test, y_train, y_test = get_train_test(df)

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print("Naive Bayes - Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != gnb.fit(X_train, y_train).predict(X_test)).sum()))

    if plot_roc:
        tpr, fpr = roc(y_test, y_pred)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

    if plot_pr:
        precision, recall = pr(y_test, y_pred)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

    return gnb, tpr, fpr, precision, recall


# Random Forest model
# INPUT
# df = dataframe
# max_depth = int, depth of random forest
# OUTPUT
# rf = trained random forest model
# tpr
# fpr
# precision
# recall
def random_forest(df, max_depth, plot_roc = False, plot_pr = False):

    X_train, X_test, y_train, y_test = get_train_test(df)

    rf = RandomForestClassifier(max_depth=max_depth, random_state=0)
    y_pred = rf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print("Random Forest - Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != rf.fit(X_train, y_train).predict(X_test)).sum()))

    if plot_roc:
        tpr, fpr = roc(y_test, y_pred)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

    if plot_pr:
        precision, recall = pr(y_test, y_pred)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

    return rf, tpr, fpr, precision, recall

# XGBoost model
# INPUT
# df = dataframe
# learning_rate = int, learning rate
# max_depth = int, depth of forest
# n_estimators = number of estimators
# OUTPUT
# xgb = trained xgboost model
# tpr
# fpr
# precision
# recall
def xgboost(df, learning_rate = 0.5, max_depth = 10, n_estimators = 10, plot_roc = False, plot_pr = False):

    X_train, X_test, y_train, y_test = get_train_test(df)

    xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
    y_pred = xgb.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print("XGBoost - Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != xgb.fit(X_train, y_train).predict(X_test)).sum()))

    if plot_roc:
        tpr, fpr = roc(y_test, y_pred)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

    if plot_pr:
        precision, recall = pr(y_test, y_pred)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

    return xgb, tpr, fpr, precision, recall


# AdaBoost model
# INPUT
# df = dataframe
# n_estimators = number of estimators
# OUTPUT
# ada = trained adaboost model
# tpr
# fpr
# precision
# recall
def adaboost(df, n_estimators = 10, plot_roc = False, plot_pr = False):

    X_train, X_test, y_train, y_test = get_train_test(df)

    ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=0)
    y_pred = ada.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print("Adaboost - Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != ada.fit(X_train, y_train).predict(X_test)).sum()))

    if plot_roc:
        tpr, fpr = roc(y_test, y_pred)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

    if plot_pr:
        precision, recall = pr(y_test, y_pred)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

    return ada, tpr, fpr, precision, recall


# AdaBoost model
# INPUT
# df = dataframe
# n_estimators = number of estimators
# OUTPUT
# sv = trained svm model
# tpr
# fpr
# precision
# recall
def svm(df, C = 0.1, epsilon = 0.1, plot_roc = False, plot_pr = False):

    X_train, X_test, y_train, y_test = get_train_test(df)

    sv = SVR(C=C, epsilon=epsilon)
    y_pred = sv.fit(X_train, y_train).predict(X_test)
    y_rounded = np.array(pd.Series(y_pred).round())
    print("SVM - Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_rounded).sum()))

    if plot_roc:
        tpr, fpr = roc(y_test, y_pred)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

    if plot_pr:
        precision, recall = pr(y_test, y_pred)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

    return sv, tpr, fpr, precision, recall



# Plot ROC curve
# INPUT
# y_test = y test values
# y_pred = y predictions
# plot = bool, true if plot and false if not
# OUTPUT
# tpr = true positive rate
# fpr = false postiive rate
def roc(y_test, y_pred, plot = True):

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    if plot:
        plt.figure()
        plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("fpr")
        plt.ylabel("tpr")

    return tpr, fpr

# Plot PR curve
# INPUT
# y_test = y test values
# y_pred = y predictions
# plot = bool, true if plot and false if not
# OUTPUT
# precision
# recall
def pr(y_test, y_pred, plot = True):

    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

    if plot:
        plt.figure()
        plt.plot(recall, precision, label='AP = %0.2f' % (np.mean(precision)))
        plt.legend(loc='lower right')
        plt.title("PR Curve")
        plt.xlabel("recall")
        plt.ylabel("precision")

    return precision, recall




