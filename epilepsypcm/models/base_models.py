
# IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVR

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


# Get training and testing dataframes
# INPUT
# df = dataframe to split
# X_cols = list of columsn to use as x values
# test_size = float, size of test split, default = 0.25
# SMOTE, true if u want train set upsampled and false if not
# OUTPUT
# X_train, X_test, y_train, y_test = training and
# testing dataframes
def get_train_test(df, X_cols, smote="no", test_size=0.25):

    # taking absolute value of z-scores
    if "n1Zscore" in X_cols:
        df["n1Zscore"] = abs(df["n1Zscore"])
        df["n2Zscore"] = abs(df["n2Zscore"])
        df["p2Zscore"] = abs(df["p2Zscore"])

    #X = df[["n1Zscore", "n2Zscore", "p2Zscore", "n1Latency", "n2Latency", "p2Latency"]]
    X = df[X_cols]
    y = df[["outcome"]]["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, stratify = y)

    if smote==True:
        train = pd.concat([X_train, y_train], axis=1)
        y = train['outcome']
        oversample = SMOTE()
        X_nochannel = train.drop(columns='Channels')
        X_train, y_train = oversample.fit_resample(X_nochannel, y)
        X_train["Channels"] = 0
        X_train = X_train.drop(columns = "outcome")

    if smote=="resample":
        train = pd.concat([X_train, y_train], axis=1)
        y = train['outcome']
        X = train.drop(columns='outcome')
        oversample = RandomOverSampler(sampling_strategy='minority')
        X_train, y_train = oversample.fit_resample(X, y)
        X_train["Channels"] = 0
     #   X_train = X_train.drop(columns="outcome")

    return X_train, X_test, y_train, y_test

# Linear regression model
# INPUT
# df = dataframe
# X_cols = x columns to train on
# OUTPUT
# lr = trained linear regression model
# Smote, true or false
# test_channels
# y_pred
# y_test
# tpr
# fpr
# roc_thresholds
# precision
# recall
def logistic_regression(df, X_cols, plot_roc = False, plot_pr = False, smote = False):

    X_train, X_test, y_train, y_test = get_train_test(df, X_cols, smote)

    test_channels = list(X_test["Channels"])
    X_train = X_train.drop(columns="Channels")
    X_test = X_test.drop(columns="Channels")

    lr = LogisticRegression()
    y_pred = lr.fit(X_train, y_train).predict_proba(X_test)[:,1]
    y_rounded = np.array(pd.Series(y_pred).round())
    print("Logistic Regression - Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_rounded).sum()))

    if plot_roc:
        tpr, fpr = roc(y_test, y_pred)
    else:
        fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

    if plot_pr:
        precision, recall = pr(y_test, y_pred)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

    return lr, test_channels, y_pred, y_test, tpr, fpr, roc_thresholds, precision, recall

# Naive Bayes model
# INPUT
# df = dataframe
# X_cols = x columns to train on
# smote, true or false
# OUTPUT
# gnb = trained gaussian naive bayes model
# test_channels
# y_pred
# y_test
# tpr
# fpr
# roc_thresholds
# precision
# recall
def naive_bayes(df, X_cols, plot_roc = False, plot_pr = False, smote = False):

    X_train, X_test, y_train, y_test = get_train_test(df, X_cols, smote)

    test_channels = list(X_test["Channels"])
    X_train = X_train.drop(columns="Channels")
    X_test = X_test.drop(columns="Channels")

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print("Naive Bayes - Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != gnb.fit(X_train, y_train).predict(X_test)).sum()))

    if plot_roc:
        tpr, fpr = roc(y_test, y_pred)
    else:
        fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

    if plot_pr:
        precision, recall = pr(y_test, y_pred)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

    return gnb, test_channels, y_pred, y_test, tpr, fpr, roc_thresholds, precision, recall


# Random Forest model
# INPUT
# df = dataframe
# X_cols = x columns to train on
# max_depth = int, depth of random forest
# smote, true or false
# OUTPUT
# rf = trained random forest model
# test_channels
# y_pred
# y_test
# tpr
# fpr
# roc_thresholds
# precision
# recall
def random_forest(df, X_cols, max_depth, plot_roc = False, plot_pr = False, smote = False):

    X_train, X_test, y_train, y_test = get_train_test(df, X_cols, smote)

    test_channels = list(X_test["Channels"])
    X_train = X_train.drop(columns="Channels")
    X_test = X_test.drop(columns="Channels")

    rf = RandomForestClassifier(max_depth=max_depth, random_state=0)
    y_pred = rf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print("Random Forest - Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != rf.fit(X_train, y_train).predict(X_test)).sum()))

    if plot_roc:
        tpr, fpr = roc(y_test, y_pred)
    else:
        fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

    if plot_pr:
        precision, recall = pr(y_test, y_pred)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

    return rf, test_channels, y_pred, y_test, tpr, fpr, roc_thresholds, precision, recall

# XGBoost model
# INPUT
# df = dataframe
# X_cols = x columns to train on
# learning_rate = int, learning rate
# max_depth = int, depth of forest
# n_estimators = number of estimators
# smote, true or false
# OUTPUT
# xgb = trained xgboost model
# test_channels
# y_pred
# y_test
# tpr
# fpr
# roc_thresholds
# precision
# recall
def xgboost(df, X_cols, learning_rate = 0.5, max_depth = 10, n_estimators = 10, plot_roc = False, plot_pr = False, smote = False):

    X_train, X_test, y_train, y_test = get_train_test(df, X_cols, smote)

    test_channels = list(X_test["Channels"])
    X_train = X_train.drop(columns="Channels")
    X_test = X_test.drop(columns="Channels")

    xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
    y_pred = xgb.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print("XGBoost - Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != xgb.fit(X_train, y_train).predict(X_test)).sum()))

    if plot_roc:
        tpr, fpr = roc(y_test, y_pred)
    else:
        fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

    if plot_pr:
        precision, recall = pr(y_test, y_pred)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

    return xgb, test_channels, y_pred, y_test, tpr, fpr, roc_thresholds, precision, recall


# AdaBoost model
# INPUT
# df = dataframe
# X_cols = x columns to train on
# n_estimators = number of estimators
# smote, true or false
# OUTPUT
# ada = trained adaboost model
# test_channels
# y_pred
# y_test
# tpr
# fpr
# roc_thresholds
# precision
# recall
def adaboost(df, X_cols, n_estimators = 10, plot_roc = False, plot_pr = False, smote = False):

    X_train, X_test, y_train, y_test = get_train_test(df, X_cols, smote)

    test_channels = list(X_test["Channels"])
    X_train = X_train.drop(columns="Channels")
    X_test = X_test.drop(columns="Channels")

    ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=0)
    y_pred = ada.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print("Adaboost - Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != ada.fit(X_train, y_train).predict(X_test)).sum()))

    if plot_roc:
        tpr, fpr = roc(y_test, y_pred)
    else:
        fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

    if plot_pr:
        precision, recall = pr(y_test, y_pred)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

    return ada, test_channels, y_pred, y_test, tpr, fpr, roc_thresholds, precision, recall


# AdaBoost model
# INPUT
# df = dataframe
# X_cols = x columns to train on
# n_estimators = number of estimators
# smote, true or false
# OUTPUT
# sv = trained svm model
# test_channels
# y_pred
# y_test
# tpr
# fpr
# roc_thresholds
# precision
# recall
def svm(df, X_cols, C = 0.1, epsilon = 0.1, plot_roc = False, plot_pr = False, smote = False):

    X_train, X_test, y_train, y_test = get_train_test(df, X_cols, smote)

    test_channels = list(X_test["Channels"])
    X_train = X_train.drop(columns="Channels")
    X_test = X_test.drop(columns="Channels")

    sv = SVR(C=C, epsilon=epsilon)
    y_pred = sv.fit(X_train, y_train).predict(X_test)
    y_rounded = np.array(pd.Series(y_pred).round())
    print("SVM - Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_rounded).sum()))

    if plot_roc:
        tpr, fpr = roc(y_test, y_pred)
    else:
        fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

    if plot_pr:
        precision, recall = pr(y_test, y_pred)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

    return sv, test_channels, y_pred, y_test, tpr, fpr, roc_thresholds, precision, recall



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




