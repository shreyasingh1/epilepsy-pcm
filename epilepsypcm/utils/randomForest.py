#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 00:14:31 2021

@author: richardlee
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

def randomForest(df,keys):
    X = df[keys]
    y = df.outcome
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    rfc=RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train,y_train)
    y_pred=rfc.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    #create feature importance object
    feature_imp = pd.Series(rfc.feature_importances_,index=keys).sort_values(ascending=False)
    #print(feature_imp)
    #plot the feature importance
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.savefig('RF Feature Importance.png')
    plt.show()
   
    #plot the ROC curve
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
    plt.plot([0,1],[0,1],label='No Model')
    plt.title('ROC Curve: Random Forest Model')
    plt.legend()
    plt.savefig('RF ROC Curve.png')
    plt.show()
   
    
    #plot the Precision-Recall curve
    ax = plt.gca()
    rfc_disp = PrecisionRecallDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
    baseline = len(y_test[y_test==1])/len(y_test)
    plt.plot([0,1],[baseline,baseline],label='No Model (AP=%f)'%(int(baseline*100)/100))
    plt.title('Precision Recall Curve: Random Forest Model')
    plt.legend()
    plt.savefig('RF Precision-Recall Curve.png')
    plt.show()
    
    AUC = metrics.roc_auc_score(y_test, rfc.predict_proba(X_test)[:,1])
    AP = metrics.average_precision_score(y_test, rfc.predict_proba(X_test)[:,1])
    
    print('AUC_ROC = %f'%(AUC))
    print('AP_PR =%f'%(AP))
    
    return AUC, AP, rfc.feature_importances_