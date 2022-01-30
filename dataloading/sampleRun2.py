#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 04:23:45 2021

@author: richardlee
"""

import json
import glob
import os
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from epilepsypcm.utils.make_df import make_df, get_df_list, concat_dfs
from epilepsypcm.utils.outcome_params import engel_score


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay

#Location
base_path = '/Users/richardlee/Desktop/2021 Fall/Precision Care Medicine/Coding/' #modify for your file location



#Function to get a list of all dataframes for all positive patients, in the format [patient number, df]
df_list = get_df_list(base_path, "1")

"""
#Code to loop through this list
for i in range(len(df_list)):
    print("Patient ID: ", df_list[i][0])
    print("Patient Dataframe: ", df_list[i][1].head(3))
"""

    
#Function to get the concatenated dataframe for all positive patients
all_positive_patients = concat_dfs(base_path, "1")

# print("Full Dataframe: ", all_positive_patients.head(3))

keys = all_positive_patients.keys()

X = all_positive_patients[keys[2:8]]
y = all_positive_patients.outcome

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) # 70% training and 30% test

#Create a Gaussian Classifier
rfc=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

feature_imp = pd.Series(rfc.feature_importances_,index=all_positive_patients.keys()[2:8]).sort_values(ascending=False)
feature_imp

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

rfc_disp = RocCurveDisplay(rfc, X_test, y_test,estimator_name='example estimator')
rfc_disp.plot()
plt.show()

























