#GLM with absolute z-scores
#Author : Zaiwei Liu
#run time: ~1min
  

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn import metrics
from sklearn.metrics import average_precision_score, accuracy_score

from epilepsypcm.utils.make_df import make_df, get_df_list, concat_dfs, class_balance
from epilepsypcm.utils.outcome_params import engel_score

#Location
base_path = '/Users/david/Desktop/PCM_Data/' #modify for your file location


#Function to get a list of all dataframes for all positive patients, in the format [patient number, df]
df_list = get_df_list(base_path, "1")

#Code to loop through this list
#for i in range(len(df_list)):
#    print("Patient ID: ", df_list[i][0])
#    print("Patient Dataframe: ", df_list[i][1].head(3))
 
#Function to get the concatenated dataframe for all positive patients
## balance parameter can be changed to "None", "upsample", or "downsample"
all_positive_patients = concat_dfs(base_path, "1", balance = "None")

#print("Full Dataframe: ", all_positive_patients.head(3))

from sklearn.model_selection import train_test_split

# drop certain columns and get X and Y
X = all_positive_patients.drop(['chNames','outcome','significant','flipped'], axis=1)
Y = all_positive_patients['outcome']

# run test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=100)
x_train, y_train = class_balance(x_train, y_train, balance = 'downsample')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train[['n1Zscore','n2Zscore','p2Zscore','n1Latency','n2Latency','p2Latency']]=scaler.fit_transform(x_train[['n1Zscore','n2Zscore','p2Zscore','n1Latency','n2Latency','p2Latency']])
x_test[['n1Zscore','n2Zscore','p2Zscore','n1Latency','n2Latency','p2Latency']]=scaler.fit_transform(x_test[['n1Zscore','n2Zscore','p2Zscore','n1Latency','n2Latency','p2Latency']])
# absolute value of zscore
x_train = x_train.abs()
x_test = x_test.abs()

# Importing libraries to build the moddel
import statsmodels.api as sm

# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(x_train)), family = sm.families.Binomial())
res1 = logm1.fit()
#res1.summary()

# all parameters test set
x_test_final = sm.add_constant(x_test)

y_test_pred = res1.predict(x_test_final)
y_test_pred_final = pd.DataFrame({'EZ':y_test.values, 'EZ_Prob':y_test_pred})
y_test_pred_final['NodeID'] = y_test.index

# find best probability
accuracy = []
for i in np.arange(0,1,0.0001):
    y_test_pred_final['predicted'] = y_test_pred_final.EZ_Prob.map(lambda x: 1 if x > i else 0)
    accuracy.append(metrics.accuracy_score(y_test_pred_final.EZ, y_test_pred_final.predicted))
best_pro_index = accuracy.index(max(accuracy))

# Creating new column 'predicted' 
y_test_pred_final['predicted'] = y_test_pred_final.EZ_Prob.map(lambda x: 1 if x > (best_pro_index/10000) else 0)

# print('Accuracy on the test set is',metrics.accuracy_score(y_test_pred_final.EZ, y_test_pred_final.predicted))

# ROC Curve and auc value for the test set

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred)
roc_auc = metrics.auc(fpr,tpr)
# print('AUC of the test set is', roc_auc)

# pr curve and average
precision,recall,threshold = metrics.precision_recall_curve(y_test, y_test_pred)
Ave_Prec = average_precision_score(y_test, y_test_pred, average='macro', pos_label=1, sample_weight=None)
# print('AP of the test set is', Ave_Prec)

# dummy classifier for pr curve 
y_pred_dummy = np.random.randint(0, 2, len(y_test))
pr_dummy = len(y_test[y_test==1]) / len(y_test)
# print('AP for the dummy classifier is',pr_dummy)

# plot ROC
plt.figure()
plt.plot([0,1],[0,1],linestyle = '--')
plt.plot(fpr,tpr)
plt.legend(['No Model: AUC is 0.5','GLM: AUC is {}'.format('%0.2f'%roc_auc)], loc = 'best')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve GLM')

# PR Curve and AP value for the test set
plt.figure()
plt.plot(recall,precision)
plt.plot([0, 1], [pr_dummy, pr_dummy], linestyle = '--')
plt.legend(['No Model: AP is {}'.format('%0.2f'%pr_dummy),'GLM: AP is {}'.format('%0.2f'%Ave_Prec)], loc = 'best')
plt.title('PR Curve GLM')
