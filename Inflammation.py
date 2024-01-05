# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:02:53 2023

@author: Christina
"""

import pandas as pd 

features = ["a1", "a2", "a3", "a4", "a5", "a6", "d1", "d2"]

data = pd.read_csv("diagnosis.data.csv", delimiter=",", names=features)

data2 = pd.get_dummies(data, dtype=int)

inputs = data2.iloc[:, 0:-4:]
target = data2.iloc[:, -1:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.30,random_state=1)

# svm prediction

from sklearn.svm import SVC #svc(support vector classifier)

dvs = SVC(kernel="poly", degree=3)

dvs.fit(x_train, y_train)

prediction_dvs_train = dvs.predict(x_train)
prediction_dvs_test = dvs.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

cm_dvs_train = confusion_matrix(y_train, prediction_dvs_train)
cm_dvs_test = confusion_matrix(y_test, prediction_dvs_test)

acs_dvs_train = accuracy_score(y_train, prediction_dvs_train)
acs_dvs_test = accuracy_score(y_test, prediction_dvs_test)

#precision and recall will show us the d/f in fn or fp values 

ps_dvs_train = precision_score(y_train, prediction_dvs_train)
rs_dvs_train = recall_score(y_train,prediction_dvs_train)

ps_dvs_test = precision_score(y_test, prediction_dvs_test)
rs_dvs_test = recall_score(y_test, prediction_dvs_test)




































