# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:07:14 2023

@author: Christina
"""

import pandas as pd

features_10 = ["ID","Radius","Texture","Perimeter","Area","Smoothness","Compactness","Concavity","Conc_Points","Symmetry","Fractal_Dimension" ]

data_10 = pd.read_csv("breast-cancer-wisconsin.data",names=features_10)

features_30 = ["ID","Diagnosis","Radius_mean","Texture_mean","Perimeter_mean","Area_mean","Smoothness_mean","Compactness_mean","Concavity_mean","Conc_Points_mean","Symmetry_mean","Fractal_Dimension_mean",\
               "Radius_SE","Texture_SE","Perimeter_SE","Area_SE","Smoothness_SE","Compactness_SE","Concavity_SE","Conc_Points_SE","Symmetry_SE","Fractal_Dimension_SE",\
               "Radius_Worst","Texture_Worst","Perimeter_Worst","Area_Worst","Smoothness_Worst","Compactness_Worst","Concavity_Worst","Conc_Points_Worst","Symmetry_Worst","Fractal_Dimension_Worst"]

data_30 = pd.read_csv("wdbc.data", names=features_30)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
target = le.fit_transform(data_30.iloc[:, 1:2])

inputs = data_30.iloc[:, 2:32]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.30, random_state=24)

from sklearn.neural_network import MLPClassifier

nnc = MLPClassifier(random_state=0, max_iter=2000, learning_rate_init=0.0005, batch_size=10)
nnc.fit(x_train, y_train)

pred_nnc_train = nnc.predict(x_train)
pred_nnc_test = nnc.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm_nnc_train = confusion_matrix(y_train, pred_nnc_train)
cm_nnc_test = confusion_matrix(y_test, pred_nnc_test) 

acs_nnc_train = accuracy_score(y_train, pred_nnc_train)
acs_nnc_test = accuracy_score(y_test, pred_nnc_test)

print("YSA Egitim dogruluk Skoru = ",acs_nnc_train)
print("YSA Test dogruluk Skoru = ",acs_nnc_test)


from sklearn.metrics import recall_score, precision_score

rs_nnc_train = recall_score(y_train, pred_nnc_train)
rs_nnc_test = recall_score(y_test, pred_nnc_test)

ps_nnc_train = precision_score(y_train, pred_nnc_train)
ps_nnc_test = precision_score(y_test, pred_nnc_test)

print("YSA Egitim Hassasiyet Skoru = ",rs_nnc_train)
print("YSA Test Hassasiyet Skoru = ",rs_nnc_test)

print("YSA Train Kesinlik Skoru = ",ps_nnc_train)
print("YSA Test Kesinlik Skoru = ",ps_nnc_test)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=12,criterion="entropy", min_samples_leaf=4) #criterion="entropy",min_samples_leaf=3
dtc.fit(x_train, y_train)

pred_dtc_train = dtc.predict(x_train)
pred_dtc_test = dtc.predict(x_test)

cm_dtc_train = confusion_matrix(y_train, pred_dtc_train)
#cm_dtc_test = confusion_matrix(y_train, pred_dtc_test)

acs_dtc_train = accuracy_score(y_train, pred_dtc_train)
acs_dtc_test = accuracy_score(y_test, pred_dtc_test)

print("Karar Agaci Dogruluk Skoru = ", acs_dtc_train)
print("Karar Agaci Test Dogruluk Skoru = ", acs_dtc_test)

from sklearn import tree
tree.export_graphviz(dtc, out_file="diag_tree.dot", feature_names= x_train.columns)

rs_dtc_train = recall_score(y_train, pred_dtc_train)
rs_dtc_test = recall_score(y_test, pred_dtc_test)

ps_dtc_train = precision_score(y_train, pred_dtc_train)
ps_dtc_test = precision_score(y_test, pred_dtc_test)

print("Karar Agaci Egitim Hassasiyet Skoru = ",rs_dtc_train)
print("Karar Agaci Test Hassasiyet Skoru = ",rs_dtc_test)

print("Karar Agaci Kesinlik Skoru = ", ps_dtc_train)
print("Karar Agaci Test Kesinlik Skoru = ", ps_dtc_test)

#logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=2000,solver="sag")
lr.fit(x_train, y_train)

pred_lr_train = lr.predict(x_train)
pred_lr_test = lr.predict(x_test)

cm_lr_train = confusion_matrix(y_train, pred_lr_train)
cm_lr_test = confusion_matrix(y_test, pred_lr_test)

acs_lr_train = accuracy_score(y_train, pred_lr_train)
acs_lr_test = accuracy_score(y_test, pred_lr_test)

rs_lr_train = recall_score(y_train, pred_lr_train)
rs_lr_test = recall_score(y_test, pred_lr_test)

ps_lr_train = precision_score(y_train, pred_lr_train)
ps_lr_test = precision_score(y_test, pred_lr_test)

print("Logistik Regression Train Dogruluk Skoru = ", acs_lr_train)
print("Logistik Regression Test Dogruluk Skoru = ", acs_lr_test)

print("Logistik Regression Train Hassasiyet Skoru = ", rs_lr_train)
print("Logistik Regression Test Hassasiyet Skoru = ", rs_lr_test)

print("Logistik Regression Train Kesinlik Skoru = ", ps_lr_train)
print("Logistik Regression Test Kesinlik Skoru = ", ps_lr_test)

#svm yontemi de kullanilabilir burda









