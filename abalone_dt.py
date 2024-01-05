# -*- coding: utf-8 -*-
"""
Created on Wed May 24 23:14:23 2023

@author: Christina
"""

import pandas as pd

feature_names = ["Sex", "Length", "Diameter", "Height", "Whole_Weight", \
                 "Shucked_Weight", "Viscera_Weight", "Shell_Weight", "Rings"]

data = pd.read_csv("abalone.data", names=feature_names)

inputs1 = data.iloc[:, 1:-1:]
sex = pd.get_dummies(data.iloc[:, 0:1], dtype=int)

inputs = pd.concat([sex, inputs1], axis=1)

target = data.iloc[:, -1:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.25, random_state=0)

#decision tree classifier

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(min_samples_leaf=3, splitter="best", max_depth=2, criterion="log_loss") #max_depth=2 to decrease the size of tree

dtc.fit(x_train, y_train)

prediction_train = dtc.predict(x_train)
prediction_test = dtc.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm_train = confusion_matrix(y_train, prediction_train)
cm_test = confusion_matrix(y_test, prediction_test)

acs_train = accuracy_score(y_train, prediction_train)
acs_test = accuracy_score(y_test, prediction_test)

#decision tree regressor

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(max_depth=2, criterion="poisson")

dtr.fit(x_train, y_train)

prediction2_train = dtr.predict(x_train)
prediction2_test = dtr.predict(x_test)

#ağaç gösterleşilmesi

from sklearn import tree

tree.export_graphviz(dtc, out_file="classifier_tree.dot", feature_names=x_train.columns)

tree.export_graphviz(dtr, out_file="regressor_tree.dot", feature_names=x_train.columns) 

#Metrikleri

from sklearn.metrics import mean_absolute_error, mean_squared_error 

mae_classifier_train = mean_absolute_error(y_train, prediction_train)
mae_classifier_test = mean_absolute_error(y_test, prediction_test)

mse_classifier_train = mean_squared_error(y_train, prediction_train)
mse_classifier_test = mean_squared_error(y_test, prediction_test)

mae_regressor_train = mean_absolute_error(y_train, prediction2_train)
mae_regressor_test = mean_absolute_error(y_test, prediction2_test)

mse_regressor_train = mean_squared_error(y_train, prediction2_train)
mse_regressor_test = mean_squared_error(y_test, prediction2_test)

print("MAE Classifier Train = ", mae_classifier_train)
print("MAE Classifier Test = ", mae_classifier_test)
print("MSE Classifier Train = ", mse_classifier_train)
print("MSE Classifier Test", mse_classifier_test)
print("MAE Regressor Train = ", mae_regressor_train)
print("MAE Regressor Test", mae_regressor_test)
print("MSE Regressor Train = ", mse_regressor_train)
print("MSE Regressor Test", mse_regressor_test)




