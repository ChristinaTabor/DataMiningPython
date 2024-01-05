# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:27:32 2023

@author: Christina
"""

import pandas as pd

data = pd.read_csv("car.data", names=["buying_price", "maint_costs", "doors","persons","lug_boot","safety","acceptability"])

inputs = data.iloc[:, 0:6]
target_raw = data.iloc[:, -1:]

ind = 0       
a = target_raw.values

for i in a:
    if a[ind] != "unacc":
        a[ind] = "1"
    else:
        a[ind] = "0"      
    ind += 1 
target = pd.DataFrame(a, columns=["acceptability"]) 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.30, random_state=5)

#modeleme
x_train2 = pd.get_dummies(x_train)
x_test2 = pd.get_dummies(x_test)

   
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()

bnb.fit(x_train2, y_train)

#tahmin
pred_train = bnb.predict(x_train2)

pred_test = bnb.predict(x_test2)

from sklearn.metrics import accuracy_score

acs_train = accuracy_score(y_train,pred_train)
acs_test = accuracy_score(y_test,pred_test)

print("Eğitim seti doğruluk skoru = ", acs_train)
print("Test seti doğruluk skoru = ", acs_test)

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ps_train = precision_score(y_train, pred_train, pos_label="1")
ps_test = precision_score(y_test, pred_test, pos_label="1")

rs_train = recall_score(y_train, pred_train, pos_label="1")
rs_test = recall_score(y_test, pred_test, pos_label="1")

print("Egitim seti kesinlik degeri = ", ps_train)
print("Test seti kesinlik degeri = ", ps_test)

print("Egitim seti hassasiyet degeri = ", rs_train)
print("Test seti hassasiyet degeri = ", rs_test)

f1_train = f1_score(y_train, pred_train, pos_label="1")
f1_test = f1_score(y_test, pred_test, pos_label="1")

cm_train = confusion_matrix(y_train, pred_train)
cm_test = confusion_matrix(y_test, pred_test)

print("Egitim seti f1 skoru = ", f1_train)
print("Test seti f1 skoru = ", f1_test)

precision_train = cm_train[1][1] / (cm_train[1][1] + cm_train[0][1])
print("Egitim seti precision score = ", precision_train)

precision_test = cm_test[1][1] / (cm_test[1][1] + cm_test[0][1])
print("Test seti precision score = ", precision_test)

recall_train = cm_train[1][1] / (cm_train[1][1] + cm_train[1][0])
print("Egitim seti recall score = ", recall_train)

recall_test = cm_test[1][1] / (cm_test[1][1] + cm_test[1][0])
print("Test seti recall score = ", recall_test)

specificitiy_train = cm_train[0][0] / (cm_train[1][0] + cm_train[0][1])
print("Egitim seti specificitiy score = ", specificitiy_train)

specificitiy_test = cm_test[0][0] / (cm_test[0][0] + cm_test[0][1])
print("Test seti specificitiy score = ", specificitiy_test)

npv_train = cm_train[0][0] / cm_train[0][0] + cm_train[1][0]
print("Egitim seti npv score = ", npv_train)

npv_test = cm_test[0][0] / cm_test[0][0] + cm_test[1][0]
print("Test seti npv score = ",npv_test)






