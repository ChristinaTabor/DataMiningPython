# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:01:03 2023

@author: Christina
"""

# Bussiness Understanding(işin anlaşılması): Çiplerin nisbi performanslarını tahmin edilmesi

# Data Understanding(veri anlaşılması): machine.names dosyasında dokümante edilmiş

# Data preparation(veri hazırlanması): 

import pandas as pd

data = pd.read_csv("machine.data", names=["Vendor_Name", "Model_Name", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"])

vendor = pd.get_dummies(data["Vendor_Name"], dtype=int)

numeric_data = data.iloc[:, 2:-2:]

inputs = numeric_data

target = data["PRP"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.25, random_state=9)


# Modeling:

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR #support vector Regressor

nnr = MLPRegressor(max_iter=2000, learning_rate_init=0.0005)
dvr = SVR(kernel="sigmoid")

nnr.fit(x_train, y_train)
dvr.fit(x_train, y_train)

# Evaluation (değerlendırme):

pred_nnr_train = nnr.predict(x_train)
pred_nnr_test = nnr.predict(x_test)
pred_dvr_train = dvr.predict(x_train)
pred_dvr_test = dvr.predict(x_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np

mae_nnr_train = mean_absolute_error(y_train, pred_nnr_train)
mae_nnr_test = mean_absolute_error(y_test, pred_nnr_test)
rmse_nnr_train = np.sqrt(mean_squared_error(y_train, pred_nnr_train))
rmse_nnr_train = np.sqrt(mean_squared_error(y_test, pred_nnr_test))

mae_dvr_train = mean_absolute_error(y_train, pred_dvr_train)
mae_dvr_test = mean_absolute_error(y_test, pred_dvr_test)
rmse_dvr_train = np.sqrt(mean_squared_error(y_train, pred_dvr_train))
rmse_dvr_train = np.sqrt(mean_squared_error(y_test, pred_dvr_test))

mae_erp = mean_absolute_error(target, data["ERP"])
mse_erp = mean_squared_error(target, data["ERP"])
















