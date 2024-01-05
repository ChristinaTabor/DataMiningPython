# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:52:13 2023

@author: Christina
"""

import pandas as pd

feature_names = ["Sex", "Length", "Diameter", "Height", "Whole_Weight", \
                 "Shucked_Weight", "Viscera_Weight", "Shell_Weight", "Rings"]

data = pd.read_csv("abalone.data", names=feature_names)

inputs1 = data.iloc[:, 1:-1:]
sex = pd.get_dummies(data.iloc[:, 0:1], dtype=int)

inputs = pd.concat([sex, inputs1], axis=1) #axis=1 sutunlar duzeyinde bir birlestirme yapar

target = data.iloc[:, -1:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.25, random_state=0)

from sklearn.neural_network import MLPRegressor

nnr = MLPRegressor(activation="relu", solver="adam", random_state=24)
                   #learning_rate_init=0.0001, max_iter=900) 
#randomState, ann baslangic agirliklarini rassal olarak atilmasini belirleyen

#nnr = MLPRegressor(activation="relu", solver="lbfgs", random_state=24) #bu parametrelerle daha iyi bir deger elde ettigimizi goruyoruz

#10 tane input layer neuronlerini icerir bu model

nnr.fit(x_train, y_train)

"""
#Evaluation

prediction_test = pd.DataFrame(nnr.predict(x_test))
prediction_train = pd.DataFrame(nnr.predict(x_train))

prediction_test_int = prediction_test.round(0)
prediction_train_int = prediction_train.round(0)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_test = mean_absolute_error(y_test, prediction_test_int)
mae_train = mean_absolute_error(y_train, prediction_train_int)

mae_test_2 = mean_absolute_error(y_test, prediction_test)
mae_train_2 = mean_absolute_error(y_train, prediction_train)

mse_test = mean_squared_error(y_test, prediction_test_int)
mse_train = mean_squared_error(y_train, prediction_train_int)

mse_test_2 = mean_squared_error(y_test, prediction_test)
mse_train_2 = mean_squared_error(y_train, prediction_train)

import numpy as np

#mean_y_test = np.mean(y_test)
#std_y_test = np.std(y_test)

mean_ring = np.mean(target) #aritmetik ort
std_ring = np.std(target)


print("Arithmetic Mean for No of Rings  = ", mean_ring)
print("Std Deviation for No of Rings = ", std_ring)

print("Integer Tahminlere Gore: ")

print("Train Bolumu MAE = ", mae_train)
print("Test Bolumu MAE = ", mae_test)
print("Train Bolumu MSE = ", mse_train)
print("Test Bolumu MSE = ", mse_test)
print("Train Bolumu RMSE = ", np.sqrt(mse_train))
print("Test Bolumu RMSE = ", np.sqrt(mse_test))

print("Ondalikli Tahminlere Gore: ")

print("Train Bolumu MAE 2 = ", mae_train_2)
print("Test Bolumu MAE 2 = ", mae_test_2)
print("Train Bolumu MSE 2 = ", mse_train_2)
print("Test Bolumu MSE 2 = ", mse_test_2)
print("Train Bolumu RMSE 2 = ", np.sqrt(mse_train_2))
print("Test Bolumu RMSE 2 = ", np.sqrt(mse_test_2))
"""

#Deployment

inputs_deploy = pd.DataFrame(index=[0], columns=["Sex_F","Sex_I","Sex_M" "Length", "Diameter", "Height", "Whole_Weight", \
                 "Shucked_Weight", "Viscera_Weight", "Shell_Weight"])

inputs_deploy.at[0, "Sex_F"] = input("Sex F (1/0) : ")
inputs_deploy.at[0, "Sex_I"] = input("Sex I (1/0) : ")   
inputs_deploy.at[0, "Sex_M"] = input("Sex M (1/0) : ")   
inputs_deploy.at[0, "Length"] = input("Length : ")
inputs_deploy.at[0, "Diameter"] = input("Diameter : ")
inputs_deploy.at[0, "Height"] = input("Height : ")
inputs_deploy.at[0, "Whole_Weight"] = input("Whole_Weight : ")
inputs_deploy.at[0, "Shucked_Weight"] = input("Shucked_Weight : ")
inputs_deploy.at[0, "Viscera_Weight"] = input("Viscera_Weight : ")
inputs_deploy.at[0, "Shell_Weight"] = input("Shell_Weight : ")
    
tahmin = nnr.predict(inputs_deploy)    
    
print("No of Rings prediction = ", tahmin)    
    
    
    
    
    








