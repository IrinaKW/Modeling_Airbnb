#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt 
from tabular_data import load_airbnb

df=pd.read_csv('tabular_data/clean_tabular_data.csv')
X, y = load_airbnb(df, 'Price_Night')
X = scale(X)
y = scale(y)
xtrain, xtest, ytrain, ytest=train_test_split(X, y, test_size=0.15)

model_sgdr = SGDRegressor()
model_sgdr.fit(xtrain, ytrain)

#train set
ypred = model_sgdr.predict(xtrain)
mse = mean_squared_error(ytrain, ypred)
score = model_sgdr.score(xtrain, ytrain)
print ('Train Set')
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0))
print("R-squared: ", score)

#test set
ypred = model_sgdr.predict(xtest)
mse = mean_squared_error(ytest, ypred)
score = model_sgdr.score(xtest, ytest)
print ('\nTest Set')
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0))
print("R-squared: ", score)

x_ax = range(len(ytest))
plt.plot(x_ax, ytest, label="original")
plt.plot(x_ax, ypred, label="predicted")
plt.title("Test set and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()

# %%
