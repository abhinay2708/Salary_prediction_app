import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"C:\Users\abhin\Downloads\Salary_Data.csv")
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.7,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

comparison=pd.DataFrame({'Actual':y_test,'Prdicted':y_pred})
print(comparison)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experince (Test Set)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()

m_slope=regressor.coef_
print(m_slope)

c_intercept=regressor.intercept_
print(c_intercept)

y_12=(m_slope*12)+c_intercept
print(y_12)

y_20=(m_slope*20)+c_intercept
print(y_20)

dataset.mean()
dataset['Salary'].mean()

dataset.median()
dataset['Salary'].median()

dataset.mode()
dataset['Salary'].mode()

dataset.var()
dataset['Salary'].var()

dataset.std()
dataset['Salary'].std()

# coefficient of vaiation
from scipy.stats import variation
variation(dataset.values)

dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])

dataset.skew()
dataset['Salary'].skew()

dataset.sem()
dataset['Salary'].sem()
 # statistics library
import scipy.stats as stats
dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])
#Degree of freedom
a=dataset.shape[0] #Number of rows
b=dataset.shape[1] #Number of columns

degree_of_freedom=a-b
print(degree_of_freedom)

y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean))**2
print(SSR)

#Sum of square Error(SSE)
y=y[0:6]
y_pred=y_pred[:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

SST=SSR+SSE

r_square=1-(SSR/SST)
r_square

bias=regressor.score(x_train,y_train)
print(bias)

variance=regressor.score(x_test,y_test)
print(variance)

from sklearn.metrics import mean_squared_error
train_mse=mean_squared_error(y_train,regressor.predict(x_train))
y_test=y_test[:6]
test_mse=mean_squared_error(y_test,y_pred)

import pickle

filename='linear_regression_model.pkl'

with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print("Model has been pickled and saved as linear_regression_model.pickle")

import os
os.getcwd()











