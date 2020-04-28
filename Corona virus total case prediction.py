# Corona Virus Epedemic

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('total_cases.csv')

X = dataset.iloc[85:,0:1].values
y = dataset.iloc[85:,2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Coronavirus Total Cases')
plt.xlabel('No of days')
plt.ylabel('Total Cases')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Coronavirus Total Cases')
plt.xlabel('No of days')
plt.ylabel('Total Cases')
plt.show()

#Predicting the Linear Regression Results
lin_reg.predict([[120]])

#Predicting the Polynomial Regression Results
lin_reg_2.predict(poly_reg.fit_transform([[192]]))
