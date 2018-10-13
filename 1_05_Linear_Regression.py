# Sample 1.5: Diabetes Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2]

X_train = X[:-20]
X_test = X[-20:]

y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, y_train)

y_predict = linear_regression.predict(X_test)

# The coefficients
print('Coefficients: \n', linear_regression.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_predict))

# Plot outputs
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
