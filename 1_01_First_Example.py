# Sample 1.1: Simple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Making arbitrary dataset
X = [x for x in range(10)]
X = np.array(X).reshape(10, 1)

y = [2 * x + 1 for x in range(10)]
y = np.array(y).reshape(10, 1)

# Make model and fit
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X, y)

y_predict = linear_regression.predict(X)

# The coefficients
print('Coefficient: \n', linear_regression.coef_)
print('Intercept: \n', linear_regression.intercept_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y, y_predict))

# Plot outputs
plt.scatter(X, y, color='black')
plt.plot(X, y_predict, color='blue', linewidth=3)

plt.xticks(range(10))
plt.yticks(range(1, 21, 2))

plt.show()
