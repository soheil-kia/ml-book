# Sample 1.2: Linear Regression over y=x^3+2x^2+3
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

X = [x for x in range(20)]
X = np.array(X).reshape(20, 1)

y = [x ** 3 + 2 * (x ** 2) + 3 for x in range(20)]
y = np.array(y).reshape(20, 1)

# Make model and fit
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X, y)

y_predict = linear_regression.predict(X)

# Plot outputs
plt.scatter(X, y, color='black')
plt.plot(X, y_predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
