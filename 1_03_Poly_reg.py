# Sample 1.3: Polynomial Regression over y=x^3+2x^2+3
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X = [x for x in range(20)]
X = np.array(X).reshape(20, 1)

y = [x ** 3 + 2 * (x ** 2) + 3 for x in range(20)]
y = np.array(y).reshape(20, 1)

# Make model and fit
degree = 3
linear_regression = make_pipeline(PolynomialFeatures(degree),
                                  LinearRegression())
linear_regression.fit(X, y)

y_predict = linear_regression.predict(X)

# Plot outputs
plt.scatter(X, y, color='black')
plt.plot(X, y_predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
