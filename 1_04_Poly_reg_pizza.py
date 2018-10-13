# Sample 1.4: Pizza Franchise Fee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = pd.read_fwf('pizza.txt', header=None)
X = data.values[:, 0].reshape(len(data.values[:, 0]), 1)
y = data.values[:, 1].reshape(len(data.values[:, 1]), 1)

# Make model and fit
degree = 2
linear_regression = make_pipeline(PolynomialFeatures(degree),
                                  LinearRegression())
linear_regression.fit(X, y)

# Calculate output
print("Annual Income for 1225$: %.2f"
      % linear_regression.predict(1225))
print("Annual Income for  700$: %.2f"
      % linear_regression.predict(700))

# Plot outputs
plt.scatter(X, y, color='black')
X_plot = np.array(range(int(X.min()), int(X.max()) + 2)).reshape(
    int(X.max() + 2) - int(X.min()), 1
)
y_predict = linear_regression.predict(X_plot)
plt.plot(X_plot, y_predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
