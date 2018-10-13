# Sample 1.6: Diabetes Linear Regression (Multi-Valued)
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
X = diabetes.data

X_train = X[:-20, :]
X_test = X[-20:, :]

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
