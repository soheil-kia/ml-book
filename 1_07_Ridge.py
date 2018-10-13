# Sample 1.7: Diabetes Ridge Regression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge

diabetes = datasets.load_diabetes()
X = diabetes.data
for i in X:
    i[8] = 1000 * i[8]
    i[9] = 1000 * i[9]

X_train = X[:-20, :]
X_test = X[-20:, :]

y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]

degree = 2
linear_regression = make_pipeline(PolynomialFeatures(degree),
                                  LinearRegression())
linear_regression.fit(X_train, y_train)
y_predict = linear_regression.predict(X_test)
print("Linear Regression MSE: %.2f"
      % mean_squared_error(y_test, y_predict))

ridge_regression = make_pipeline(PolynomialFeatures(degree),
                                 Ridge(alpha=1e-3))
ridge_regression.fit(X_train, y_train)
ridge_predict = ridge_regression.predict(X_test)
print("Ridge Regression MSE: %.2f"
      % mean_squared_error(y_test, ridge_predict))
