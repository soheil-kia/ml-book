# Sample 3.8: Diabetes Ensemble Regression
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

bagreg = BaggingRegressor()
bagreg.fit(X_train, y_train)
bagreg_predict = bagreg.predict(X_test)
print("Bagging Mean squared error: %.2f"
      % mean_squared_error(y_test, bagreg_predict))

rfreg = RandomForestRegressor()
rfreg.fit(X_train, y_train)
rfreg_predict = rfreg.predict(X_test)
print("Random Forest Mean squared error: %.2f"
      % mean_squared_error(y_test, rfreg_predict))
