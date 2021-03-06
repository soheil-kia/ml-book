# Sample 2.1: Logistic Regression
import pandas as pd
import numpy as np
from sklearn import linear_model

data = pd.read_fwf('apple_juice.txt', header=None)
X = data.values[:, :-1]
y = data.values[:, -1]

X_train = X[:-15, :]
X_test = X[-15:, :]

y_train = y[:-15]
y_test = y[-15:]

logreg = linear_model.LogisticRegression()
logreg.fit(X_train, y_train)
print("Accuracy on test set: %.2f%s" %
      (logreg.score(X_test, y_test) * 100, '%'))
