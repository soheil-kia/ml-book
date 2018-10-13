# Sample 3.3: Bagging
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('wifi.txt', header=None)

X = data.values[:, :-1]
y = data.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

bagging = BaggingClassifier()
svm = svm.SVC()
logreg = LogisticRegression()

bagging.fit(X_train, y_train)
svm.fit(X_train, y_train)
logreg.fit(X_train, y_train)

print("Bagging Accuracy: %.2f%s" %
      (bagging.score(X_test, y_test) * 100, '%'))
print("SVM Accuracy: %.2f%s" %
      (svm.score(X_test, y_test) * 100, '%'))
print("LogReg Accuracy: %.2f%s" %
      (logreg.score(X_test, y_test) * 100, '%'))
