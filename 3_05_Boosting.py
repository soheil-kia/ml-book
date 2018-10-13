# Sample 3.5: Boosting
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

data = pd.read_csv('solution.txt', header=None)

X = data.values[:, :-1]
y = data.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

boosting = AdaBoostClassifier()

boosting.fit(X_train, y_train)

print("Boosting Accuracy: %.2f%s" %
      (boosting.score(X_test, y_test) * 100, '%'))
