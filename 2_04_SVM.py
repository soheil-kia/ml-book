# Sample 2.4: SVM
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

data = pd.read_csv('wifi.txt', header=None)

X = data.values[:, :-1]
y = data.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)
print("Accuracy on test set: %.2f%s" %
      (clf.score(X_test, y_test) * 100, '%'))
