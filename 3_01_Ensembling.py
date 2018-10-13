# Sample 3.1: Ensembling
import pandas as pd
from sklearn import svm, tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

data = pd.read_csv('wifi.txt', header=None)

X = data.values[:, :-1]
y = data.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

svm = svm.SVC(probability=True)
logreg = LogisticRegression()
tree = tree.DecisionTreeClassifier()

svm.fit(X_train, y_train)
logreg.fit(X_train, y_train)
tree.fit(X_train, y_train)

print("SVM Accuracy: %.2f%s" %
      (svm.score(X_test, y_test) * 100, '%'))
print("LogReg Accuracy: %.2f%s" %
      (logreg.score(X_test, y_test) * 100, '%'))
print("Tree Accuracy: %.2f%s" %
      (tree.score(X_test, y_test) * 100, '%'))

w = [1, 1, 1]
ensemble = VotingClassifier(
    estimators=[('svm', svm),
                ('logreg', logreg),
                ('tree', tree)],
    voting='hard', weights=w)
ensemble.fit(X_train, y_train)

print("Ensemble Accuracy: %.2f%s" %
      (ensemble.score(X_test, y_test) * 100, '%'))
