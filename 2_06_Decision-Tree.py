# Sample 2.6: Decision Tree Classifier
import pandas as pd
from sklearn import svm, tree
from sklearn.model_selection import train_test_split

data = pd.read_csv('wifi.txt', header=None)

X = data.values[:, :-1]
y = data.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svmclf = svm.SVC()
treeclf = tree.DecisionTreeClassifier()

svmclf.fit(X_train, y_train)
treeclf.fit(X_train, y_train)

print("SVM Accuracy: %.2f%s" %
      (svmclf.score(X_test, y_test) * 100, '%'))
print("Tree Accuracy: %.2f%s" %
      (treeclf.score(X_test, y_test) * 100, '%'))