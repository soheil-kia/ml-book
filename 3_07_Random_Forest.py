# Sample 3.7: Random Forest
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('pls.txt', header=None)

X = data.values[:, :-1]
y = data.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

forest = RandomForestClassifier()

forest.fit(X_train, y_train)

print("Random Forest Accuracy: %.2f%s" %
      (forest.score(X_test, y_test) * 100, '%'))
