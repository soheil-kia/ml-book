# Sample 2.3: Logistic Regression(Softmax)
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data = pd.read_csv('glass.txt', header=None)

X = data.values[:, :-1]
y = data.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

logreg = linear_model.LogisticRegression(
    multi_class='multinomial', solver='lbfgs')
logreg.fit(X_train, y_train)
print("Accuracy on test set: %.2f%s" %
      (logreg.score(X_test, y_test) * 100, '%'))
