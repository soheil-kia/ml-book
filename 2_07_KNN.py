# Sample 2.7: KNN Classifier
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = neighbors.KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train, y_train)
print("Accuracy on test set: %.2f%s" %
      (clf.score(X_test, y_test) * 100, '%'))
