# Sample 2.5: Kernel SVM
from sklearn import svm

X = [[-1, 1], [0, 2], [1, 0], [-1, -1], [0, -2], [1, -1],
     [-1, 0], [1, 1], [0, 0], [0, 1], [0, -1], [-4, -4],
     [-4, -1], [-4, 2], [-3, -3], [-3, 0], [-3, 3], [-1, 4],
     [-1, -3], [0, 5], [0, -5], [1, 4], [1, -4], [3, -4],
     [3, -1], [3, 2], [4, -3], [4, 0], [4, 3]]

y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

clf = svm.SVC(kernel='rbf')
clf.fit(X, y)
print("Accuracy with RBF Kernel: %.2f%s" %
      (clf.score(X, y) * 100, '%'))

clf = svm.SVC(kernel='sigmoid')
clf.fit(X, y)
print("Accuracy with Sigmoid Kernel: %.2f%s" %
      (clf.score(X, y) * 100, '%'))
