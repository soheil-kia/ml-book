# Sample 2.8: Naive Bayes Classifier
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split

data = pd.read_csv('pima-indians-diabetes.txt', header=None)

X = data.values[:, :-1]
y = data.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

models = (GaussianNB(), MultinomialNB(), BernoulliNB())
names = ['Gaussian', 'Multinomial', 'Bernoulli']

for clf, name in zip(models, names):
    clf.fit(X_train, y_train)
    print("%s Naive Bayes accuracy on test set: %.2f%s"
          % (name, clf.score(X_test, y_test) * 100, '%'))
