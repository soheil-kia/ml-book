# Sample 4.6: GMM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture

X, y = make_blobs(n_samples=150, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X = np.dot(X, transformation)[:, ::-1]

markers = {0: '>', 1: 's', 2: 'o'}
colors = {0: 'r', 1: 'g', 2: 'b'}

gmm = GaussianMixture(n_components=3)
labels = gmm.fit(X).predict(X)
for (i, l) in zip(X, labels):
    plt.scatter(i[0], i[1], marker=markers[l], color=colors[l])

plt.xticks([])
plt.yticks([])
plt.show()
