# Sample 4.7:BIRCH
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch

X, y = make_blobs(n_samples=150, random_state=36)

birch = Birch(n_clusters=3)
birch.fit(X)
labels = birch.labels_

markers = {0: '>', 1: 's', 2: 'o'}
colors = {0: 'r', 1: 'g', 2: 'b'}

for (i, l) in zip(X, labels):
    plt.scatter(i[0], i[1], marker=markers[l], color=colors[l])

plt.xticks([])
plt.yticks([])
plt.show()
