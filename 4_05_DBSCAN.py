# Sample 4.5: DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN, KMeans

X, y = datasets.make_moons(
    n_samples=200,
    random_state=100,
    noise=.05)
df = pd.DataFrame(X)

dbscan = DBSCAN(eps=.3, min_samples=3)
dbscan.fit(X)
df[2] = dbscan.labels_

kmeans = KMeans(n_clusters=2)
kmeans.fit(np.array(df.iloc[:, 0:2]))
df[3] = kmeans.labels_

fig, ax = plt.subplots(2, sharex=True)

for i in df.values:
    if i[2] == 1:
        ax[0].scatter(i[0], i[1],
                      color='red', marker='o')
    else:
        ax[0].scatter(i[0], i[1],
                      color='blue', marker='>')
    if i[3] == 1:
        ax[1].scatter(i[0], i[1],
                      color='red', marker='o')
    else:
        ax[1].scatter(i[0], i[1],
                      color='blue', marker='>')

ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.show()
