# Sample 4.4: Agglomerative Clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv('clustering.txt', header=None)
data = df.values
Agglomerative = AgglomerativeClustering(
    n_clusters=4, affinity='euclidean', linkage='ward')
df[2] = Agglomerative.fit_predict(data)
for i in df.values:
    if i[2] == 0:
        plt.scatter(i[0], i[1], marker='o', c='b')
    elif i[2] == 1:
        plt.scatter(i[0], i[1], marker='d', c='r')
    elif i[2] == 2:
        plt.scatter(i[0], i[1], marker='s', c='g')
    elif i[2] == 3:
        plt.scatter(i[0], i[1], marker='>', c='m')
plt.xticks([])
plt.yticks([])
plt.show()
