# Sample 4.1: KMeans
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

random.seed(100)
X1 = [random.randint(1, 35) for x in range(20)]
y1 = [random.randint(1, 50) for x in range(20)]

random.seed(2000)
X2 = [random.randint(1, 50) + 300 for x in range(20)]
y2 = [random.randint(1, 50) for x in range(20)]

random.seed(1500)
X3 = [random.randint(1, 50) + 150 for x in range(15)]
y3 = [random.randint(1, 50) + 150 for x in range(15)]

random.seed(20)
X4 = [random.randint(1, 50) + 150 for x in range(18)]
y4 = [random.randint(1, 50) for x in range(18)]

X = X1 + X2 + X3 + X4
Y = y1 + y2 + y3 + y4

data = np.array([[x, y] for (x, y) in zip(X, Y)])

fig, ax = plt.subplots(2, sharex=True)

kmeans = KMeans(n_clusters=4)
kmeans.fit(data)

for i in data:
    ax[0].scatter(i[0], i[1], marker='o',
                  facecolors='none', edgecolors='black')
    if kmeans.predict(i)[0] == 0:
        ax[1].scatter(i[0], i[1], marker='o', c='b')
    elif kmeans.predict(i)[0] == 1:
        ax[1].scatter(i[0], i[1], marker='d', c='r')
    elif kmeans.predict(i)[0] == 2:
        ax[1].scatter(i[0], i[1], marker='s', c='g')
    elif kmeans.predict(i)[0] == 3:
        ax[1].scatter(i[0], i[1], marker='>', c='m')
        
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.show()
