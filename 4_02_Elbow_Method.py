# Sample 4.2: Elbow Method
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

result = [None]
for n in range(1, 10):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(list(zip(X, Y)))
    sum = 0
    for x, y in zip(X, Y):
        sum += (kmeans.cluster_centers_
                [kmeans.predict([[x, y]])[0]][0] - x) ** 2
        +(kmeans.cluster_centers_
          [kmeans.predict([[x, y]])[0]][1] - y) ** 2
    result.append(sum)

plt.yticks([])
plt.plot(result)
plt.show()
