# Sample 4.3: Mean-Shift
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from PIL import Image

image = Image.open('mshift.png')
image = np.array(image)

data = np.reshape(image, [-1, 4])
meanshift = MeanShift()
meanshift.fit(data)

labels = meanshift.labels_

plt.figure(2)

plt.subplot(2, 1, 1)
plt.subplot(2, 1, 1).set_xticks([])
plt.subplot(2, 1, 1).set_yticks([])
plt.imshow(image)
plt.subplot(2, 1, 2)
plt.subplot(2, 1, 2).set_xticks([])
plt.subplot(2, 1, 2).set_yticks([])
plt.imshow(np.reshape(labels,
                      [image.shape[0], image.shape[1]]))

plt.show()
