import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

#load data
data = pd.read_csv('kmeans_data.txt', delimiter='  ', header=None)
x = data.values


#make cluster
fx = np.linalg.norm(x, axis=1)**2
fx = fx[:, np.newaxis]
u = fx[:2, :]
d = np.zeros_like(np.dot(fx, u.T))

diff = np.zeros((fx.shape[0], u.shape[0]))
for i in range(u.shape[0]):
    diff[:, i] = np.sum((fx - u[i, :])**2, axis=1)
d = diff
c = np.argmin(d, axis=1).reshape(-1, 1)

for iter in range(10):
    u = np.array([np.mean(fx[c == i], axis=0) for i in range(2)])
    diff = np.zeros((fx.shape[0], u.shape[0]))
    for i in range(u.shape[0]):
        diff[:, i] = np.sum((fx - u[i]) ** 2, axis=1)

    c = np.argmin(diff, axis=1).reshape(-1, 1)

p = np.squeeze(c == 1)
n = np.squeeze(c == 0)

plt.scatter(*x[c.flatten() == 1].T, c='b', label='Cluster 1')
plt.scatter(*x[c.flatten() == 0].T, c='g', label='Cluster 0')
plt.legend()
plt.show()