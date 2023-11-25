import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

#load data
data = pd.read_csv('kmeans_data.txt', delimiter='  ', header=None)
x = data.values

for iter in range(10):
    z = (np.random.randint(250, size=1)).reshape(())
    squared_diff = np.sum(np.square(x - x[z, :]), axis=1)
    exponential_term = np.exp(-0.1 * squared_diff)
    fx = exponential_term.reshape(-1, 1)

    u = fx[:2,:]
    d = np.zeros_like(np.dot(fx, u.T))
    diff = fx[:, np.newaxis, :] - u
    d = np.sum(np.square(diff), axis=2)
    c = np.argmin(d, axis=1).reshape(-1, 1)

    u = np.zeros((2, fx.shape[1]))
    u[0, :] = np.mean(fx[c == 0], axis=0)
    u[1, :] = np.mean(fx[c == 1], axis=0)

    u_reshaped = u.reshape(1, u.shape[0], -1)
    diff = fx[:, np.newaxis, :] - u_reshaped
    d = np.sum(np.square(diff), axis=2)
    c = np.argmin(d, axis=1).reshape(-1, 1)

    p = np.squeeze(c == 1)
    n = np.squeeze(c == 0)

    plt.figure(iter)
    plt.scatter(*x[c.flatten() == 1].T, c='b', label='Cluster 1')
    plt.scatter(*x[c.flatten() == 0].T, c='g', label='Cluster 0')
    plt.legend()
    plt.plot(x[z,0], x[z,1], 'r*')

plt.show()