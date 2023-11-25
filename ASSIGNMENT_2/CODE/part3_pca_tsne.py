import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the data from the pickle file
file = open('mnist_small.pkl', 'rb')
data = pickle.load(file)
file.close()

x_data = data['X']
y_data = data['Y']

s_data = StandardScaler().fit_transform(x_data)
# PCA Visualization
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x_data)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
x_values = principalComponents[:, 0]
y_values = principalComponents[:, 1]
point_size = 5
colors = y_data
color_map = 'Spectral'
plt.scatter(x_values, y_values, s=point_size, c=colors, cmap=color_map)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('MNIST through PCA', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# t-SNE Visualization
s_data = StandardScaler().fit_transform(x_data)
tsne = TSNE(
    random_state=42,
    n_components=2,
    verbose=0,
    perplexity=40,
    n_iter=300
)
tsne= tsne.fit_transform(x_data)

plt.subplot(1, 2, 2)
x_values = tsne[:, 0]
y_values = tsne[:, 1]
point_size = 5
colors = y_data
color_map = 'Spectral'
plt.scatter(x_values, y_values, s=point_size, c=colors, cmap=color_map)

plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('MNIST through t-SNE', fontsize=14)
plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.tight_layout()
plt.show()
