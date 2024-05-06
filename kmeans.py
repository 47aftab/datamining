import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = np.array([[1, 1], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(data)

print("Cluster centroids:")
print(kmeans.cluster_centers_)

print("Assigned cluster labels:")
print(kmeans.labels_)


# Plot data before clustering
plt.scatter(data[:, 0], data[:, 1], label='Data')
plt.title('Data before Clustering')
plt.legend()
plt.show()

# Plot data after clustering with centroids
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, label='Data')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=200, c='red', label='Centroids')
plt.title('Data after Clustering')
plt.legend()
plt.show()