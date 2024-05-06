#K-means

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

#DBSCAN

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=7, min_samples=5)
labels = dbscan.fit_predict(data)
np.unique(labels)

#Agglomerative

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
labels_=cluster.fit_predict(data)


#Decision Tree
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_iris, y_train_iris)
y_pred_iris = clf.predict(X_test_iris)
accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)