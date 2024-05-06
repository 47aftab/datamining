from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris_data = load_iris()
X_iris, y_iris = iris_data.data, iris_data.target
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Load the Wine dataset
wine_data = load_wine()
X_wine, y_wine = wine_data.data, wine_data.target
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# K-means clustering on Iris dataset
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_iris_train)
iris_kmeans_pred = kmeans.predict(X_iris_test)

# DBSCAN clustering on Wine dataset
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_wine_train)
wine_dbscan_pred = dbscan.fit_predict(X_wine_test)

# Agglomerative clustering on Iris dataset
agg = AgglomerativeClustering(n_clusters=3)
agg.fit(X_iris_train)
iris_agg_pred = agg.fit_predict(X_iris_test)

# Decision Tree classifier on Wine dataset
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_wine_train, y_wine_train)
wine_dt_pred = dt.predict(X_wine_test)

# K-Nearest Neighbors classifier on Iris dataset
knn = KNeighborsClassifier()
knn.fit(X_iris_train, y_iris_train)
iris_knn_pred = knn.predict(X_iris_test)

# Naive Bayes classifier on Wine dataset
nb = GaussianNB()
nb.fit(X_wine_train, y_wine_train)
wine_nb_pred = nb.predict(X_wine_test)

# Calculate accuracy scores
print("Accuracy scores:")
print("Iris K-means:", accuracy_score(y_iris_test, iris_kmeans_pred))
print("Wine DBSCAN:", accuracy_score(y_wine_test, wine_dbscan_pred))
print("Iris Agglomerative:", accuracy_score(y_iris_test, iris_agg_pred))
print("Wine Decision Tree:", accuracy_score(y_wine_test, wine_dt_pred))
print("Iris K-Nearest Neighbors:", accuracy_score(y_iris_test, iris_knn_pred))
print("Wine Naive Bayes:", accuracy_score(y_wine_test, wine_nb_pred))
