
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

k_best_features = SelectKBest(score_func=chi2, k=5)
X_new = k_best_features.fit_transform(X, y)

# Get the indices of the selected features
selected_indices = np.where(k_best_features.get_support())[0]

# Print the names of the selected features
selected_features_names = data.feature_names[selected_indices]

print("Selected Features:")
for feature in selected_features_names:
    print(feature)