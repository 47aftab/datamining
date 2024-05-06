#Code for cross validation

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score

iris = datasets.load_iris()

clf = DecisionTreeClassifier(random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

iris_accuracies = cross_val_score(clf, iris.data, iris.target, cv=kf, scoring='accuracy')

print(iris_accuracies)
