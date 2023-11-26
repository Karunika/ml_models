from DecisionTree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np

class RandomForest():
    def __init__(self, num_trees=10, min_samples_split=2, max_depth=8):
        self.num_trees = nums_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        self.trees = trees


    def _bootstrap_dataset(self, X, y):
        num_samples, _ = X.shape
        idxs = np.random.choice(num_samples, num_samples, replace=True)
        return X.iloc[idxs], y.iloc[idxs]


class RandomForestClassfier(RandomForest):
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.num_trees):
            X_caret, y_caret = self._bootstrap_dataset(X, y)
            decision_tree_model = DecisionTreeClassifier()
            decision_tree_model.fit(X_caret, y_caret)

            self.tree.append(decision_tree_model)


    def predict(self, X):
        size = X.shape[0]
        arr = np.zeros(size)
        for tree in self.trees:
            predictions = tree.predict(X)
            for i in range(size):
                arr[i] += predictions[i]/self.num_trees
        return arr


class RandomForestRegressor(RandomForest):
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.num_trees):
            X_caret, y_caret = self._bootstrap_dataset(X, y)
            decision_tree_model = DecisionTreeRegressor()
            decision_tree_model.fit(X_caret, y_caret)

            self.trees.append(decision_tree_model)


    def predict(self, X):
        size = X.shape[0]
        arr = np.zeros(size)
        for tree in self.trees:
            predictions = tree.predict(X)
            for i in range(size):
                arr[i] += predictions[i]/self.num_trees
        return arr


