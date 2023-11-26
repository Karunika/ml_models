import numpy as np

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, split_factor=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.split_factor = split_factor

        self.value = value
    

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree():
    def __init__(self, min_samples_split=2, max_depth=10):
        # stopping condition
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        self.root = None


    def _build_tree(self, X, y, depth=0):
        num_samples, _ = X.shape

        if depth >= self.max_depth or num_samples <= self.min_samples_split:
            return Node(value=self._eval_leaf_val(y))
        
        split_feature, split_threshold, split_factor, left_idxs, right_idxs = self._best_split(X, y)

        if split_factor == 0:
            return Node(value=self._eval_leaf_val(y))

        left_subtree = self._build_tree(X.iloc[left_idxs], y.iloc[left_idxs], depth+1)
        right_subtree = self._build_tree(X.iloc[right_idxs], y.iloc[right_idxs], depth+1)

        return Node(split_feature, split_threshold, left_subtree, right_subtree, split_factor)


    def _best_split(self, X, y):
        split_feature, split_threshold = None, None
        best_left_idxs, best_right_idxs = None, None
        best_split_factor = float("-inf")

        for feature in X.columns:
            X_col = X[feature]
            thresholds = np.unique(X_col)

            for thr in thresholds:
                left_idxs, right_idxs = self._split(thr, X_col)

                split_factor = self._eval_split_factor(y, y.iloc[left_idxs], y.iloc[right_idxs])

                if split_factor > best_split_factor:
                    best_split_factor = split_factor
                    split_feature = feature
                    split_threshold = thr
                    best_left_idxs = left_idxs
                    best_right_idxs = right_idxs


        return split_feature, split_threshold, best_split_factor, best_left_idxs, best_right_idxs


    def _split(self, threshold, X_col):
        left_idxs = np.argwhere(X_col <= threshold).flatten()
        right_idxs = np.argwhere(X_col > threshold).flatten()
        return left_idxs, right_idxs


    def _make_prediction(self, record, node):
        if node.is_leaf_node():
            return node.value
        
        if record[node.feature] <= node.threshold:
            return self._make_prediction(record, node.left)
        else:
            return self._make_prediction(record, node.right)


    def fit(self, X, y):
        self.root = self._build_tree(X, y)


    def predict(self, X):
        arr = []
        for _, record in X.iterrows():
            arr.append(self._make_prediction(record, self.root))
        return arr


class DecisionTreeClassifier(DecisionTree):
    def __init__(self, min_samples_leaf=2, max_depth=10, criterion="gini"):
        super().__init__(min_samples_leaf, max_depth)

        self.criterion = criterion

    
    def _eval_leaf_val(self, y):
        class_labels = np.unique(y)
        dp = np.zeros(len(class_labels))

        for y_in in y:
            for label_idx, label in enumerate(class_labels):
                dp[label_idx] += 1 if y_in == label else 0
        
        return class_labels[np.argmax(dp)]


    def _eval_split_factor(self, y, y_left, y_right):
        return self._info_gain(y, y_left, y_right)


    def _info_gain(self, y, y_left, y_right):
        num_samples, = y.shape

        left_weight = y_left.shape[0] / num_samples
        right_weight = y_right.shape[0] / num_samples
        return self._entropy(y) - left_weight*self._entropy(y_left) - right_weight*self._entropy(y_right)


    def _entropy(self, y):
        class_labels = np.unique(y)
        dp = np.zeros(len(class_labels))

        for y_in in y:
            for label_idx, label in enumerate(class_labels):
                dp[label_idx] += 1 if y_in == label else 0

        dp_prob = dp / len(y)
        if self.criterion == "entropy":
            return -np.sum([p*np.log2(p) if p != 0 else 0 for p in dp_prob])

        # gini index
        return  -np.sum([p*p if p != 0 else 0 for p in dp_prob])


class DecisionTreeRegressor(DecisionTree):
    def __init__(self, min_samples_leaf=2, max_depth=10):
        super().__init__(min_samples_leaf, max_depth)


    def _eval_leaf_val(self, y):
        return 0 if y.shape[0] == 0 else np.mean(y)


    def _eval_split_factor(self, y, y_left, y_right):
        return self._variance_reduction(y, y_left, y_right)


    def _variance_reduction(self, y, y_left, y_right):
        num_samples, = y.shape
    
        left_weight = y_left.shape[0] / num_samples
        right_weight = y_right.shape[0] / num_samples

        return self._variance(y) - (left_weight*self._variance(y_left) + right_weight*self._variance(y_right))


    def _variance(self, y):
        avg = np.mean(y) / (y.shape[0] -1)
        return np.sum([(avg-x)**2 for x in y])


