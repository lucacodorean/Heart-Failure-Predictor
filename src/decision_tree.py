import numpy as np
from collections import Counter

def gini(y):                                                        # This function implements a method to evaluate the purity of a node.
    counts = np.bincount(y)                                         # gini(Node) = 0 => Node = pure.
    probabilities = counts / len(y)                                 # gini ---> inf => Node = impure.
    return 1.0 - np.sum(probabilities**2)


def split(X, y, feature, threshold):                                # The purpose of this function is to create a split based on the given threshold
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]


def best_split(X, y):                                               # This function will determine the best split.
    best_gain = 0
    best_feature, best_thresh = None, None
    current_gini = gini(y)
    n_features = X.shape[1]
    
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for thresh in thresholds:
            X_left, y_left, X_right, y_right = split(X, y, feature, thresh)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            p = len(y_left) / len(y)
            gain = current_gini - (p * gini(y_left) + (1 - p) * gini(y_right))
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_thresh = thresh

    return best_feature, best_thresh

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(set(y))
        
        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature, threshold = best_split(X, y)
        if feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        X_left, y_left, X_right, y_right = split(X, y, feature, threshold)
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        return Node(feature, threshold, left_child, right_child)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
