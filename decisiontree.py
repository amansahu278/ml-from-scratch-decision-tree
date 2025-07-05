import numpy as np
from criterion import Criterion
from splitter import Splitter

class Node:
        # Modern CART based trees only have binary splits
        def __init__(self, feature=None, left=None, right=None, value=None):
            self.feature_idx = feature
            self.left = left
            self.right = right
            self.value = value # This is the feature value, or prediction

class DecisionTree:
    
    def __init__(self, criterion='gini', splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=0, min_impurity_decrease=0, class_weight=None):
        self.tree = None
        
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

        criterion_class = self.get_criterion(criterion) # init criterion heresplitter_class = self.get_splitter(splitter)
        self.criterion = criterion_class()

        splitter_class = self.get_splitter(splitter)
        self.splitter = splitter_class(self.criterion, max_features, random_state) 

    def get_criterion(self, criterion_name):

        if criterion_name not in Criterion.registry:
            raise ValueError(f"Criteriion {criterion_name}")
    
        return Criterion.registry[criterion_name]

    def get_splitter(self, splitter_name):
        if splitter_name not in Splitter.registry:
            raise ValueError(f"Splitter {splitter_name} is not registered")

        return Splitter.registry[splitter_name]

    def stopping_criteria_met(self, X, y):

        if len(np.unique(y)) == 1:
            return True
        
        if y.shape[0] < self.min_samples_split:
            return True

        return False
    
    def _traverse(self, sample, node):
        pass

    def _leaf_node(self, y):
        # For classification
        return Node(value=np.bincount(y).argmax())

    def _build_tree(self, X, y, depth=0):
        """
        This function is for building the tree through partitioning
        Note: It is built in a way that internal nodes are always split, so there is no case of an internal node not having either children
        """

        # Checking if pure or min_samples_split is met (NOTE: before splitting)
        if self.stopping_criteria_met(X, y):
            return self._leaf_node(y)

        # Check if max_depth is met
        if self.max_depth is not None and depth >= self.max_depth:
            return self._leaf_node(y)

        # 1. From features, find the best feature to split on
        feature_idx, threshold, gain = self.splitter.split(X, y, self.categorical_features)

        # Check min_impurity_decrease
        if gain < self.min_impurity_decrease:
            return self._leaf_node(y)

        # 2. Split the data into left and right based on the best feature
        values = X[:, feature_idx]
        if feature_idx in self.categorical_features:
            left = values == threshold
        else:
            left = values <= threshold
        right = ~left
        
        # Split
        X_left, X_right = X[left], X[right]
        y_left, y_right = y[left], y[right]

        # Check for min_leaf_samples, if neither children have enough, parent becoes leaf (NOTE: this is after splitting)
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            return self._leaf_node(y)

        node = Node(feature=feature_idx, value=threshold)
        node.left = self._build_tree(X_left, y_left, depth+1)
        node.right = self._build_tree(X_right, y_right, depth+1)
        return node

    def fit(self, X, y, categorical_features):
        self.features = X.shape[1]
        self.samples = X.shape[0]
        self.classes = np.unique(y)
        self.feature_discrete_values = [np.unique(X[:, i]) for i in range(self.features)]
        self.categorical_features = categorical_features

        self._tree = self._build_tree(X, y)

    def visualize(self, node=None, depth=0):
        node = node or self._tree
        indent = "  " * depth
        if node is None:
            print(f"{indent}None")
            return

        if node.left is None and node.right is None:
            print(f"{indent}Leaf: {node.value}")
            return
        
        split_type = "==" if node.feature_idx in self.categorical_features else "<="
        print(f"{indent} if X[{node.feature_idx}] {split_type} {node.value}:")

        print(f"{indent}--> True:")
        self.visualize(node.left, depth+1)
        print(f"{indent}--> False:")
        self.visualize(node.right, depth+1)

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

class DecisionTreeClassifier(DecisionTree):

    def __init__(self, criterion='gini', splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=0, min_impurity_decrease=0, class_weight=None):
        super().__init__(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features, random_state, min_impurity_decrease, class_weight)
    
    def _leaf_node(self, y):
        return Node(value=np.bincount(y).argmax())
    

    def _traverse(self, sample, node):
        
        # Non leaf nodes
        while node.left is not None and node.right is not None:
            val = sample[node.feature_idx]
            if np.isnan(val):
                left_pred = self._traverse(sample, node.left)
                right_pred = self._traverse(sample, node.right)
                return np.bincount([left_pred, right_pred]).argmax()

            if node.feature_idx in self.categorical_features:
                if val == node.value:
                    node = node.left
                else:
                    node = node.right
            else:
                if val <= node.value:
                    node = node.left
                else:
                    node = node.right       
        # Leaf node
        return node.value

    def predict(self, X):
        predictions = []
        for sample in X:
            node = self._tree
            prediction = self._traverse(sample, node)
            predictions.append(prediction)
        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        acc = np.mean(predictions==y)
        return acc

class DecisionTreeRegressor(DecisionTree):

    def __init__(self, criterion='gini', splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=0, min_impurity_decrease=0, class_weight=None):
        super().__init__(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features, random_state, min_impurity_decrease, class_weight)
    
    def _leaf_node(self, y):
        return Node(value=np.mean(y))
    
    def _traverse(self, sample, node):
        
        while node.left is not None and node.right is not None:
            val = sample[node.feature_idx]
            if np.isnan(val):
                left_pred = self._traverse(sample, node.left)
                right_pred = self._traverse(sample, node.right)
                return np.mean([left_pred, right_pred])

            if node.feature_idx in self.categorical_features:
                if val == node.value:
                    node = node.left
                else:
                    node = node.right
            else:
                if val <= node.value:
                    node = node.left
                else:
                    node = node.right       

        return node.value

    def predict(self, X):
        predictions = []
        for sample in X:
            node = self._tree
            prediction = self._traverse(sample, node)
            predictions.append(prediction)
        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        mse = np.mean((predictions - y)**2)
        return mse
