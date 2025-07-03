import numpy as np

class DecisionTree:

    class Node:
        # Modern CART based trees only have binary splits
        def __init__(self, feature, left=None, right=None, value=None):
            self.feature_idx = feature
            self.left = left
            self.right = right
            self.value = value

    def _build_tree(self, X, y) -> Node:
        """
        This function is for building the tree through partitioning
        """
        
        
        





        pass

    def __init__(self, criterion='gini', splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=0, min_impurity_descrease=0, class_weight=None):
        self.tree = None
        self.criterion = criterion
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass