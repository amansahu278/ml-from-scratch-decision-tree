import numpy as np
from abc import ABC
from typing import Optional, Tuple

class Splitter(ABC):
    registry = {}

    def __init__(self, criterion, max_features, random_state):
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

    @classmethod
    def register(cls, name):
        def wrapper(splitter_class):
            cls.registry[name] = splitter_class
            return splitter_class
        return wrapper

    def _numerical_thresholds(self, values):
        """
        Calculate the thresholds for numerical features
        """
        unique_values = np.unique(values)
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        return thresholds

    def split(self, X, y, categorical_features) -> Tuple[Optional[int], Optional[float], float]:
        pass

    def _best_split(self, X, y, categorical_features):
        best_feature = None
        best_threshold = None
        best_gain = -np.inf
        parent_score = self.criterion.score(y)

        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]
            if len(np.unique(values)) < 2:
                continue
            if feature_idx in categorical_features:
                thresholds = np.unique(values)
            else:
                thresholds = self._numerical_thresholds(values)
                
            for threshold in thresholds:
                if feature_idx in categorical_features:
                    left_mask = values == threshold
                else:
                    left_mask = values <= threshold
                
                # TODO: A simpler splitter, without using subsets, no use of max features
                right_mask = ~left_mask

                y_left = y[left_mask]
                y_right = y[right_mask]

                # Avoiding division by
                if len(y_left) < 1 or len(y_right) < 1:
                    continue

                weighted_children_score = (len(y_left) * self.criterion.score(y_left) + len(y_right) * self.criterion.score(y_right))/len(y)
                gain = parent_score - weighted_children_score

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain

    def _random_split(self, X, y, categorical_features):
        
        np.random_seed(self.random_state)
        feature = np.random.choice(X.shape[1])
        values = X[:, feature]

        if feature in categorical_features:
            threshold = np.random.choice(np.unique(values))
        else:
            thresholds = self._numerical_thresholds(values)
            threshold = np.random.choice(thresholds)
        
        if feature in categorical_features:
            left = values == threshold
        else:
            left = values <= threshold
        right = ~left

        y_left = y[left]
        y_right = y[right]

        if len(y_left) < 1 or len(y_right) < 1:
            return None, None, -np.inf
        
        parent_score = self.criterion.score(y)
        weighted_children_score = (len(y_left) * self.criterion.score(y_left) + len(y_right) * self.criterion.score(y_right))/len(y)
        gain = parent_score - weighted_children_score

        return feature, threshold, gain

@Splitter.register('best')
class BestSplitter(Splitter):
    def split(self, X, y, categorical_features):
        return self._best_split(X, y, categorical_features)

@Splitter.register('random')
class RandomSplitter(Splitter):
    def split(self, X, y, categorical_features):
        return self._random_split(X, y, categorical_features)
