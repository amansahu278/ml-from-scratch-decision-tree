import numpy as np #type:ignore

class Splitter:
    registry = {}

    def __init__(self, criterion, max_features, min_samples_leaf, random_state):
        self.criterion = criterion
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    @classmethod
    def register(cls, name):
        def wrapper(splitter_class):
            cls.registry[name] = splitter_class
            return cls
        return wrapper

    def _numerical_thresholds(self, values):
        """
        Calculate the thresholds for numerical features
        """
        unique_values = np.unique(values)
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        return thresholds

    def best_split(self, X, y, categorical_features):
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

                # Avoiding division by zero
                if len(y_left) < 1 or len(y_right) < 1:
                    continue

                weighted_children_score = (len(y_left) * self.criterion.score(y_left) + len(y_right) * self.criterion.score(y_right))/len(y)
                gain = parent_score - weighted_children_score

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain

    def random_split(self, X, y):
        pass
    
@Splitter.register('best')
class BestSplitter(Splitter):
    pass

@Splitter.register('random')
class RandomSplitter(Splitter):
    pass