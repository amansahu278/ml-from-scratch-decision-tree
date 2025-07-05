import numpy as np #type:ignore
from abc import ABC, abstractmethod
class Criterion(ABC):
    registry = {}

    @classmethod
    def register(cls, name):
        def wrapper(criterion_class):
            cls.registry[name] = criterion_class
            return cls
        return wrapper
    
    @abstractmethod
    def score(self, y) -> float:
        pass

@Criterion.register('gini')
class GiniCriterion(Criterion):
    
    def score(self, y):
        """
        Calculate the Gini impurity
        """
        t = y.shape[0]
        if t == 0:
            return 0
        
        p = np.bincount(y) / t
        return 1 - np.sum(p ** 2)

@Criterion.register('entropy')
class EntropyCriterion(Criterion):
    
    def score(self, y):
        """
        Calculate the entropy
        """
        
        t = y.shape[0]
        if t == 0:
            return 0
        
        p = np.bincount(y)/t
        p = p[p > 0]
        return -np.sum(p*np.log2(p))

@Criterion.register('logloss')
class LogLossCriterion(Criterion):
    
    def score(self, y):
        """
        Calculate the log loss
        """
        t = y.shape[0]
        if t == 0:
            return 0
        
        p = np.bincount(y)/t
        p = p[p>0]
        return -np.sum(p*np.log(p))

@Criterion.register('mse')
class MSECriterion(Criterion):
    
    def score(self, y):
        """
        Calculate the mse
        """
        # MSE is variance of the target variable
        return np.var(y, dtype=float) if y.size > 0 else 0