# ml-from-scratch-decision-tree

## Learning
* Registry

* The modern methods, don't split a node to every value of a discrete variable, it is always a binary split based on subsets of the feature values
** Since there can be many subsets (2^(k-1) - 1)

* Splitting is based on a criterion, to maximise information gain, minimize impurity
* There are two types of split (on sklearn), 
    * best where we take time to find the best possible split or 
    * random where we select split randomly which is faster, usually combining it to form a forest

* For the split, max_features tells the number of features we should consider to form a split, min_samples_leaf, the minimum number of samples each leaf should have to be considered a node (after splitting). There is also min_samples_split, to consider how many samples should be present for us to even consider splitting (before splitting). If we decide not a split a node anymore, it becomes leaf. 