# 🌳 Decision Tree From Scratch

A minimal, educational implementation of a Decision Tree classifier in pure Python and NumPy. This project is designed to help you understand the core mechanics of decision trees, including splitting, impurity criteria, and tree construction—without relying on high-level libraries like scikit-learn.

---

## 🚀 Features

- Binary splits for both continuous and discrete features
- Pluggable impurity/loss criteria (Gini, Entropy, LogLoss, MSE)
- Customizable splitting strategies ("best" and "random")
- Configurable tree depth, min samples per split/leaf
- Simple, readable codebase for learning and experimentation

---

## 🏗️ Project Structure

```
.
├── criterion.py      # Impurity/loss criteria (Gini, Entropy, LogLoss)
├── decisiontree.py   # Main DecisionTree class and tree-building logic
├── splitter.py       # Splitter strategies (best/random)
├── main.py           # Sample usage and training script
└── README.md         # Project documentation
```

---

## 📚 Learning Notes

What i learnt while implementing this decision tree from scratch:

- **Registry Pattern:**
  - Used for managing and selecting different impurity/loss criteria dynamically.

- **Modern Decision Trees:**
  - Always use binary splits (CART), even for discrete variables (by splitting on subsets of values).
  - The number of possible splits for a discrete variable with \(k\) values is \(2^{k-1} - 1\).

- **Splitting Criteria:**
  - Splitting is based on a criterion to maximize information gain or minimize impurity.
  - Two main splitter types (as in scikit-learn):
    - **Best:** Exhaustively searches for the optimal split.
    - **Random:** Selects splits randomly (faster, used in random forests).

- **Tree Growth Parameters:**
  - `max_features`: Number of features to consider for each split.
  - `min_samples_leaf`: Minimum samples required at a leaf node (after splitting).
  - `min_samples_split`: Minimum samples required to consider splitting a node (before splitting).
  - If a node is not split further, it becomes a leaf.
  - A node is always split so that it is either a leaf, or had 2 children
---

## 📝 Usage

1. **Install dependencies**

   Requires: `numpy`

   ```bash
   pip install numpy
   pip install scikit-learn  # Optional, for testing with sklearn datasets
   ```

2. **Import and use the DecisionTree**

   ```python
   from decisiontree import DecisionTree
   import numpy as np

   # Example data
   X = np.array([[...], ...])  # shape: (n_samples, n_features)
   y = np.array([...])         # shape: (n_samples,)

   tree = DecisionTree(criterion='gini', max_depth=3)
   tree.fit(X, y)
   predictions = tree.predict(X)
   ```

---

## 🤔 Why From Scratch?

- To demystify how decision trees work under the hood (doing is learning, imo)
- To provide a foundation for building more advanced models (e.g., random forests, boosting)
- For educational and research purposes

---

## 📖 References

- [scikit-learn DecisionTreeClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
- [Wikipedia: Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)

---

**Part of the [`ml-from-scratch-xyz`](https://github.com/amansahu278?tab=repositories&q=ml-from-scratch) series.**