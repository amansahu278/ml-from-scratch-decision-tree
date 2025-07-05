import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier as SKlearnTreeClassifier
from sklearn.tree import DecisionTreeRegressor as SKlearnTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decisiontree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing


if __name__ == "__main__":

    # 1. Load dataset
    X, y = load_iris(return_X_y=True)

    # 2. Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. Train your tree
    my_tree = DecisionTreeClassifier(criterion="gini", max_depth=3)
    my_tree.fit(X_train, y_train, categorical_features=set())  # assuming all features are numerical
    y_pred_my = my_tree.predict(X_test)

    # 4. Train sklearn tree
    sk_tree = SKlearnTreeClassifier(criterion="gini", max_depth=3, random_state=42)
    sk_tree.fit(X_train, y_train)
    y_pred_sk = sk_tree.predict(X_test)

    # 5. Compare performance
    acc_my = accuracy_score(y_test, y_pred_my)
    acc_sk = accuracy_score(y_test, y_pred_sk)

    print(f"My Tree Accuracy: {acc_my:.4f}")
    print(f"Sklearn Tree Accuracy: {acc_sk:.4f}")

    print("My Tree Structure:")
    my_tree.visualize()

    X, y = fetch_california_housing(return_X_y=True)

    my_reg = DecisionTreeRegressor(criterion="mse", max_depth=4)
    my_reg.fit(X_train, y_train, categorical_features=set())

    sk_tree = SKlearnTreeRegressor(criterion="squared_error", max_depth=4, random_state=42)
    sk_tree.fit(X_train, y_train)

    sk_pred_reg = sk_tree.predict(X_test)

    my_mse = my_reg.score(X_test, y_test)
    sk_mse = sk_tree.score(X_test, y_test)

    print(f"My Regression Tree MSE: {my_mse:.4f}")
    print(f"Sklearn Regression Tree MSE: {sk_mse:.4f}")


