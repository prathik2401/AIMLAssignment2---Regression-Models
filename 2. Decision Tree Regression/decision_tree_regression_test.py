import numpy as np
import pandas as pd

# Define the decision tree regression algorithm
class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)

        num_features = X.shape[1]
        best_feature, best_threshold, best_reduction = None, None, -np.inf

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature] <= threshold)
                right_indices = np.where(X[:, feature] > threshold)

                if len(left_indices[0]) == 0 or len(right_indices[0]) == 0:
                    continue

                left_y = y[left_indices]
                right_y = y[right_indices]

                reduction = self._calculate_reduction(y, left_y, right_y)
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feature
                    best_threshold = threshold

        if best_reduction == -np.inf:
            return np.mean(y)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return (best_feature, best_threshold, left_tree, right_tree)

    def _calculate_reduction(self, parent, left_child, right_child):
        reduction = self._calculate_variance(parent) - \
                    (self._calculate_variance(left_child) * len(left_child) + \
                     self._calculate_variance(right_child) * len(right_child)) / len(parent)
        return reduction

    def _calculate_variance(self, y):
        return np.var(y)

    def predict(self, X):
        return np.array([self._predict_instance(x, self.tree) for x in X])

    def _predict_instance(self, x, tree):
        if not isinstance(tree, tuple):
            return tree

        feature, threshold, left_tree, right_tree = tree
        if x[feature] <= threshold:
            return self._predict_instance(x, left_tree)
        else:
            return self._predict_instance(x, right_tree)


# Load the dataset from CSV
data = pd.read_csv(r"C:\Users\Prathik\Downloads\AIMLAssignment2\\2. Decision Tree Regression\CO2 Emissions_Canada.csv")

# Separate features and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Take the first 40 values for expected CO2 emissions
X_expected = X
y_expected = y

# Take the later values for predicting CO2 emissions
X_pred = X[0:10]
y_pred = y[0:10]

# Instantiate and fit the decision tree regressor
regressor = DecisionTreeRegressor(max_depth=6.8)
regressor.fit(X_expected, y_expected)

# Make predictions
predictions = regressor.predict(X_pred)

# Print predictions
print("Predictions:", predictions)
