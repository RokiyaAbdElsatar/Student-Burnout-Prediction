import random
from .decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = len(X)

        for _ in range(self.n_trees):
            # Bootstrap sampling (63.2% of data on average)
            indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        all_predictions = []
        for tree in self.trees:
            all_predictions.append(tree.predict(X))

        # Majority vote
        final_predictions = []
        n_samples = len(X)
        for i in range(n_samples):
            votes = {}
            for tree_idx in range(self.n_trees):
                pred = all_predictions[tree_idx][i]
                votes[pred] = votes.get(pred, 0) + 1
            final_predictions.append(max(votes, key=votes.get))

        return final_predictions
