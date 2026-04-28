import math

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return [self._predict_one(sample, self.tree) for sample in X]

    def _build_tree(self, X, y, depth):
        num_samples = len(y)
        if num_samples == 0:
            return None

        # Count class frequencies
        class_counts = {}
        for label in y:
            class_counts[label] = class_counts.get(label, 0) + 1

        # Stopping criteria
        if depth >= self.max_depth or len(class_counts) == 1:
            return {'type': 'leaf', 'class': max(class_counts, key=class_counts.get)}

        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return {'type': 'leaf', 'class': max(class_counts, key=class_counts.get)}

        # Split data
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(num_samples):
            if X[i][best_feature] <= best_threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        if len(left_y) == 0 or len(right_y) == 0:
            return {'type': 'leaf', 'class': max(class_counts, key=class_counts.get)}

        return {
            'type': 'node',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(left_X, left_y, depth + 1),
            'right': self._build_tree(right_X, right_y, depth + 1)
        }

    def _find_best_split(self, X, y):
        num_samples = len(X)
        num_features = len(X[0])
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_idx in range(num_features):
            values = [X[i][feature_idx] for i in range(num_samples)]
            unique_values = sorted(set(values))

            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                left_y, right_y = [], []

                for j in range(num_samples):
                    if X[j][feature_idx] <= threshold:
                        left_y.append(y[j])
                    else:
                        right_y.append(y[j])

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gain = self._information_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, parent_y, left_y, right_y):
        parent_entropy = self._entropy(parent_y)
        n = len(parent_y)
        n_left, n_right = len(left_y), len(right_y)

        if n_left == 0 or n_right == 0:
            return 0

        child_entropy = (n_left / n) * self._entropy(left_y) + (n_right / n) * self._entropy(right_y)
        return parent_entropy - child_entropy

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        class_counts = {}
        for label in y:
            class_counts[label] = class_counts.get(label, 0) + 1

        entropy = 0
        total = len(y)
        for count in class_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _predict_one(self, sample, node):
        if node['type'] == 'leaf':
            return node['class']

        if sample[node['feature']] <= node['threshold']:
            return self._predict_one(sample, node['left'])
        else:
            return self._predict_one(sample, node['right'])
