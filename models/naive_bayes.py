import math

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.feature_stats = {}  # class -> feature_idx -> (mean, std)

    def fit(self, X, y):
        self.classes = list(set(y))
        n_features = len(X[0])

        # Calculate class priors
        for cls in self.classes:
            self.class_priors[cls] = sum(1 for label in y if label == cls) / len(y)

        # Calculate mean and std for each feature per class
        for cls in self.classes:
            X_cls = [X[i] for i in range(len(X)) if y[i] == cls]
            self.feature_stats[cls] = []

            for feature_idx in range(n_features):
                values = [row[feature_idx] for row in X_cls]
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                std = math.sqrt(variance) if variance > 0 else 0.001
                self.feature_stats[cls].append((mean, std))

    def predict(self, X):
        return [self._predict_one(sample) for sample in X]

    def _predict_one(self, sample):
        best_class = None
        best_prob = float('-inf')

        for cls in self.classes:
            # Start with log prior
            log_prob = math.log(self.class_priors[cls] + 1e-10)

            # Add log likelihood for each feature
            for i in range(len(sample)):
                mean, std = self.feature_stats[cls][i]
                if std > 0:
                    # Gaussian PDF (log scale)
                    log_prob += math.log(1 / (math.sqrt(2 * math.pi) * std) + 1e-10) - 0.5 * ((sample[i] - mean) / std) ** 2

            if log_prob > best_prob:
                best_prob = log_prob
                best_class = cls

        return best_class
