import math
import random

class LogisticRegressionScratch:
    def __init__(self, num_classes=8, learning_rate=0.01, epochs=1000):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])

        # Initialize weights for each class
        self.weights = [[0] * n_features for _ in range(self.num_classes)]
        self.bias = [0] * self.num_classes

        for epoch in range(self.epochs):
            # Forward pass
            predictions = self._softmax(X)

            # Compute gradients
            for cls in range(self.num_classes):
                for i in range(n_samples):
                    # One-hot encoding
                    y_true = 1 if y[i] == cls else 0
                    error = predictions[i][cls] - y_true

                    # Update weights
                    for j in range(n_features):
                        self.weights[cls][j] -= self.learning_rate * error * X[i][j]
                    self.bias[cls] -= self.learning_rate * error

    def predict(self, X):
        probabilities = self._softmax(X)
        return [max(range(self.num_classes), key=lambda cls: probabilities[i][cls]) for i in range(len(X))]

    def _softmax(self, X):
        results = []
        for sample in X:
            logits = [sum(self.weights[cls][j] * sample[j] for j in range(len(sample))) + self.bias[cls] for cls in range(self.num_classes)]
            max_logit = max(logits)
            exp_logits = [math.exp(l - max_logit) for l in logits]
            sum_exp = sum(exp_logits)
            results.append([exp_logits[cls] / sum_exp for cls in range(self.num_classes)])
        return results
