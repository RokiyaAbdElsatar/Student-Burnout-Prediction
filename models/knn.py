import math

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict_one(sample) for sample in X]

    def _predict_one(self, sample):
        # Calculate distances to all training samples
        distances = []
        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(sample, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        # Sort by distance
        distances.sort(key=lambda x: x[0])

        # Get k nearest neighbors
        k_nearest = distances[:self.k]

        # Majority vote
        votes = {}
        for _, label in k_nearest:
            votes[label] = votes.get(label, 0) + 1

        return max(votes, key=votes.get)

    def _euclidean_distance(self, a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))
