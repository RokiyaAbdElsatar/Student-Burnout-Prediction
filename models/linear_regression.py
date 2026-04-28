class LinearRegressionScratch:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])

        # Add bias term
        X_bias = [row + [1] for row in X]

        # Normal equation: (X^T * X)^(-1) * X^T * y
        XT = [[X_bias[j][i] for j in range(n_samples)] for i in range(n_features + 1)]
        XTX = [[sum(XT[i][k] * X_bias[k][j] for k in range(n_samples)) for j in range(n_features + 1)] for i in range(n_features + 1)]
        XTy = [sum(XT[i][k] * y[k] for k in range(n_samples)) for i in range(n_features + 1)]

        try:
            XTX_inv = self._inverse_matrix(XTX)
            beta = [sum(XTX_inv[i][j] * XTy[j] for j in range(n_features + 1)) for i in range(n_features + 1)]
            self.coefficients = beta[:-1]
            self.intercept = beta[-1]
        except:
            # Fallback to simple solution
            self.coefficients = [0] * n_features
            self.intercept = sum(y) / len(y) if y else 0

    def predict(self, X):
        if self.coefficients is None:
            return [0] * len(X)
        return [sum(self.coefficients[i] * row[i] for i in range(len(row))) + self.intercept for row in X]

    def _transpose(self, matrix):
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

    def _multiply(self, A, B):
        n = len(A)
        m = len(B[0])
        p = len(B)
        return [[sum(A[i][k] * B[k][j] for k in range(p)) for j in range(m)] for i in range(n)]

    def _inverse_matrix(self, matrix):
        n = len(matrix)
        # Augment with identity
        augmented = [matrix[i] + [1 if j == i else 0 for j in range(n)] for i in range(n)]

        for col in range(n):
            # Find pivot
            pivot = col
            for row in range(col + 1, n):
                if abs(augmented[row][col]) > abs(augmented[pivot][col]):
                    pivot = row
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

            # Normalize pivot row
            pivot_val = augmented[col][col]
            if pivot_val == 0:
                raise ValueError("Matrix is not invertible")
            for j in range(2 * n):
                augmented[col][j] /= pivot_val

            # Eliminate other rows
            for row in range(n):
                if row != col and augmented[row][col] != 0:
                    factor = augmented[row][col]
                    for j in range(2 * n):
                        augmented[row][j] -= factor * augmented[col][j]

        return [augmented[i][n:] for i in range(n)]


class OneVsRestLinearRegression:
    def __init__(self, num_classes=8):
        self.num_classes = num_classes
        self.models = [LinearRegressionScratch() for _ in range(num_classes)]

    def fit(self, X, y):
        for cls in range(self.num_classes):
            y_binary = [1 if label == cls else 0 for label in y]
            self.models[cls].fit(X, y_binary)

    def predict(self, X):
        scores = []
        for model in self.models:
            scores.append(model.predict(X))

        predictions = []
        for i in range(len(X)):
            best_class = 0
            best_score = scores[0][i]
            for cls in range(1, self.num_classes):
                if scores[cls][i] > best_score:
                    best_score = scores[cls][i]
                    best_class = cls
            predictions.append(best_class)
        return predictions
