import csv
import math
import random


def load_raw_data(filepath):
    """Load CSV data and return header and rows."""
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def aggregate_per_person(raw_data):
    """Aggregate time-series data per person. Returns (X, y) where X is list of feature vectors."""
    from collections import defaultdict

    person_data = defaultdict(list)
    for row in raw_data:
        person_id = row[0]
        features = [float(x) for x in row[1:8]]  # acc_x, acc_y, acc_z, eda, bvp, hr, temp
        pain_scale = int(row[8])
        person_data[person_id].append((features, pain_scale))

    X = []
    y = []
    for person_id, records in person_data.items():
        features_list = [r[0] for r in records]
        pain_scales = [r[1] for r in records]

        if pain_scales.count(pain_scales[0]) != len(pain_scales):
            continue

        means = [sum(col) / len(col) for col in zip(*features_list)]
        stds = []
        for i in range(len(features_list[0])):
            vals = [f[i] for f in features_list]
            mean = means[i]
            variance = sum((x - mean) ** 2 for x in vals) / len(vals)
            stds.append(math.sqrt(variance))

        mins = [min(col) for col in zip(*features_list)]
        maxs = [max(col) for col in zip(*features_list)]

        feature_vector = means + stds + mins + maxs
        X.append(feature_vector)
        y.append(pain_scales[0])

    return X, y


def split_data(X, y, test_size=0.2, shuffle=True):
    """Split data into train and test sets."""
    data = list(zip(X, y))
    if shuffle:
        random.seed(42)
        random.shuffle(data)

    split_idx = int(len(data) * (1 - test_size))
    train = data[:split_idx]
    test = data[split_idx:]

    X_train = [x[0] for x in train]
    y_train = [x[1] for x in train]
    X_test = [x[0] for x in test]
    y_test = [x[1] for x in test]

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test):
    """Apply z-score normalization using training statistics."""
    num_features = len(X_train[0])
    means = []
    stds = []

    for i in range(num_features):
        col = [x[i] for x in X_train]
        mean = sum(col) / len(col)
        variance = sum((x - mean) ** 2 for x in col) / len(col)
        std = math.sqrt(variance)
        means.append(mean)
        stds.append(std if std > 0 else 1.0)

    X_train_norm = [[(x[i] - means[i]) / stds[i] for i in range(num_features)] for x in X_train]
    X_test_norm = [[(x[i] - means[i]) / stds[i] for i in range(num_features)] for x in X_test]

    return X_train_norm, X_test_norm


def encode_labels(y):
    """Encode pain_scale (1-8) to 0-indexed labels."""
    return [y_val - 1 for y_val in y]
