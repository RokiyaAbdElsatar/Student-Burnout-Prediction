import sys
import json
import os

# Get project root (directory containing this file)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader import load_raw_data, aggregate_per_person, split_data, normalize, encode_labels
from utils.metrics import accuracy, precision, recall, f1_score, print_metrics

NUM_CLASSES = 8


def load_and_preprocess():
    """Load and preprocess data. Returns X_train, X_test, y_train, y_test."""
    print("Loading data...")
    header, rows = load_raw_data(os.path.join(PROJECT_ROOT, 'dataset', 'pain_dataset_200P_4hz.csv'))
    X, y = aggregate_per_person(rows)
    y = encode_labels(y)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_norm, X_test_norm = normalize(X_train, X_test)
    return X_train_norm, X_test_norm, y_train, y_test


def run_builtin_models(X_train, X_test, y_train, y_test):
    """Run all 6 built-in models and return results."""
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)
    y_test_list = list(y_test)

    results = {}

    # 1. Decision Tree
    print("\n" + "="*50)
    print("1. Decision Tree (Built-in)")
    print("="*50)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train_np, y_train_np)
    y_pred = dt.predict(X_test_np).tolist()
    acc = accuracy(y_test_list, y_pred)
    prec, prec_macro = precision(y_test_list, y_pred, NUM_CLASSES)
    rec, rec_macro = recall(y_test_list, y_pred, NUM_CLASSES)
    f1 = f1_score(prec_macro, rec_macro)
    print_metrics(y_test_list, y_pred, NUM_CLASSES)
    results['Decision Tree'] = {'Accuracy': acc, 'Macro Precision': prec_macro, 'Macro Recall': rec_macro, 'Macro F1': f1}

    # 2. Random Forest
    print("\n" + "="*50)
    print("2. Random Forest (Built-in)")
    print("="*50)
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_train_np, y_train_np)
    y_pred = rf.predict(X_test_np).tolist()
    acc = accuracy(y_test_list, y_pred)
    prec, prec_macro = precision(y_test_list, y_pred, NUM_CLASSES)
    rec, rec_macro = recall(y_test_list, y_pred, NUM_CLASSES)
    f1 = f1_score(prec_macro, rec_macro)
    print_metrics(y_test_list, y_pred, NUM_CLASSES)
    results['Random Forest'] = {'Accuracy': acc, 'Macro Precision': prec_macro, 'Macro Recall': rec_macro, 'Macro F1': f1}

    # 3. Linear Regression (One-vs-Rest for classification)
    print("\n" + "="*50)
    print("3. Linear Regression (Built-in)")
    print("="*50)
    lr_models = []
    for cls in range(NUM_CLASSES):
        model = LinearRegression()
        y_binary = (y_train_np == cls).astype(int)
        model.fit(X_train_np, y_binary)
        lr_models.append(model)

    predictions = []
    for sample in X_test_np:
        scores = [model.predict([sample])[0] for model in lr_models]
        predictions.append(np.argmax(scores))

    y_pred = predictions
    acc = accuracy(y_test_list, y_pred)
    prec, prec_macro = precision(y_test_list, y_pred, NUM_CLASSES)
    rec, rec_macro = recall(y_test_list, y_pred, NUM_CLASSES)
    f1 = f1_score(prec_macro, rec_macro)
    print_metrics(y_test_list, y_pred, NUM_CLASSES)
    results['Linear Regression'] = {'Accuracy': acc, 'Macro Precision': prec_macro, 'Macro Recall': rec_macro, 'Macro F1': f1}

    # 4. Logistic Regression
    print("\n" + "="*50)
    print("4. Logistic Regression (Built-in)")
    print("="*50)
    logreg = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    logreg.fit(X_train_np, y_train_np)
    y_pred = logreg.predict(X_test_np).tolist()
    acc = accuracy(y_test_list, y_pred)
    prec, prec_macro = precision(y_test_list, y_pred, NUM_CLASSES)
    rec, rec_macro = recall(y_test_list, y_pred, NUM_CLASSES)
    f1 = f1_score(prec_macro, rec_macro)
    print_metrics(y_test_list, y_pred, NUM_CLASSES)
    results['Logistic Regression'] = {'Accuracy': acc, 'Macro Precision': prec_macro, 'Macro Recall': rec_macro, 'Macro F1': f1}

    # 5. KNN
    print("\n" + "="*50)
    print("5. K-Nearest Neighbors (Built-in)")
    print("="*50)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_np, y_train_np)
    y_pred = knn.predict(X_test_np).tolist()
    acc = accuracy(y_test_list, y_pred)
    prec, prec_macro = precision(y_test_list, y_pred, NUM_CLASSES)
    rec, rec_macro = recall(y_test_list, y_pred, NUM_CLASSES)
    f1 = f1_score(prec_macro, rec_macro)
    print_metrics(y_test_list, y_pred, NUM_CLASSES)
    results['KNN'] = {'Accuracy': acc, 'Macro Precision': prec_macro, 'Macro Recall': rec_macro, 'Macro F1': f1}

    # 6. Naive Bayes
    print("\n" + "="*50)
    print("6. Naive Bayes (Built-in)")
    print("="*50)
    nb = GaussianNB()
    nb.fit(X_train_np, y_train_np)
    y_pred = nb.predict(X_test_np).tolist()
    acc = accuracy(y_test_list, y_pred)
    prec, prec_macro = precision(y_test_list, y_pred, NUM_CLASSES)
    rec, rec_macro = recall(y_test_list, y_pred, NUM_CLASSES)
    f1 = f1_score(prec_macro, rec_macro)
    print_metrics(y_test_list, y_pred, NUM_CLASSES)
    results['Naive Bayes'] = {'Accuracy': acc, 'Macro Precision': prec_macro, 'Macro Recall': rec_macro, 'Macro F1': f1}

    return results


def run_scratch_models(X_train, X_test, y_train, y_test):
    """Run all 6 from-scratch models and return results."""
    from models.decision_tree import DecisionTree
    from models.random_forest import RandomForest
    from models.linear_regression import OneVsRestLinearRegression
    from models.logistic_regression import LogisticRegressionScratch
    from models.knn import KNN
    from models.naive_bayes import NaiveBayes

    y_test_list = list(y_test)
    results = {}

    # 1. Decision Tree
    print("\n" + "="*50)
    print("1. Decision Tree (From Scratch)")
    print("="*50)
    dt = DecisionTree(max_depth=5)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = accuracy(y_test_list, y_pred)
    prec, prec_macro = precision(y_test_list, y_pred, NUM_CLASSES)
    rec, rec_macro = recall(y_test_list, y_pred, NUM_CLASSES)
    f1 = f1_score(prec_macro, rec_macro)
    print_metrics(y_test_list, y_pred, NUM_CLASSES)
    results['Decision Tree'] = {'Accuracy': acc, 'Macro Precision': prec_macro, 'Macro Recall': rec_macro, 'Macro F1': f1}

    # 2. Random Forest
    print("\n" + "="*50)
    print("2. Random Forest (From Scratch)")
    print("="*50)
    rf = RandomForest(n_trees=10, max_depth=5)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy(y_test_list, y_pred)
    prec, prec_macro = precision(y_test_list, y_pred, NUM_CLASSES)
    rec, rec_macro = recall(y_test_list, y_pred, NUM_CLASSES)
    f1 = f1_score(prec_macro, rec_macro)
    print_metrics(y_test_list, y_pred, NUM_CLASSES)
    results['Random Forest'] = {'Accuracy': acc, 'Macro Precision': prec_macro, 'Macro Recall': rec_macro, 'Macro F1': f1}

    # 3. Linear Regression (One-vs-Rest)
    print("\n" + "="*50)
    print("3. Linear Regression (From Scratch)")
    print("="*50)
    lr = OneVsRestLinearRegression(num_classes=NUM_CLASSES)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc = accuracy(y_test_list, y_pred)
    prec, prec_macro = precision(y_test_list, y_pred, NUM_CLASSES)
    rec, rec_macro = recall(y_test_list, y_pred, NUM_CLASSES)
    f1 = f1_score(prec_macro, rec_macro)
    print_metrics(y_test_list, y_pred, NUM_CLASSES)
    results['Linear Regression'] = {'Accuracy': acc, 'Macro Precision': prec_macro, 'Macro Recall': rec_macro, 'Macro F1': f1}

    # 4. Logistic Regression
    print("\n" + "="*50)
    print("4. Logistic Regression (From Scratch)")
    print("="*50)
    logreg = LogisticRegressionScratch(num_classes=NUM_CLASSES, learning_rate=0.01, epochs=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    acc = accuracy(y_test_list, y_pred)
    prec, prec_macro = precision(y_test_list, y_pred, NUM_CLASSES)
    rec, rec_macro = recall(y_test_list, y_pred, NUM_CLASSES)
    f1 = f1_score(prec_macro, rec_macro)
    print_metrics(y_test_list, y_pred, NUM_CLASSES)
    results['Logistic Regression'] = {'Accuracy': acc, 'Macro Precision': prec_macro, 'Macro Recall': rec_macro, 'Macro F1': f1}

    # 5. KNN
    print("\n" + "="*50)
    print("5. K-Nearest Neighbors (From Scratch)")
    print("="*50)
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy(y_test_list, y_pred)
    prec, prec_macro = precision(y_test_list, y_pred, NUM_CLASSES)
    rec, rec_macro = recall(y_test_list, y_pred, NUM_CLASSES)
    f1 = f1_score(prec_macro, rec_macro)
    print_metrics(y_test_list, y_pred, NUM_CLASSES)
    results['KNN'] = {'Accuracy': acc, 'Macro Precision': prec_macro, 'Macro Recall': rec_macro, 'Macro F1': f1}

    # 6. Naive Bayes
    print("\n" + "="*50)
    print("6. Naive Bayes (From Scratch)")
    print("="*50)
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    acc = accuracy(y_test_list, y_pred)
    prec, prec_macro = precision(y_test_list, y_pred, NUM_CLASSES)
    rec, rec_macro = recall(y_test_list, y_pred, NUM_CLASSES)
    f1 = f1_score(prec_macro, rec_macro)
    print_metrics(y_test_list, y_pred, NUM_CLASSES)
    results['Naive Bayes'] = {'Accuracy': acc, 'Macro Precision': prec_macro, 'Macro Recall': rec_macro, 'Macro F1': f1}

    return results


def save_results(builtin_results, scratch_results):
    """Save results to JSON files."""
    import os
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'built_in_results.json'), 'w') as f:
        json.dump(builtin_results, f, indent=2)
    with open(os.path.join(results_dir, 'from_scratch_results.json'), 'w') as f:
        json.dump(scratch_results, f, indent=2)
    print("\n" + "="*60)
    print(f"Results saved to {results_dir}/")
    print("="*60)


def print_comparison_table(builtin_results, scratch_results):
    """Print comparison table of all models."""
    print("\n" + "="*100)
    print("COMPARISON TABLE - BUILT-IN vs FROM SCRATCH")
    print("="*100)
    print(f"{'Model':<25} {'Type':<15} {'Accuracy':<12} {'Macro Prec':<12} {'Macro Rec':<12} {'Macro F1':<12}")
    print("-"*100)

    for model in builtin_results.keys():
        b = builtin_results[model]
        s = scratch_results[model]
        print(f"{model:<25} {'Built-in':<15} {b['Accuracy']:<12.4f} {b['Macro Precision']:<12.4f} {b['Macro Recall']:<12.4f} {b['Macro F1']:<12.4f}")
        print(f"{'':<25} {'From Scratch':<15} {s['Accuracy']:<12.4f} {s['Macro Precision']:<12.4f} {s['Macro Recall']:<12.4f} {s['Macro F1']:<12.4f}")
        print("-"*100)

    # Find best model overall
    best_model = None
    best_f1 = -1
    best_type = None
    for model, metrics in builtin_results.items():
        if metrics['Macro F1'] > best_f1:
            best_f1 = metrics['Macro F1']
            best_model = model
            best_type = 'Built-in'
    for model, metrics in scratch_results.items():
        if metrics['Macro F1'] > best_f1:
            best_f1 = metrics['Macro F1']
            best_model = model
            best_type = 'From Scratch'

    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model} ({best_type}) - Macro F1: {best_f1:.4f}")
    print("="*60)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess()

    print("\n" + "="*60)
    print("RUNNING BUILT-IN MODELS")
    print("="*60)
    builtin_results = run_builtin_models(X_train, X_test, y_train, y_test)

    print("\n" + "="*60)
    print("RUNNING FROM-SCRATCH MODELS")
    print("="*60)
    scratch_results = run_scratch_models(X_train, X_test, y_train, y_test)

    save_results(builtin_results, scratch_results)
    print_comparison_table(builtin_results, scratch_results)
