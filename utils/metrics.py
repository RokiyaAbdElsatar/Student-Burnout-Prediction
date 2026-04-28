def accuracy(y_true, y_pred):
    """Calculate accuracy."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, num_classes):
    """Generate confusion matrix."""
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def precision(y_true, y_pred, num_classes):
    """Calculate per-class and macro-average precision."""
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    precisions = []
    for i in range(num_classes):
        tp = matrix[i][i]
        total_pred = sum(matrix[j][i] for j in range(num_classes))
        if total_pred == 0:
            precisions.append(0.0)
        else:
            precisions.append(tp / total_pred)
    macro = sum(precisions) / num_classes
    return precisions, macro


def recall(y_true, y_pred, num_classes):
    """Calculate per-class and macro-average recall."""
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    recalls = []
    for i in range(num_classes):
        tp = matrix[i][i]
        total_true = sum(matrix[i])
        if total_true == 0:
            recalls.append(0.0)
        else:
            recalls.append(tp / total_true)
    macro = sum(recalls) / num_classes
    return recalls, macro


def f1_score(prec, rec):
    """Calculate F1 score from precision and recall."""
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def print_metrics(y_true, y_pred, num_classes):
    """Print all metrics in a formatted table."""
    acc = accuracy(y_true, y_pred)
    prec_list, prec_macro = precision(y_true, y_pred, num_classes)
    rec_list, rec_macro = recall(y_true, y_pred, num_classes)
    f1_list = [f1_score(p, r) for p, r in zip(prec_list, rec_list)]
    f1_macro = f1_score(prec_macro, rec_macro)

    print("=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {prec_macro:.4f}")
    print(f"Macro Recall: {rec_macro:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print()
    print("Per-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    for i in range(num_classes):
        print(f"{i:<10} {prec_list[i]:<12.4f} {rec_list[i]:<12.4f} {f1_list[i]:<12.4f}")
    print("=" * 60)

    print()
    print("Confusion Matrix:")
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    print("    ", end="")
    for i in range(num_classes):
        print(f"{i:>4}", end="")
    print()
    for i in range(num_classes):
        print(f"{i:>3} ", end="")
        for j in range(num_classes):
            print(f"{matrix[i][j]:>4}", end="")
        print()
