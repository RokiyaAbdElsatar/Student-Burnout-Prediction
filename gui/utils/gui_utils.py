import pandas as pd
import json
import os

def get_project_root():
    """Get absolute path to project root."""
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def load_comparison_data():
    """Load model comparison data from CSV."""
    csv_path = os.path.join(get_project_root(), 'results', 'comparison.csv')
    return pd.read_csv(csv_path)

def load_results_json(result_type='built_in'):
    """Load results from JSON file."""
    if result_type == 'built_in':
        path = os.path.join(get_project_root(), 'results', 'built_in_results.json')
    else:
        path = os.path.join(get_project_root(), 'results', 'from_scratch_results.json')
    with open(path, 'r') as f:
        return json.load(f)

def load_markdown_file(filename):
    """Load markdown file content."""
    path = os.path.join(get_project_root(), filename)
    with open(path, 'r') as f:
        return f.read()

def load_dataset_sample(nrows=10):
    """Load raw dataset sample."""
    path = os.path.join(get_project_root(), 'dataset', 'pain_dataset_200P_4hz.csv')
    return pd.read_csv(path, nrows=nrows)

def load_aggregated_data():
    """Load and aggregate dataset."""
    sys_path = os.path.join(get_project_root(), 'utils')
    import sys
    if sys_path not in sys.path:
        sys.path.append(sys_path)
    from data_loader import load_raw_data, aggregate_per_person, encode_labels
    header, rows = load_raw_data(os.path.join(get_project_root(), 'dataset', 'pain_dataset_200P_4hz.csv'))
    X, y = aggregate_per_person(rows)
    return X, y

def get_best_model():
    """Return best model info."""
    return {
        'name': 'Logistic Regression (From Scratch)',
        'accuracy': 0.4250,
        'macro_precision': 0.4062,
        'macro_recall': 0.4021,
        'macro_f1': 0.4042
    }
