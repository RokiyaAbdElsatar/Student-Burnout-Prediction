import pandas as pd
import json
import os

def load_comparison_data():
    """Load model comparison data from CSV."""
    csv_path = '../results/comparison.csv'
    if not os.path.exists(csv_path):
        csv_path = 'results/comparison.csv'
    return pd.read_csv(csv_path)

def load_results_json(result_type='built_in'):
    """Load results from JSON file."""
    if result_type == 'built_in':
        path = '../results/built_in_results.json'
    else:
        path = '../results/from_scratch_results.json'

    if not os.path.exists(path):
        path = path.replace('../', '')

    with open(path, 'r') as f:
        return json.load(f)

def load_markdown_file(filename):
    """Load markdown file content."""
    path = f'../{filename}'
    if not os.path.exists(path):
        path = filename
    with open(path, 'r') as f:
        return f.read()

def get_best_model():
    """Return best model info."""
    return {
        'name': 'Logistic Regression (From Scratch)',
        'accuracy': 0.4250,
        'macro_precision': 0.4062,
        'macro_recall': 0.4021,
        'macro_f1': 0.4042
    }
