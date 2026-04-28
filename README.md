# Student Burnout Prediction

## Dataset

**Source:** Pain Dataset (Kaggle) - `pain_dataset_200P_4hz.csv`

**Description:** Physiological time-series data from 200 persons with sensor measurements.

**Target Variable:**
- `pain_scale` → 1 to 8 (multi-class classification, 8 classes)

**Features (7 sensor measurements per person):**
- `acc_x, acc_y, acc_z` - Accelerometer readings
- `eda` - Electrodermal activity
- `bvp` - Blood volume pulse
- `hr` - Heart rate
- `temp` - Temperature

**Preprocessing:** Data aggregated per person (mean, std, min, max for each sensor) → 28 features per person.

## Algorithms

- Decision Tree
- Random Forest
- Linear Regression
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes

## Supporting Files

### `utils/data_loader.py` - Handles data loading and preprocessing
- `load_raw_data(filepath)` - Load CSV data and return header and rows
- `aggregate_per_person(raw_data)` - Aggregate time-series data per person (mean, std, min, max for each sensor)
- `split_data(X, y, test_size=0.2)` - Split data into train and test sets
- `normalize(X_train, X_test)` - Apply z-score normalization using training statistics
- `encode_labels(y)` - Encode pain_scale (1-8) to 0-indexed labels

### `utils/metrics.py` - Implements evaluation metrics
- `accuracy(y_true, y_pred)` - Calculate accuracy
- `confusion_matrix(y_true, y_pred, num_classes)` - Generate confusion matrix
- `precision(y_true, y_pred, num_classes)` - Calculate per-class and macro-average precision
- `recall(y_true, y_pred, num_classes)` - Calculate per-class and macro-average recall
- `f1_score(prec, rec)` - Calculate F1 score from precision and recall
- `print_metrics(y_true, y_pred, num_classes)` - Print all metrics in a formatted table

## Results

### Best Model: Logistic Regression (From Scratch)
- **Macro F1-Score**: 0.4042
- **Accuracy**: 0.4250
- **Macro Precision**: 0.4062
- **Macro Recall**: 0.4021

### Comparison Table: Built-in vs From Scratch

| Model | Type | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|------|----------|----------------|--------------|----------|
| Decision Tree | Built-in | 0.4250 | 0.4007 | 0.3979 | 0.3993 |
| Decision Tree | From Scratch | 0.3250 | 0.3219 | 0.2625 | 0.2892 |
| Random Forest | Built-in | 0.4250 | 0.3006 | 0.3396 | 0.3189 |
| Random Forest | From Scratch | 0.4250 | 0.3329 | 0.3688 | 0.3499 |
| Linear Regression | Both | 0.4500 | 0.2923 | 0.3750 | 0.3285 |
| Logistic Regression | Built-in | 0.4250 | 0.3482 | 0.3875 | 0.3668 |
| **Logistic Regression** | **From Scratch** | **0.4250** | **0.4062** | **0.4021** | **0.4042** |
| KNN | Built-in | 0.2000 | 0.1531 | 0.1833 | 0.1669 |
| KNN | From Scratch | 0.1750 | 0.1442 | 0.1500 | 0.1471 |
| Naive Bayes | Built-in | 0.3750 | 0.3042 | 0.3500 | 0.3255 |
| Naive Bayes | From Scratch | 0.3750 | 0.3069 | 0.3500 | 0.3271 |

### Why Logistic Regression (From Scratch) Performed Best

1. **Probabilistic Nature**: Softmax activation provides well-calibrated probabilities for 8-class classification
2. **Gradient Descent Optimization**: 1000 epochs with lr=0.01 converged well for 160 training samples
3. **No Overfitting**: Simple model with 28 features and 160 samples
4. **Linear Relationships**: Aggregated sensor features (mean, std, min, max) have approximately linear relationships with pain_scale

### Worst Performance: KNN
- High dimensionality (28 features) dilutes distance metric
- Small dataset (160 training samples after aggregation)
- Sensor noise affects distance calculations

For detailed analysis, see `results/best_model_analysis.md`.

## How to Contribute

1. Create a new branch named after the algorithm (e.g., `decision-tree`, `knn`).
2. Apply the algorithm code in your branch.
3. Create a pull request to merge your changes.

## How to Run

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn numpy

# Run all models
python main.py
```

Results will be saved to `results/` directory.
