# Best Model Analysis

## Best Model: Logistic Regression (From Scratch)
- **Macro F1-Score**: 0.4042
- **Accuracy**: 0.4250
- **Macro Precision**: 0.4062
- **Macro Recall**: 0.4021

## Performance Comparison

| Model | Type | Macro F1 | Difference from Best |
|-------|------|----------|---------------------|
| Logistic Regression (Scratch) | From Scratch | 0.4042 | - |
| Decision Tree (Built-in) | Built-in | 0.3993 | -0.0049 |
| Logistic Regression (Built-in) | Built-in | 0.3668 | -0.0374 |
| Random Forest (Scratch) | From Scratch | 0.3499 | -0.0543 |
| Random Forest (Built-in) | Built-in | 0.3189 | -0.0853 |
| Linear Regression (Both) | Both | 0.3285 | -0.0757 |
| Naive Bayes (Scratch) | From Scratch | 0.3271 | -0.0771 |
| Naive Bayes (Built-in) | Built-in | 0.3255 | -0.0787 |
| KNN (Built-in) | Built-in | 0.1669 | -0.2373 |
| KNN (Scratch) | From Scratch | 0.1471 | -0.2571 |

## Why Logistic Regression (From Scratch) Performed Best

### 1. **Probabilistic Nature**
Logistic Regression outputs well-calibrated probabilities via softmax activation, making it naturally suited for multi-class classification with 8 balanced classes.

### 2. **Gradient Descent Optimization**
The from-scratch implementation uses gradient descent with 1000 epochs and learning rate 0.01, which:
- Converges to a good solution for this dataset size (160 training samples)
- Avoids some of the convergence issues that limited the built-in version (which uses L-BFGS solver)

### 3. **No Overfitting**
With 28 features and 160 training samples, Logistic Regression's simplicity prevents overfitting compared to:
- Decision Trees (prone to overfitting without proper pruning)
- Random Forest (can still overfit with small data)

### 4. **Handles Feature Relationships Linearly**
The aggregated sensor features (mean, std, min, max) likely have approximately linear relationships with the pain scale classes, which suits Logistic Regression.

### 5. **Class Imbalance Handling**
The dataset has uneven class distribution (pain_scale 4: 17280 samples, pain_scale 6: 7680 samples). Logistic Regression's probabilistic approach handles this better than:
- KNN (struggles with imbalanced data and high-dimensional feature space)
- Naive Bayes (assumes feature independence, which is violated)

## Why Other Models Underperformed

### KNN (Worst Performance)
- **High dimensionality**: 28 features dilute the distance metric
- **Small dataset**: Only ~160 training samples after aggregation
- **Noisy features**: Sensor noise affects distance calculations

### Decision Tree (Built-in beat Scratch)
- **Built-in**: Better split criteria and pruning
- **From-scratch**: Limited to max_depth=5, information gain heuristic may not find optimal splits

### Random Forest
- **Ensemble benefit**: From-scratch (0.3499) beat built-in (0.3189)
- **Bootstrap sampling**: From-scratch used proper bootstrap, built-in may have different default behavior

### Linear Regression
- **Not designed for classification**: Forced into one-vs-rest setup
- **Output mismatch**: Linear outputs don't represent probabilities well

## Conclusion

Logistic Regression (From Scratch) achieved the best performance because:
1. It's well-suited for multi-class classification with probabilistic outputs
2. The gradient descent implementation converged well for this dataset
3. It avoids overfitting while capturing linear relationships in the aggregated sensor features
4. It handles the 8-class problem better than tree-based methods on this small dataset

**Recommendation**: For production use, consider:
- Collecting more data (currently only ~160 samples after aggregation)
- Feature selection to reduce dimensionality
- Trying ensemble methods with Logistic Regression as base estimator
