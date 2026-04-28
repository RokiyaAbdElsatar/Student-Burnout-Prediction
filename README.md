# Student Burnout Prediction

A machine learning project that predicts student burnout using physiological sensor data. Implements 6 algorithms from scratch and compares them with built-in libraries.

## Quick Start

### Option 1: Run GUI (Recommended)

```bash
# Clone the repository
git clone https://github.com/RokiyaAbdElsatar/Student-Burnout-Prediction.git
cd Student-Burnout-Prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install streamlit scikit-learn numpy pandas plotly

# Launch the modern GUI with interactive charts
streamlit run gui/app.py
```

Or use the convenience script:

```bash
./start_gui.sh
```

### Option 2: Run Models from Command Line

```bash
# Setup (if not done already)
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn numpy

# Run all 12 models (6 built-in + 6 from scratch)
python main.py
```

Results will be saved to `results/` directory.

## GUI Features

The modern GUI includes:

- **Dashboard**: Interactive charts showing model performance, dataset distribution, and metrics
- **Dataset Info**: Visualizations of sensor data and class distribution
- **Model Comparison**: Interactive charts comparing built-in vs from-scratch implementations
- **Metrics Detail**: Radar charts and heatmaps for each model
- **Best Model Analysis**: Performance breakdowns and why it won
- **Retrain Models**: One-click retraining with progress tracking
- **Export Report**: Download HTML reports with embedded interactive charts

## Project Structure

```
Student-Burnout-Prediction/
├── dataset/                    # Dataset files
│   └── pain_dataset_200P_4hz.csv
├── gui/                        # Modern Streamlit GUI
│   ├── app.py                 # Main application with Plotly charts
│   ├── pages/                 # GUI pages
│   └── utils/                 # GUI utilities
├── models/                     # ML models from scratch
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── knn.py
│   └── naive_bayes.py
├── utils/                      # Data processing utilities
│   ├── data_loader.py
│   └── metrics.py
├── results/                    # Output files
├── main.py                     # Run all models
├── start_gui.sh               # GUI startup script
└── README.md
```

## Requirements

- Python 3.7+
- streamlit
- scikit-learn
- numpy
- pandas
- plotly

## Models Implemented

| Model | Built-in | From Scratch |
|-------|----------|--------------|
| Decision Tree | ✅ | ✅ |
| Random Forest | ✅ | ✅ |
| Linear Regression | ✅ | ✅ |
| Logistic Regression | ✅ | ✅ |
| K-Nearest Neighbors | ✅ | ✅ |
| Naive Bayes | ✅ | ✅ |

## Results Summary

**Best Model**: Logistic Regression (From Scratch)
- Macro F1-Score: **0.4042**
- Accuracy: **0.4250**
- Macro Precision: **0.4062**
- Macro Recall: **0.4021**

View full results in the GUI or check `results/` directory.

## Dataset

**Source**: Pain Dataset (Kaggle) - `pain_dataset_200P_4hz.csv`

- **Target**: `pain_scale` (1-8, 8 classes)
- **Features**: 7 sensor measurements (acc_x, acc_y, acc_z, eda, bvp, hr, temp)
- **Preprocessing**: 28 features per person (mean, std, min, max for each sensor)
- **Samples**: ~200 persons, aggregated to ~160 training samples

## Contributing

1. Create a branch: `git checkout -b feature-name`
2. Make changes and commit: `git commit -m "Description"`
3. Push to branch: `git push origin feature-name`
4. Create a Pull Request

## License

This project is open source.
