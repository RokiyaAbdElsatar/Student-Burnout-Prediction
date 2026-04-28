#   

## Dataset

**Source:** Student Mental Health and Burnout Dataset (Kaggle)

**File:** `dataset/pain_dataset_200P_4hz.csv`

**Target Variable:**
- `burnout_level` → Low, Medium, High (multi-class classification)

**Features:**

| Category | Features |
|----------|----------|
| Demographics | age, gender, course, year |
| Academic | CGPA, attendance, study_hours |
| Psychological | anxiety_score, depression_score, stress_level |
| Lifestyle | sleep_hours, physical_activity, screen_time |
| Social / Financial | financial_stress, social_pressure |

**Note:** This dataset is synthetic, so results may not fully reflect real-world scenarios.

## Algorithms

- Decision Tree
- Random Forest
- Linear Regression
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes

## How to Contribute

1. Create a new branch named after the algorithm (e.g., `decision-tree`, `knn`).
2. Apply the algorithm code in your branch.
3. Create a pull request to merge your changes.
