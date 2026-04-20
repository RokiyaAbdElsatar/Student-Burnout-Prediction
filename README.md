# 🎓 Student Burnout Prediction – Machine Learning Project

## 📌 Project Overview

This project aims to predict **student burnout levels** (Low, Medium, High) using machine learning models built **from scratch (no ready-made ML libraries)**.

The goal is not only to achieve good accuracy, but also to:

* Understand how ML algorithms work internally
* Compare different models fairly
* Build a clean, structured, and professional ML pipeline

---

## 📊 Dataset Description

We are using the **Student Mental Health and Burnout Dataset** from Kaggle.

### 🔹 Target Variable

* `burnout_level` → (Low, Medium, High)
  👉 This makes it a **multi-class classification problem**

### 🔹 Feature Categories

1. **Demographics**

   * age, gender, course, year

2. **Academic**

   * CGPA, attendance, study_hours

3. **Psychological**

   * anxiety_score, depression_score, stress_level

4. **Lifestyle**

   * sleep_hours, physical_activity, screen_time

5. **Social / Financial**

   * financial_stress, social_pressure

⚠️ Note: The dataset is **synthetic**, so results may not fully reflect real-world scenarios.

---

## 🛠️ What Will Be Implemented

### 🔹 Models (Built from Scratch)

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Naive Bayes
* (Optional) Linear Regression for experimentation

---

### 🔹 Core Components

* Data preprocessing (cleaning, encoding, normalization)
* Training algorithms (manual implementation)
* Evaluation metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix

---

### 🔹 Advanced Part (Bonus / Professional Level)

* Ensemble Methods:

  * Voting (Hard & Soft)
  * Weighted Voting
* Feature importance analysis
* Overfitting detection (train vs test comparison)

---

## 📂 Project Structure

```
project/
│
├── dataset/                # Dataset files
├── models/              # ML models (from scratch)
│   ├── logistic.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── naive_bayes.py
│
├── utils/               # Helper functions
│   ├── preprocessing.py
│   ├── metrics.py
│
├── ensemble/            # Hybrid models
│   ├── voting.py
│
├── results/              
│   ├── metrics.json    # For storing the final result (ACC , Recall ..etc)
│   ├── comparison.csv
│
├── visualization/       # Data visualization & plots
│   ├── visualize.py
│
├── gui/                 # GUI application
│   ├── app.py
│
├── notebooks/           # EDA & experiments
├── main.py              # Run training & evaluation
├── README.md
```

---

## 🚀 Workflow (How the Project Will Progress)

### 1️⃣ Data Understanding (EDA)

* Explore dataset
* Visualize distributions
* Check correlations

---

### 2️⃣ Preprocessing

* Handle missing values (if any)
* Encode categorical features
* Normalize numerical features

---

### 3️⃣ Model Implementation

* Build each algorithm from scratch
* Train on dataset
* Generate predictions

---

### 4️⃣ Evaluation

* Compare models using metrics
* Analyze strengths & weaknesses

---

### 5️⃣ Ensemble (Hybrid Model)

* Combine models using Voting
* Compare performance with single models

---

### 6️⃣ Final Comparison

* Create comparison table
* Select best-performing approach

---

## 📈 Expected Output

* A complete comparison between models
* Insights about what affects student burnout
* A clean and structured ML project

---

## 💡 Optional Enhancements

* Simple UI using Streamlit or Flask
* Model visualization
* Feature importance graphs

---

## 👥 Team Notes

* Each member can take responsibility for one model
* Follow the same coding style and structure
* Document your work clearly

---

## ✅ Goal

Build a **professional, well-structured ML project** that demonstrates:

* Strong understanding of algorithms
* Clean implementation
* Clear analysis and comparison

---

