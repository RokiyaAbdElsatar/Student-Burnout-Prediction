import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

st.title("Dataset Information")
st.subheader("Dataset Overview")
st.markdown("""
**Source:** Pain Dataset (Kaggle) - `pain_dataset_200P_4hz.csv`
**Description:** Physiological time-series data from 200 persons.
**Target:** `pain_scale` → 1 to 8 (8 classes)
**Features:** acc_x, acc_y, acc_z, eda, bvp, hr, temp (7 sensors)
**Preprocessing:** 28 features per person (mean, std, min, max)
""")

st.subheader("Raw Data Sample (First 10 rows)")
raw_df = pd.read_csv('../dataset/pain_dataset_200P_4hz.csv', nrows=10)
st.dataframe(raw_df, use_container_width=True)

st.subheader("Aggregated Data Sample (5 persons)")
from utils.data_loader import load_raw_data, aggregate_per_person, encode_labels
header, rows = load_raw_data('../dataset/pain_dataset_200P_4hz.csv')
X, y = aggregate_per_person(rows)
agg_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(len(X[0]))])
agg_df['pain_scale'] = y
st.dataframe(agg_df.head(), use_container_width=True)

st.subheader("Class Distribution (pain_scale)")
fig, ax = plt.subplots(figsize=(10, 5))
class_counts = pd.Series(y).value_counts().sort_index()
ax.bar(class_counts.index, class_counts.values, color='skyblue', edgecolor='navy')
ax.set_xlabel('Pain Scale (1-8)')
ax.set_ylabel('Count')
ax.set_title('Class Distribution')
st.pyplot(fig)
