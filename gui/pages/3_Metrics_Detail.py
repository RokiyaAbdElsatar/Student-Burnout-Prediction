import streamlit as st
import pandas as pd
import numpy as np
import json

st.title("Detailed Metrics")
comparison_df = pd.read_csv('../results/comparison.csv')

with open('../results/built_in_results.json', 'r') as f:
    built_in_results = json.load(f)
with open('../results/from_scratch_results.json', 'r') as f:
    scratch_results = json.load(f)

model_options = [f"{row['Model']} ({row['Type']})" for _, row in comparison_df.iterrows()]
selected = st.selectbox("Select Model:", model_options)
model_name = selected.split(' (')[0]
model_type = selected.split('(')[1].replace(')', '')

if model_type == 'Built-in':
    metrics = built_in_results[model_name]
else:
    metrics = scratch_results[model_name]

st.subheader(f"Metrics: {model_name} ({model_type})")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
with col2:
    st.metric("Macro Precision", f"{metrics['Macro Precision']:.4f}")
with col3:
    st.metric("Macro Recall", f"{metrics['Macro Recall']:.4f}")
with col4:
    is_best = metrics['Macro F1'] == comparison_df['Macro F1'].max()
    st.metric("Macro F1", f"{metrics['Macro F1']:.4f}", "Best!" if is_best else None)

st.subheader("Confusion Matrix (Simulated)")
cm = np.random.randint(0, 5, size=(8, 8))
np.fill_diagonal(cm, np.random.randint(3, 8, size=8))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(8))
ax.set_yticks(range(8))
ax.set_xticklabels(range(8))
ax.set_yticklabels(range(8))
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
for i in range(8):
    for j in range(8):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax)
st.pyplot(fig)

st.subheader("Comparison to Best Model (Logistic Regression Scratch)")
best_f1 = 0.4042
diff = metrics['Macro F1'] - best_f1
col1, col2 = st.columns(2)
with col1:
    st.metric("This Model's F1", f"{metrics['Macro F1']:.4f}")
with col2:
    st.metric("Difference from Best", f"{diff:.4f}", f"{diff:.4f}")
if diff >= 0:
    st.success(f"✅ This model matches the best performance!")
else:
    st.warning(f"⚠️ This model is {abs(diff):.4f} below the best model.")
