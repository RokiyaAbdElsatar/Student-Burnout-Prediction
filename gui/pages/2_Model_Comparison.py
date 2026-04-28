import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Model Comparison: Built-in vs From Scratch")
comparison_df = pd.read_csv('../results/comparison.csv')

def highlight_best(row):
    if row['Model'] == 'Logistic Regression' and row['Type'] == 'From Scratch':
        return ['background-color: #1a472a; color: #90EE90'] * len(row)
    return [''] * len(row)

st.subheader("Comparison Table")
styled_df = comparison_df.style.apply(highlight_best, axis=1)
st.dataframe(styled_df, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Macro F1 Comparison")
    fig, ax = plt.subplots(figsize=(10, 8))
    models = comparison_df['Model'] + '\n(' + comparison_df['Type'] + ')'
    colors = ['#FF4B4B' if t == 'Built-in' else '#4B4BFF' for t in comparison_df['Type']]
    bars = ax.barh(range(len(models)), comparison_df['Macro F1'], color=colors)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel('Macro F1 Score')
    best_idx = comparison_df['Macro F1'].idxmax()
    bars[best_idx].set_edgecolor('lime')
    bars[best_idx].set_linewidth(3)
    st.pyplot(fig)

with col2:
    st.subheader("Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(models)), comparison_df['Accuracy'], color=colors)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel('Accuracy')
    best_idx = comparison_df['Accuracy'].idxmax()
    bars[best_idx].set_edgecolor('lime')
    bars[best_idx].set_linewidth(3)
    st.pyplot(fig)

st.subheader("Summary Statistics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Best Macro F1", f"{comparison_df['Macro F1'].max():.4f}")
with col2:
    st.metric("Worst Macro F1", f"{comparison_df['Macro F1'].min():.4f}")
with col3:
    best_model = comparison_df.loc[comparison_df['Macro F1'].idxmax(), 'Model']
    st.metric("Best Model", best_model)
with col4:
    st.metric("Total Models", len(comparison_df))
