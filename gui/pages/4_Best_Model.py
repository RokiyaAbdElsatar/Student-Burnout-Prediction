import streamlit as st
import pandas as pd

st.title("Best Model Analysis")
st.subheader("Best Model: Logistic Regression (From Scratch)")
st.markdown("**Macro F1-Score: 0.4042**")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", "0.4250", "Best!")
with col2:
    st.metric("Macro Precision", "0.4062", "Best!")
with col3:
    st.metric("Macro Recall", "0.4021", "Best!")
with col4:
    st.metric("Macro F1", "0.4042", "Best Overall!")

with open('../results/best_model_analysis.md', 'r') as f:
    st.markdown(f.read())

st.markdown("---")
st.subheader("All Models Ranking (by Macro F1)")
comparison_df = pd.read_csv('../results/comparison.csv')
ranked = comparison_df.sort_values('Macro F1', ascending=False).reset_index(drop=True)
ranked.index += 1
ranked.index.name = 'Rank'

def highlight_rank(row):
    if row.name == 1:
        return ['background-color: #1a472a; color: #90EE90'] * len(row)
    return [''] * len(row)

st.dataframe(ranked.style.apply(highlight_rank, axis=1), use_container_width=True)
