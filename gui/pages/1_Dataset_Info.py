import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
st.title("Dataset Information")

# Quick Stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Dataset", "Pain Dataset")
with col2:
    st.metric("👥 Persons", "200")
with col3:
    st.metric("🎯 Classes", "8 (pain_scale)")

st.markdown("---")

# Interactive Raw Data Sample
st.subheader("📈 Raw Data Preview (Interactive)")
try:
    from utils.gui_utils import load_dataset_sample
    raw_df = load_dataset_sample(nrows=50)
    st.dataframe(raw_df, use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"Error: {e}")

# Interactive Correlation Heatmap
st.subheader("🎨 Feature Correlation Heatmap")
try:
    from utils.gui_utils import load_aggregated_data
    X, y = load_aggregated_data()
    agg_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(len(X[0]))])

    fig = px.imshow(
        agg_df.corr(),
        title='Feature Correlation Matrix (Interactive)',
        color_continuous_scale='Blues',
        aspect='auto'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error: {e}")

# Class Distribution - Interactive
st.subheader("📊 Class Distribution (Interactive)")
try:
    X, y = load_aggregated_data()
    class_counts = pd.Series(y).value_counts().sort_index().reset_index()
    class_counts.columns = ['pain_scale', 'count']

    fig = px.bar(
        class_counts,
        x='pain_scale',
        y='count',
        title='Class Distribution (pain_scale 1-8)',
        color='count',
        color_continuous_scale='Viridis',
        hover_data=['count']
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_title='Pain Scale (1-8)',
        yaxis_title='Count'
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error: {e}")
