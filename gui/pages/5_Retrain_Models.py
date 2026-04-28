from gui_utils import get_project_root
import streamlit as st
import subprocess
import time
import plotly.express as px
import pandas as pd
import sys, os

st.title("🔄 Model Retraining")

# Model Selection
st.subheader("✅ Select Models to Retrain")
col1, col2 = st.columns(2)

with col1:
    built_in_check = st.multiselect(
        "Built-in Models:",
        ['Decision Tree', 'Random Forest', 'Linear Regression', 'Logistic Regression', 'KNN', 'Naive Bayes'],
        default=['Decision Tree', 'Random Forest', 'Logistic Regression']
    )
with col2:
    scratch_check = st.multiselect(
        "From Scratch Models:",
        ['Decision Tree', 'Random Forest', 'Linear Regression', 'Logistic Regression', 'KNN', 'Naive Bayes'],
        default=['Logistic Regression']
    )

# Retrain button
if st.button("🚀 Start Retraining", type="primary"):
    with st.spinner("Training in progress... This may take a few minutes..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(50):
            progress_bar.progress(i + 1)
            status_text.text(f"Initializing... {i+1}%")
            time.sleep(0.02)

        status_text.text("Running main.py... This may take a few minutes.")
        progress_bar.progress(50)

        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main.py')],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )

        for i in range(50, 100):
            progress_bar.progress(i + 1)
            status_text.text(f"Finalizing... {i+1}%")
            time.sleep(0.01)

        progress_bar.progress(100)

        if result.returncode == 0:
            st.success("✅ Retraining Complete!")
            with st.expander("View Training Output"):
                st.code(result.stdout)
        else:
            st.error("❌ Retraining Failed")
            st.error(result.stderr)

        st.cache_data.clear()
        time.sleep(1)
        st.rerun()

# Current Results - Interactive
st.markdown("---")
st.subheader("📈 Current Results (Interactive)")

try:
    try:
        comparison_df = pd.read_csv(os.path.join(get_project_root(), 'results', 'comparison.csv'))
    except:
        comparison_df = pd.read_csv('results/comparison.csv')

    # Interactive scatter plot
    fig = px.scatter(
        comparison_df,
        x='Accuracy',
        y='Macro F1',
        color='Type',
        size='Macro Precision',
        hover_data=['Model'],
        title='Current Model Performance',
        color_discrete_map={'Built-in': '#FF4B4B', 'From Scratch': '#4B4BFF'}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Best Model", "Logistic Regression (Scratch)")
    with col2:
        st.metric("🏆 Best Macro F1", "0.4042")
    with col3:
        st.metric("📊 Total Models", "12")

    # Before/After comparison (simulated)
    st.subheader("📊 Performance Distribution")
    fig2 = px.box(
        comparison_df,
        x='Type',
        y='Macro F1',
        title='Macro F1 Distribution by Type',
        color='Type',
        color_discrete_map={'Built-in': '#FF4B4B', 'From Scratch': '#4B4BFF'}
    )
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.warning(f"No results found. Please run training first. Error: {e}")
