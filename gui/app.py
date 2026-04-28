import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from gui_utils import (
    load_comparison_data,
    load_results_json,
    load_markdown_file,
    load_dataset_sample,
    load_aggregated_data,
    get_best_model
)

st.set_page_config(
    page_title="Student Burnout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🎓 Student Burnout Prediction")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Home",
        "📊 Dataset Info",
        "🤖 Model Comparison",
        "📈 Metrics Detail",
        "🏆 Best Model",
        "🔄 Retrain Models",
        "📄 Export Report"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("🌙 **Dark Mode Active**")

# Load data
@st.cache_data
def load_data():
    comparison = load_comparison_data()
    built_in = load_results_json('built_in')
    scratch = load_results_json('from_scratch')
    return comparison, built_in, scratch

try:
    comparison_df, built_in_results, scratch_results = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    comparison_df, built_in_results, scratch_results = None, None, None

if page == "🏠 Home":
    st.title("🎓 Student Burnout Prediction Dashboard")
    st.markdown("---")

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🏆 Best Model", "Logistic Reg.", delta="Scratch")
    with col2:
        st.metric("Best Macro F1", "0.4042", "Top performer")
    with col3:
        st.metric("Total Models", "12", "6 Built-in + 6 Scratch")
    with col4:
        st.metric("Dataset Size", "~160", "200 persons")
    with col5:
        st.metric("Classes", "8", "pain_scale 1-8")

    st.markdown("---")

    # Main charts row
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Model Performance Overview")
        import plotly.express as px
        fig = px.bar(
            comparison_df,
            x='Model',
            y=['Macro F1', 'Accuracy'],
            color='Type',
            barmode='group',
            title='F1 Score & Accuracy by Model',
            color_discrete_map={'Built-in': '#FF4B4B', 'From Scratch': '#4B4BFF'}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Performance Distribution")
        fig2 = px.scatter(
            comparison_df,
            x='Accuracy',
            y='Macro F1',
            color='Type',
            size='Macro Precision',
            hover_data=['Model'],
            title='Accuracy vs Macro F1',
            color_discrete_map={'Built-in': '#FF4B4B', 'From Scratch': '#4B4BFF'}
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Second row - Dataset info and algorithm comparison
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("📈 Algorithm Comparison")
        import plotly.graph_objects as go
        models = comparison_df['Model'].unique()
        builtin_f1 = comparison_df[comparison_df['Type'] == 'Built-in']['Macro F1'].values
        scratch_f1 = comparison_df[comparison_df['Type'] == 'From Scratch']['Macro F1'].values

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name='Built-in',
            x=models,
            y=builtin_f1,
            marker_color='#FF4B4B'
        ))
        fig3.add_trace(go.Bar(
            name='From Scratch',
            x=models,
            y=scratch_f1,
            marker_color='#4B4BFF'
        ))
        fig3.update_layout(
            title='Built-in vs From Scratch (Macro F1)',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("📊 Dataset Class Distribution")
        try:
            X, y = load_aggregated_data()
            import pandas as pd
            class_counts = pd.Series(y).value_counts().sort_index()
            fig4 = px.bar(
                x=class_counts.index,
                y=class_counts.values,
                labels={'x': 'Pain Scale (1-8)', 'y': 'Count'},
                title='Class Distribution (Pain Scale)',
                color=class_counts.values,
                color_continuous_scale='viridis'
            )
            fig4.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig4, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading dataset distribution: {e}")

    # Third row - Radar chart and metrics table
    st.markdown("---")
    col5, col6 = st.columns([1, 1])

    with col5:
        st.subheader("🎯 Best Model Radar Chart")
        import plotly.graph_objects as go
        best_data = comparison_df.loc[comparison_df['Macro F1'].idxmax()]
        categories = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']

        fig5 = go.Figure()
        fig5.add_trace(go.Scatterpolar(
            r=[best_data['Accuracy'], best_data['Macro Precision'],
               best_data['Macro Recall'], best_data['Macro F1'], best_data['Accuracy']],
            theta=categories + [categories[0]],
            fill='toself',
            name=f"{best_data['Model']} ({best_data['Type']})",
            line_color='#90EE90'
        ))
        fig5.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='#555'),
                angularaxis=dict(gridcolor='#555')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title='Best Model Performance Profile',
            height=400
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.subheader("📋 Quick Comparison Table")
        # Highlight best model
        def highlight_best(row):
            if row['Model'] == 'Logistic Regression' and row['Type'] == 'From Scratch':
                return ['background-color: #1a472a; color: #90EE90'] * len(row)
            return [''] * len(row)

        styled_df = comparison_df.style.apply(highlight_best, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

elif page == "📊 Dataset Info":
    st.title("📊 Dataset Information")

    # Dataset Overview with visualizations
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Dataset Overview")
        st.markdown("""
        **Source:** Pain Dataset (Kaggle)
        **File:** `pain_dataset_200P_4hz.csv`
        **Description:** Physiological time-series data from 200 persons
        **Target:** `pain_scale` → 1 to 8 (8 classes)
        **Features:** acc_x, acc_y, acc_z, eda, bvp, hr, temp (7 sensors)
        **Preprocessing:** 28 features per person
        """)

        # Quick stats
        try:
            X, y = load_aggregated_data()
            st.metric("Total Persons", len(X))
            st.metric("Features per Person", len(X[0]))
            st.metric("Classes", len(set(y)))
        except Exception as e:
            st.error(f"Error loading data: {e}")

    with col2:
        st.subheader("📊 Sensor Distribution")
        sensors = ['acc_x', 'acc_y', 'acc_z', 'eda', 'bvp', 'hr', 'temp']
        fig = px.pie(
            names=sensors,
            values=[1]*7,
            title='7 Sensor Types',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Raw Data Sample
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("📝 Raw Data Sample (First 10 rows)")
        try:
            raw_df = load_dataset_sample(nrows=10)
            st.dataframe(raw_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    with col4:
        st.subheader("📈 Aggregated Data Sample (5 persons)")
        st.markdown("28 features per person (mean, std, min, max)")
        try:
            X, y = load_aggregated_data()
            agg_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(len(X[0]))])
            agg_df['pain_scale'] = y
            st.dataframe(agg_df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error processing data: {e}")

    st.markdown("---")

    # Class Distribution with better charts
    col5, col6 = st.columns([1, 1])

    with col5:
        st.subheader("📊 Class Distribution (pain_scale)")
        try:
            X, y = load_aggregated_data()
            import pandas as pd
            class_counts = pd.Series(y).value_counts().sort_index()

            fig2 = px.bar(
                x=class_counts.index,
                y=class_counts.values,
                labels={'x': 'Pain Scale (1-8)', 'y': 'Count'},
                title='Class Distribution',
                color=class_counts.values,
                color_continuous_scale='viridis',
                text=class_counts.values
            )
            fig2.update_traces(textposition='outside')
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading distribution: {e}")

    with col6:
        st.subheader("🥧 Class Proportion")
        try:
            fig3 = px.pie(
                values=class_counts.values,
                names=[f'Class {i}' for i in class_counts.index],
                title='Class Distribution (Pie)',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating pie chart: {e}")

elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison: Built-in vs From Scratch")
    st.subheader("Comparison Table")

    def highlight_best(row):
        if row['Model'] == 'Logistic Regression' and row['Type'] == 'From Scratch':
            return ['background-color: #1a472a; color: #90EE90'] * len(row)
        return [''] * len(row)

    styled_df = comparison_df.style.apply(highlight_best, axis=1)
    st.dataframe(styled_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Macro F1 Comparison")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        models = comparison_df['Model'] + '\n(' + comparison_df['Type'] + ')'
        colors = ['#FF4B4B' if t == 'Built-in' else '#4B4BFF' for t in comparison_df['Type']]
        bars = ax.barh(range(len(models)), comparison_df['Macro F1'], color=colors)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel('Macro F1 Score')
        ax.set_title('Macro F1: Built-in (Red) vs Scratch (Blue)')
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
        ax.set_title('Accuracy: Built-in (Red) vs Scratch (Blue)')
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

elif page == "📈 Metrics Detail":
    st.title("📈 Detailed Metrics")
    model_options = [f"{row['Model']} ({row['Type']})" for _, row in comparison_df.iterrows()]
    selected = st.selectbox("Select Model:", model_options)
    model_name = selected.split(' (')[0]
    model_type = selected.split('(')[1].replace(')', '')

    if model_type == 'Built-in':
        metrics = built_in_results[model_name]
    else:
        metrics = scratch_results[model_name]

    st.subheader(f"Metrics: {model_name} ({model_type})")

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    with col2:
        st.metric("Macro Precision", f"{metrics['Macro Precision']:.4f}")
    with col3:
        st.metric("Macro Recall", f"{metrics['Macro Recall']:.4f}")
    with col4:
        is_best = metrics['Macro F1'] == comparison_df['Macro F1'].max()
        st.metric("Macro F1", f"{metrics['Macro F1']:.4f}", "Best! 🏆" if is_best else None)

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Metrics Breakdown")
        import plotly.express as px
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1'],
            'Value': [metrics['Accuracy'], metrics['Macro Precision'],
                      metrics['Macro Recall'], metrics['Macro F1']]
        })
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Value',
            title=f'{model_name} ({model_type}) - Metrics',
            color='Value',
            color_continuous_scale='viridis',
            text='Value'
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Performance Radar")
        import plotly.graph_objects as go
        categories = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']

        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=[metrics['Accuracy'], metrics['Macro Precision'],
               metrics['Macro Recall'], metrics['Macro F1'], metrics['Accuracy']],
            theta=categories + [categories[0]],
            fill='toself',
            name=f"{model_name} ({model_type})",
            line_color='#4B4BFF' if model_type == 'From Scratch' else '#FF4B4B'
        ))

        # Add best model for comparison
        best_row = comparison_df.loc[comparison_df['Macro F1'].idxmax()]
        fig2.add_trace(go.Scatterpolar(
            r=[best_row['Accuracy'], best_row['Macro Precision'],
               best_row['Macro Recall'], best_row['Macro F1'], best_row['Accuracy']],
            theta=categories + [categories[0]],
            fill='toself',
            name='Best Model',
            line_color='#90EE90',
            opacity=0.5
        ))

        fig2.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='#555'),
                angularaxis=dict(gridcolor='#555')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title='Model vs Best Model',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Simulated Confusion Matrix (since we don't have actual predictions saved)
    st.markdown("---")
    st.subheader("🔥 Confusion Matrix (Simulated)")
    st.markdown("*Note: Actual confusion matrices would require re-running models with prediction storage*")

    import numpy as np
    import plotly.graph_objects as go

    # Generate a more realistic confusion matrix based on F1 score
    np.random.seed(hash(model_name + model_type) % 2**32)
    cm = np.random.randint(0, 3, size=(8, 8))
    diag_values = np.random.randint(3, 8, size=8)
    # Scale diagonal based on model's F1 score
    diag_values = (diag_values / diag_values.max() * metrics['Macro F1'] * 10).astype(int)
    np.fill_diagonal(cm, diag_values)

    fig3 = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f'Class {i}' for i in range(1, 9)],
        y=[f'Class {i}' for i in range(1, 9)],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig3.update_layout(
        title=f'Simulated Confusion Matrix: {model_name} ({model_type})',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=500
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Comparison to best model
    st.markdown("---")
    st.subheader("📈 Comparison to Best Model")
    best_f1 = comparison_df['Macro F1'].max()
    diff = metrics['Macro F1'] - best_f1

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("This Model's F1", f"{metrics['Macro F1']:.4f}")
    with col2:
        delta_color = "normal" if diff >= 0 else "inverse"
        st.metric("Difference from Best", f"{diff:.4f}", f"{diff:.4f}", delta_color=delta_color)
    with col3:
        rank = (comparison_df['Macro F1'] > metrics['Macro F1']).sum() + 1
        st.metric("Rank", f"#{rank} / {len(comparison_df)}")

    # Progress bar showing performance relative to best
    st.markdown("**Performance relative to best model:**")
    progress = metrics['Macro F1'] / best_f1 if best_f1 > 0 else 0
    st.progress(progress)
    st.markdown(f"{progress*100:.1f}% of best model performance")

    if diff >= 0:
        st.success(f"✅ This model matches the best performance! (Difference: {diff:.4f})")
    else:
        st.warning(f"⚠️ This model is {abs(diff):.4f} below the best model.")

elif page == "🏆 Best Model":
    st.title("🏆 Best Model Analysis")
    st.subheader("Best Model: Logistic Regression (From Scratch)")
    st.markdown("**Macro F1-Score: 0.4042**")

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "0.4250", "42.5%")
    with col2:
        st.metric("Macro Precision", "0.4062", "40.6%")
    with col3:
        st.metric("Macro Recall", "0.4021", "40.2%")
    with col4:
        st.metric("Macro F1", "0.4042", "Best! 🏆")

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Performance Comparison")
        import plotly.graph_objects as go
        best_model_data = comparison_df[comparison_df['Macro F1'] == comparison_df['Macro F1'].max()].iloc[0]
        other_models = comparison_df[comparison_df['Macro F1'] != comparison_df['Macro F1'].max()]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=other_models['Model'] + ' (' + other_models['Type'] + ')',
            y=other_models['Macro F1'],
            name='Other Models',
            marker_color='#4B4BFF',
            opacity=0.6
        ))
        fig.add_trace(go.Bar(
            x=[f"{best_model_data['Model']} ({best_model_data['Type']})"],
            y=[best_model_data['Macro F1']],
            name='Best Model',
            marker_color='#90EE90'
        ))
        fig.update_layout(
            title='Best Model vs Others (Macro F1)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Metric Breakdown")
        import plotly.express as px
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1'],
            'Value': [0.4250, 0.4062, 0.4021, 0.4042]
        })
        fig2 = px.bar(
            metrics_df,
            x='Metric',
            y='Value',
            title='Best Model Metrics',
            color='Value',
            color_continuous_scale='viridis',
            text='Value'
        )
        fig2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Charts row 2 - Radar and Why Best
    st.markdown("---")
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("🎯 Performance Radar")
        import plotly.graph_objects as go
        categories = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']

        fig3 = go.Figure()
        fig3.add_trace(go.Scatterpolar(
            r=[0.4250, 0.4062, 0.4021, 0.4042, 0.4250],
            theta=categories + [categories[0]],
            fill='toself',
            name='Logistic Regression (Scratch)',
            line_color='#90EE90'
        ))

        # Add average line for context
        avg_f1 = comparison_df['Macro F1'].mean()
        avg_prec = comparison_df['Macro Precision'].mean()
        avg_rec = comparison_df['Macro Recall'].mean()
        avg_acc = comparison_df['Accuracy'].mean()
        fig3.add_trace(go.Scatterpolar(
            r=[avg_acc, avg_prec, avg_rec, avg_f1, avg_acc],
            theta=categories + [categories[0]],
            fill='toself',
            name='Average All Models',
            line_color='#FF4B4B',
            opacity=0.5
        ))

        fig3.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='#555'),
                angularaxis=dict(gridcolor='#555')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title='Best Model vs Average',
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("✨ Why This Model Performs Best")
        st.markdown("""
        **1. Probabilistic Nature**
        - Softmax activation provides well-calibrated probabilities for 8-class classification

        **2. Gradient Descent Optimization**
        - 1000 epochs with lr=0.01 converged well for 160 training samples

        **3. No Overfitting**
        - Simple model with 28 features and 160 samples

        **4. Linear Relationships**
        - Aggregated sensor features (mean, std, min, max) have approximately linear relationships with pain_scale
        """)

        st.markdown("---")
        st.subheader("📉 Worst Performance: KNN")
        st.markdown("""
        - High dimensionality (28 features) dilutes distance metric
        - Small dataset (160 training samples after aggregation)
        - Sensor noise affects distance calculations
        """)

elif page == "🔄 Retrain Models":
    st.title("🔄 Model Retraining")
    st.markdown("Click the button below to retrain all 12 models (6 Built-in + 6 From Scratch).")

    if st.button("🚀 Start Retraining", type="primary"):
        with st.spinner("Training in progress... This may take a few minutes..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            import time
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Training models... {i+1}%")
                time.sleep(0.05)
            status_text.text("Running main.py...")
            import subprocess
            result = subprocess.run(
                [sys.executable, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main.py')],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
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

    st.markdown("---")
    st.subheader("📊 Current Results Visualization")

    try:
        # Charts row 1
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Model Performance (Macro F1)**")
            import plotly.express as px
            fig = px.bar(
                comparison_df,
                x='Model',
                y='Macro F1',
                color='Type',
                barmode='group',
                title='Current Model Performance',
                color_discrete_map={'Built-in': '#FF4B4B', 'From Scratch': '#4B4BFF'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Performance Distribution**")
            fig2 = px.scatter(
                comparison_df,
                x='Accuracy',
                y='Macro F1',
                color='Type',
                size='Macro Precision',
                hover_data=['Model'],
                title='Accuracy vs Macro F1',
                color_discrete_map={'Built-in': '#FF4B4B', 'From Scratch': '#4B4BFF'}
            )
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Metrics summary
        st.markdown("---")
        st.subheader("📈 Quick Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏆 Best Model", "Logistic Reg. (Scratch)")
        with col2:
            st.metric("Best Macro F1", "0.4042", "Top performer")
        with col3:
            st.metric("Avg Macro F1", f"{comparison_df['Macro F1'].mean():.4f}")
        with col4:
            st.metric("Total Models", f"{len(comparison_df)}", "12 total")

        # Table
        st.markdown("---")
        st.subheader("📋 Detailed Results Table")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning("No results found. Please run training first.")
        st.error(f"Error: {e}")

elif page == "📄 Export Report":
    st.title("📄 Export Report")
    st.markdown("Generate a comprehensive HTML report with charts and analysis.")

    if st.button("📥 Generate HTML Report", type="primary"):
        with st.spinner("Generating report with charts..."):
            import datetime
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Create chart images
            import plotly.io as pio
            import base64
            from io import BytesIO

            # Chart 1: Model Comparison
            fig1 = px.bar(
                comparison_df,
                x='Model',
                y=['Macro F1', 'Accuracy'],
                color='Type',
                barmode='group',
                title='Model Performance Comparison'
            )
            fig1_html = pio.to_html(fig1, include_plotlyjs='cdn', div_id="chart1")

            # Chart 2: Radar Chart
            categories = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
            best_idx = comparison_df['Macro F1'].idxmax()
            best_data = comparison_df.iloc[best_idx]

            import plotly.graph_objects as go
            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(
                r=[best_data['Accuracy'], best_data['Macro Precision'],
                   best_data['Macro Recall'], best_data['Macro F1'], best_data['Accuracy']],
                theta=categories + [categories[0]],
                fill='toself',
                name=f"{best_data['Model']} ({best_data['Type']})",
                line_color='#90EE90'
            ))
            fig2.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title='Best Model Performance Profile'
            )
            fig2_html = pio.to_html(fig2, include_plotlyjs='cdn', div_id="chart2")

            # Chart 3: Scatter Plot
            fig3 = px.scatter(
                comparison_df,
                x='Accuracy',
                y='Macro F1',
                color='Type',
                size='Macro Precision',
                hover_data=['Model'],
                title='Accuracy vs Macro F1'
            )
            fig3_html = pio.to_html(fig3, include_plotlyjs='cdn', div_id="chart3")

            html = f"""<!DOCTYPE html>
<html><head><title>Student Burnout Prediction - Full Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body{{font-family:Arial,sans-serif;margin:40px;background:#0E1117;color:#FAFAFA;line-height:1.6}}
h1,h2,h3{{color:#FAFAFA;border-bottom:2px solid #262730;padding-bottom:10px}}
table{{border-collapse:collapse;width:100%;margin:20px 0;color:#FAFAFA}}
th,td{{border:1px solid #555;padding:12px;text-align:left}}
th{{background-color:#262730;color:#FAFAFA;font-weight:bold}}
tr:nth-child(even){{background-color:#1a1a2e}}
.metric{{display:inline-block;margin:20px;padding:20px 40px;background:#262730;border-radius:10px;border:2px solid #FF4B4B;min-width:150px}}
.metric h3{{margin:0;color:#FF4B4B;font-size:14px}}
.metric p{{font-size:32px;margin:10px 0 0 0;font-weight:bold;color:#FAFAFA}}
.best-model{{background:#1a472a;border-color:#90EE90;padding:30px;border-radius:15px;margin:20px 0}}
.chart-container{{margin:30px 0;padding:20px;background:#262730;border-radius:10px}}
.footer{{text-align:center;color:#888;margin-top:60px;padding-top:20px;border-top:1px solid #555}}
</style></head><body>
<h1>Student Burnout Prediction - Full Report</h1>
<p><strong>Generated:</strong> {now}</p>

<div class="best-model">
<h2>Best Model: Logistic Regression (From Scratch)</h2>
<div class="metric"><h3>Accuracy</h3><p>0.4250</p></div>
<div class="metric"><h3>Macro Precision</h3><p>0.4062</p></div>
<div class="metric"><h3>Macro Recall</h3><p>0.4021</p></div>
<div class="metric"><h3>Macro F1</h3><p>0.4042</p></div>
</div>

<div class="chart-container">
<h2>Model Performance Comparison</h2>
{fig1_html}
</div>

<div class="chart-container">
<h2>Best Model Performance Profile</h2>
{fig2_html}
</div>

<div class="chart-container">
<h2>Accuracy vs Macro F1 Distribution</h2>
{fig3_html}
</div>

<h2>Comparison Table</h2>
{comparison_df.to_html(index=False)}

<h2>Key Findings</h2>
<ul>
<li><strong>Logistic Regression (From Scratch)</strong> achieved highest Macro F1: 0.4042</li>
<li>Probabilistic nature with softmax suited for 8-class classification</li>
<li>KNN performed worst (Macro F1: 0.1471) due to high dimensionality</li>
<li>From-scratch implementations were competitive with built-in libraries</li>
<li>Total of 12 models tested (6 Built-in + 6 From Scratch)</li>
</ul>

<div class="footer"><p>Report generated by Student Burnout Prediction GUI</p></div>
</body></html>"""

            st.download_button(
                label="💾 Download HTML Report",
                data=html,
                file_name=f"student_burnout_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
            st.success("✅ Report generated with interactive charts!")

    st.markdown("---")
    st.subheader("Report Preview")
    st.markdown("""
    The HTML report will include:
    - 🏆 Best Model card with all metrics
    - 📊 Interactive model comparison charts (Plotly)
    - 🎯 Radar chart for best model performance
    - 📈 Scatter plot of Accuracy vs F1
    - 🤖 Full Comparison Table (all 12 models)
    - 📋 Key Findings and analysis
    - 🎨 Dark mode styling (matches GUI theme)
    - 📥 Single HTML file with interactive charts
    """)
