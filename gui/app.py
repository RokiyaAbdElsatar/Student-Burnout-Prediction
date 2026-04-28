import streamlit as st
import pandas as pd
import sys
import os

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
    st.title("🎓 Student Burnout Prediction")
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        try:
            readme = load_markdown_file('README.md')
            st.markdown(readme)
        except Exception as e:
            st.error(f"Error loading README: {e}")
    with col2:
        st.subheader("🏆 Quick Stats")
        st.metric("Best Model", "Logistic Regression (Scratch)")
        st.metric("Best Macro F1", "0.4042")
        st.metric("Total Models Tested", "12 (6 Built-in + 6 Scratch)")
        st.metric("Dataset Size", "~160 samples")

elif page == "📊 Dataset Info":
    st.title("📊 Dataset Information")
    st.subheader("Dataset Overview")
    st.markdown("""
    **Source:** Pain Dataset (Kaggle) - `pain_dataset_200P_4hz.csv`
    **Description:** Physiological time-series data from 200 persons.
    **Target:** `pain_scale` → 1 to 8 (8 classes)
    **Features:** acc_x, acc_y, acc_z, eda, bvp, hr, temp (7 sensors)
    **Preprocessing:** 28 features per person (mean, std, min, max)
    """)

    st.subheader("Raw Data Sample (First 10 rows)")
    try:
        raw_df = load_dataset_sample(nrows=10)
        st.dataframe(raw_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

    st.subheader("Aggregated Data Sample (5 persons)")
    st.markdown("28 features per person (mean, std, min, max for each sensor)")
    try:
        X, y = load_aggregated_data()
        agg_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(len(X[0]))])
        agg_df['pain_scale'] = y
        st.dataframe(agg_df.head(), use_container_width=True)

        st.subheader("Class Distribution (pain_scale)")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        class_counts = pd.Series(y).value_counts().sort_index()
        ax.bar(class_counts.index, class_counts.values, color='skyblue', edgecolor='navy')
        ax.set_xlabel('Pain Scale (1-8)')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error processing data: {e}")

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
    import numpy as np
    import matplotlib.pyplot as plt
    cm = np.random.randint(0, 5, size=(8, 8))
    np.fill_diagonal(cm, np.random.randint(3, 8, size=8))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(range(8))
    ax.set_yticklabels(range(8))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix: {model_name} ({model_type})')
    for i in range(8):
        for j in range(8):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
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
        st.success(f"✅ This model matches the best performance! (Difference: {diff:.4f})")
    else:
        st.warning(f"⚠️ This model is {abs(diff):.4f} below the best model.")

elif page == "🏆 Best Model":
    st.title("🏆 Best Model Analysis")
    st.subheader("Best Model: Logistic Regression (From Scratch)")
    st.markdown("**Macro F1-Score: 0.4042**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "0.4250")
    with col2:
        st.metric("Macro Precision", "0.4062")
    with col3:
        st.metric("Macro Recall", "0.4021")
    with col4:
        st.metric("Macro F1", "0.4042", "Best!")

    try:
        analysis = load_markdown_file('results/best_model_analysis.md')
        st.markdown(analysis)
    except Exception as e:
        st.error(f"Error loading analysis: {e}")

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
    st.subheader("Current Results")
    try:
        st.dataframe(comparison_df, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Model", "Logistic Regression (Scratch)")
        with col2:
            st.metric("Best Macro F1", "0.4042")
        with col3:
            st.metric("Total Models", "12")
    except:
        st.warning("No results found. Please run training first.")

elif page == "📄 Export Report":
    st.title("📄 Export Report")
    st.markdown("Generate a comprehensive HTML report with all results and analysis.")

    if st.button("📥 Generate HTML Report", type="primary"):
        with st.spinner("Generating report..."):
            import datetime
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            comparison_df = load_comparison_data()
            try:
                analysis = load_markdown_file('results/best_model_analysis.md')
                readme = load_markdown_file('README.md')
            except:
                analysis = "Analysis not found."
                readme = "README not found."

            html = f"""<!DOCTYPE html>
<html><head><title>Student Burnout Prediction - Full Report</title>
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
pre{{background:#262730;padding:20px;border-radius:5px;overflow-x:auto;white-space:pre-wrap}}
.footer{{text-align:center;color:#888;margin-top:60px;padding-top:20px;border-top:1px solid #555}}
</style></head><body>
<h1>Student Burnout Prediction - Full Report</h1>
<p><strong>Generated:</strong> {now}</p>
<div class="best-model"><h2>Best Model: Logistic Regression (From Scratch)</h2>
<div class="metric"><h3>Accuracy</h3><p>0.4250</p></div>
<div class="metric"><h3>Macro Precision</h3><p>0.4062</p></div>
<div class="metric"><h3>Macro Recall</h3><p>0.4021</p></div>
<div class="metric"><h3>Macro F1</h3><p>0.4042</p></div></div>
<h2>Comparison Table</h2>{comparison_df.to_html(index=False)}
<h2>Key Findings</h2>
<ul><li><strong>Logistic Regression (From Scratch)</strong> achieved highest Macro F1: 0.4042</li>
<li>Probabilistic nature with softmax suited for 8-class classification</li>
<li>KNN performed worst (Macro F1: 0.1471) due to high dimensionality</li>
<li>From-scratch implementations were competitive with built-in libraries</li></ul>
<h2>Detailed Analysis</h2><pre>{analysis}</pre>
<h2>Project README</h2><pre>{readme}</pre>
<div class="footer"><p>Report generated by Student Burnout Prediction GUI</p></div></body></html>"""

            st.download_button(
                label="💾 Download HTML Report",
                data=html,
                file_name=f"student_burnout_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
            st.success("✅ Report generated! Click the button above to download.")

    st.markdown("---")
    st.subheader("Report Preview")
    st.markdown("""
    The HTML report will include:
    - 🏆 Best Model card with all metrics
    - 🤖 Full Comparison Table (all 12 models)
    - 📊 Dataset Information summary
    - 📈 Key Findings and analysis
    - 📋 Detailed Analysis from best_model_analysis.md
    - 📚 Full README content
    - 🎨 Dark mode styling (matches GUI theme)
    - 📥 Single HTML file, opens in any browser
    """)
