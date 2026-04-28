import streamlit as st
import subprocess
import time

st.title("Model Retraining")
st.markdown("Click the button below to retrain all 12 models (6 Built-in + 6 From Scratch).")

if st.button("Start Retraining", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(50):
        progress_bar.progress(i + 1)
        status_text.text(f"Initializing... {i+1}%")
        time.sleep(0.02)

    status_text.text("Running main.py... This may take a few minutes.")
    progress_bar.progress(50)

    result = subprocess.run(
        ["../venv/bin/python", "main.py"],
        capture_output=True,
        text=True,
        cwd='..'
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

st.markdown("---")
st.subheader("Current Results")
try:
    comparison_df = pd.read_csv('../results/comparison.csv')
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
