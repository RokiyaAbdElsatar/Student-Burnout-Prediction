from gui_utils import get_project_root
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Model Comparison: Built-in vs From Scratch")

# Load data
try:
    comparison_df = pd.read_csv(os.path.join(get_project_root(), 'results', 'comparison.csv'))
except:
    comparison_df = pd.read_csv('results/comparison.csv')

# Comparison Table
st.subheader("📋 Comparison Table")
def highlight_best(row):
    if row['Model'] == 'Logistic Regression' and row['Type'] == 'From Scratch':
        return ['background-color: #1a472a; color: #90EE90'] * len(row)
    return [''] * len(row)

styled_df = comparison_df.style.apply(highlight_best, axis=1)
st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Interactive Grouped Bar Chart
st.subheader("📊 Performance Comparison (Interactive)")
fig = px.bar(
    comparison_df,
    x='Model',
    y=['Macro F1', 'Accuracy', 'Macro Precision', 'Macro Recall'],
    color='Type',
    barmode='group',
    title='All Metrics by Model & Type',
    color_discrete_map={'Built-in': '#FF4B4B', 'From Scratch': '#4B4BFF'}
)
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)
st.plotly_chart(fig, use_container_width=True)

# Parallel Coordinates Plot
st.subheader("🎯 Parallel Coordinates Plot")
fig2 = px.parallel_coordinates(
    comparison_df,
    dimensions=['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1'],
    color='Macro F1',
    color_continuous_scale='Viridis',
    title='Model Performance Across All Metrics'
)
fig2.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)
st.plotly_chart(fig2, use_container_width=True)

# Scatter Plot
col1, col2 = st.columns(2)
with col1:
    st.subheader("🎯 Accuracy vs Macro F1")
    fig3 = px.scatter(
        comparison_df,
        x='Accuracy',
        y='Macro F1',
        color='Type',
        size='Macro Precision',
        hover_data=['Model'],
        title='Model Performance Distribution',
        color_discrete_map={'Built-in': '#FF4B4B', 'From Scratch': '#4B4BFF'}
    )
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("🥧 Built-in vs Scratch")
    type_avg = comparison_df.groupby('Type')['Macro F1'].mean().reset_index()
    fig4 = px.pie(
        type_avg,
        values='Macro F1',
        names='Type',
        title='Average Macro F1 by Type',
        color='Type',
        color_discrete_map={'Built-in': '#FF4B4B', 'From Scratch': '#4B4BFF'}
    )
    fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig4, use_container_width=True)

# Summary Stats
st.markdown("---")
st.subheader("📈 Summary Statistics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🏆 Best Macro F1", f"{comparison_df['Macro F1'].max():.4f}", "🏆")
with col2:
    st.metric("⬇️ Worst Macro F1", f"{comparison_df['Macro F1'].min():.4f}")
with col3:
    best_model = comparison_df.loc[comparison_df['Macro F1'].idxmax(), 'Model']
    st.metric("🏆 Best Model", best_model)
with col4:
    st.metric("📊 Total Models", len(comparison_df))
