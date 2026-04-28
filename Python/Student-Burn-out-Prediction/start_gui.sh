#!/bin/bash
# Start the Streamlit GUI for Student Burnout Prediction

cd "$(dirname "$0")"
source venv/bin/activate
streamlit run gui/app.py
