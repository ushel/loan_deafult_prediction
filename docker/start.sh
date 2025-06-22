#!/bin/bash
mlflow ui --host 0.0.0.0 --port 5000 &
streamlit run app/streamlit_app.py