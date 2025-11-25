# Diabetes Prediction App

This Streamlit application predicts diabetes using 8 trained machine learning models 
in formats: `.joblib` and `.h5` ONLY.

## Features:
- 2 input modes (manual typing + selectable ranges)
- Light/Dark theme switch
- English / Arabic language switch
- University logo at the top
- Separate result page with large clear output
- Supports only joblib and h5 models (no more .pkl)

## Project Structure

diabetes-prediction-app/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│     ├── model1.joblib
│     ├── model2.h5
│     ├── model3.joblib
│     └── ... (total = 8)
│
├── images/
│     └── university_logo.jpg
│
└── assets/
      └── translations.json

## Run locally:
