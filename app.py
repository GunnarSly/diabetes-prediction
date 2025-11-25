import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
import json
from PIL import Image

# ------------------------------------------------------------
# Load Translations
# ------------------------------------------------------------
with open("assets/translations.json", "r", encoding="utf-8") as f:
    translations = json.load(f)

# ------------------------------------------------------------
# Translation Helper
# ------------------------------------------------------------
def T(key):
    return translations[st.session_state.lang].get(key, key)

# ------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"

if "page" not in st.session_state:
    st.session_state.page = "home"

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ------------------------------------------------------------
# Theme Toggle
# ------------------------------------------------------------
def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def apply_theme():
    if st.session_state.theme == "dark":
        st.markdown("""
            <style>
            .stApp { background-color: #0e1117 !important; color: #ffffff !important; }
            h1, h2, h3, h4, h5, h6, label, span, p, div { color: #ffffff !important; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp { background-color: #ffffff !important; color: #000000 !important; }
            h1, h2, h3, h4, h5, h6, label, span, p, div { color: #000000 !important; }
            </style>
        """, unsafe_allow_html=True)

apply_theme()

# ------------------------------------------------------------
# Load Models (joblib + h5 ONLY)
# ------------------------------------------------------------
def load_models():
    models = {}
    if not os.path.isdir("models"):
        return models

    for file in os.listdir("models"):
        path = os.path.join("models", file)

        if file.endswith(".joblib"):
            try:
                models[file] = joblib.load(path)
            except:
                pass

        elif file.endswith(".h5"):
            try:
                models[file] = load_model(path)
            except:
                pass

    return models

models = load_models()

# ------------------------------------------------------------
# Page Navigation
# ------------------------------------------------------------
def go_to(page):
    st.session_state.page = page

# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------
def home_page():

    # Logo
    try:
        st.image("images/university_logo.jpg", width=200)
    except:
        pass

    # University Name
    st.markdown(f"### {T('university')}")

    st.title(T("title"))
    st.write(T("description"))
    st.divider()

    # SIDEBAR
    with st.sidebar:

        st.button(T("theme"), on_click=toggle_theme)

        lang_choice = st.radio(T("language"), ["English", "العربية"])
        new_lang = "ar" if lang_choice == "العربية" else "en"

        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st.experimental_rerun()

        st.subheader(T("select_model"))
        if len(models) == 0:
            st.error("No models found in models/")
            model_name = None
        else:
            model_name = st.selectbox("", list(models.keys()))

    st.subheader(T("input_data"))

    # Input fields with logical ranges
    col1, col2 = st.columns(2)

    with col1:
        preg = st.number_input(T("pregnancies"), 0, 20)
        glucose = st.number_input(T("glucose"), 0, 220)
        bp = st.number_input(T("bp"), 0, 140)
        skin = st.number_input(T("skin"), 0, 100)

    with col2:
        insulin = st.number_input(T("insulin"), 0, 900)
        bmi = st.number_input(T("bmi"), 0.0, 70.0)
        dpf = st.number_input(T("dpf"), 0.0, 3.0, format="%.4f")
        age = st.number_input(T("age"), 18, 90)

    st.divider()

    # Prediction Button
    if st.button(T("predict"), use_container_width=True):
        st.session_state.pred_input = np.array(
            [[preg, glucose, bp, skin, insulin, bmi, dpf, age]], dtype=float
        )
        st.session_state.selected_model = model_name
        go_to("result")

# ------------------------------------------------------------
# RESULT PAGE
# ------------------------------------------------------------
def result_page():

    try:
        st.image("images/university_logo.jpg", width=200)
    except:
        pass

    st.markdown(f"### {T('university')}")
    st.title(T("result_title"))
    st.divider()

    model = models.get(st.session_state.selected_model, None)
    data = st.session_state.pred_input

    if model is None:
        st.error("Model not found.")
        return

    try:
        pred = model.predict(data)
    except:
        pred = model.predict(data)[0][0]

    pred = float(pred)
    result_label = T("diabetic") if pred > 0.5 else T("not_diabetic")

    st.markdown(
        f"<h1 style='text-align:center; font-size:60px;'>{result_label}</h1>",
        unsafe_allow_html=True
    )

    st.divider()
    st.button(T("back"), on_click=lambda: go_to("home"), use_container_width=True)

# ------------------------------------------------------------
# RENDER
# ------------------------------------------------------------
if st.session_state.page == "home":
    home_page()
else:
    result_page()
