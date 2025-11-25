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
# Language Handler
# ------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"

def T(key):
    return translations[st.session_state.lang].get(key, key)

# ------------------------------------------------------------
# Page Navigation
# ------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page

# ------------------------------------------------------------
# Theme (Light / Dark)
# ------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def apply_theme():
    if st.session_state.theme == "dark":
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #0e1117;
                color: #f5f5f5 !important;
            }
            h1, h2, h3, h4, h5, h6, label, span, div, p {
                color: #f5f5f5 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #ffffff;
                color: #000000 !important;
            }
            h1, h2, h3, h4, h5, h6, label, span, div, p {
                color: #000000 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

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
# HOME PAGE
# ------------------------------------------------------------
def home_page():

    # Logo (JPG)
    try:
        st.image("images/university_logo.jpg", use_column_width=False, width=200)
    except:
        st.write("")

    st.markdown(f"### {T('university')}", unsafe_allow_html=True)
    st.title(T("title"))
    st.write(T("description"))
    st.divider()

    # Sidebar
    with st.sidebar:
        st.button(T("theme"), on_click=toggle_theme)
        lang_choice = st.radio(T("language"), ["English", "العربية"])
        st.session_state.lang = "ar" if lang_choice == "العربية" else "en"

        st.subheader(T("select_model"))
        if len(models) == 0:
            st.error("No models found in 'models/'")
            model_name = None
        else:
            model_name = st.selectbox("", list(models.keys()))

    st.subheader(T("input_data"))

    # Ranges
    col1, col2 = st.columns(2)

    with col1:
        preg_manual = st.number_input(T("pregnancies"), 0, 20)
        glucose_manual = st.number_input(T("glucose"), 0, 220)
        bp_manual = st.number_input(T("bp"), 0, 140)
        skin_manual = st.number_input(T("skin"), 0, 100)

    with col2:
        insulin_manual = st.number_input(T("insulin"), 0, 900)
        bmi_manual = st.number_input(T("bmi"), 0.0, 70.0)
        dpf_manual = st.number_input(T("dpf"), 0.0, 3.0, format="%.4f")
        age_manual = st.number_input(T("age"), 18, 90)

    st.divider()

    manual_mode = st.radio(T("input_method"), [T("manual"), T("range")])

    if manual_mode == T("manual"):
        values = [
            preg_manual, glucose_manual, bp_manual, skin_manual,
            insulin_manual, bmi_manual, dpf_manual, age_manual
        ]
    else:
        st.write("Use the ranges above")  
        values = [
            preg_manual, glucose_manual, bp_manual, skin_manual,
            insulin_manual, bmi_manual, dpf_manual, age_manual
        ]

    if st.button(T("predict"), use_container_width=True):
        st.session_state.pred_input = np.array([values], dtype=float)
        st.session_state.selected_model = model_name
        go_to("result")


# ------------------------------------------------------------
# RESULT PAGE
# ------------------------------------------------------------
def result_page():

    try:
        st.image("images/university_logo.jpg", use_column_width=False, width=200)
    except:
        st.write("")

    st.markdown(f"### {T('university')}", unsafe_allow_html=True)
    st.title(T("result_title"))
    st.divider()

    model = models.get(st.session_state.selected_model, None)
    data = st.session_state.pred_input

    try:
        pred = model.predict(data)
    except:
        pred = model.predict(data)[0][0]

    pred_val = float(pred)
    final = T("diabetic") if pred_val > 0.5 else T("not_diabetic")

    st.markdown(
        f"<h1 style='text-align:center; font-size:60px;'>{final}</h1>",
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
