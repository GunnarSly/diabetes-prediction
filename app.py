import streamlit as st
import pandas as np
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
# Load Models (joblib + h5 ONLY)
# ------------------------------------------------------------
def load_models():
    models = {}
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
# Theme switch
# ------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------
def home_page():

    # Logo (UPDATED → JPG)
    st.image("images/university_logo.jpg", use_column_width=False, width=200)

    st.title(T("title"))
    st.write(T("description"))
    st.divider()

    # Sidebar (Theme + Language)
    with st.sidebar:
        st.button(T("theme"), on_click=toggle_theme)
        lang_choice = st.radio(T("language"), ["English", "العربية"])
        st.session_state.lang = "ar" if lang_choice == "العربية" else "en"

        st.subheader(T("select_model"))
        model_name = st.selectbox("", list(models.keys()))

    st.subheader("Input Data")

    # Input fields (Manual + Range)
    col1, col2 = st.columns(2)

    with col1:
        preg_manual = st.number_input(T("pregnancies") + " (Manual)", min_value=0)
        preg_range = st.slider(T("pregnancies") + " (Range)", 0, 20, 3)

        glucose_manual = st.number_input(T("glucose") + " (Manual)", min_value=0)
        glucose_range = st.slider(T("glucose") + " (Range)", 0, 200, 100)

        bp_manual = st.number_input(T("bp") + " (Manual)", min_value=0)
        bp_range = st.slider(T("bp") + " (Range)", 0, 150, 70)

        skin_manual = st.number_input(T("skin") + " (Manual)", min_value=0)
        skin_range = st.slider(T("skin") + " (Range)", 0, 100, 20)

    with col2:
        insulin_manual = st.number_input(T("insulin") + " (Manual)", min_value=0)
        insulin_range = st.slider(T("insulin") + " (Range)", 0, 900, 80)

        bmi_manual = st.number_input(T("bmi") + " (Manual)", min_value=0.0)
        bmi_range = st.slider(T("bmi") + " (Range)", 0.0, 70.0, 25.0)

        dpf_manual = st.number_input(T("dpf") + " (Manual)", min_value=0.0, format="%.4f")
        dpf_range = st.slider(T("dpf") + " (Range)", 0.0, 3.0, 0.5)

        age_manual = st.number_input(T("age") + " (Manual)", min_value=0)
        age_range = st.slider(T("age") + " (Range)", 0, 120, 30)

    st.divider()

    manual_mode = st.radio("Input method:", ["Manual", "Range"])

    if manual_mode == "Manual":
        values = [
            preg_manual, glucose_manual, bp_manual, skin_manual,
            insulin_manual, bmi_manual, dpf_manual, age_manual
        ]
    else:
        values = [
            preg_range, glucose_range, bp_range, skin_range,
            insulin_range, bmi_range, dpf_range, age_range
        ]

    if st.button(T("predict"), use_container_width=True):
        st.session_state.pred_input = np.array([values])
        st.session_state.selected_model = model_name
        go_to("result")


# ------------------------------------------------------------
# RESULT PAGE
# ------------------------------------------------------------
def result_page():

    # Logo (UPDATED → JPG)
    st.image("images/university_logo.jpg", use_column_width=False, width=200)

    st.title(T("result_title"))
    st.divider()

    model = models[st.session_state.selected_model]
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
