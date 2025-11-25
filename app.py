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
# Session State Initialization (DEFAULT: English + Dark mode)
# ------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"

if "page" not in st.session_state:
    st.session_state.page = "home"

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ------------------------------------------------------------
# Translation Helper
# ------------------------------------------------------------
def T(key):
    return translations[st.session_state.lang].get(key, key)

# ------------------------------------------------------------
# Update Language / Theme
# ------------------------------------------------------------
def update_language():
    choice = st.session_state.lang_selector
    st.session_state.lang = "ar" if choice == "العربية" else "en"

def update_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

# ------------------------------------------------------------
# Apply Theme + Animations
# ------------------------------------------------------------
def apply_theme():
    if st.session_state.theme == "dark":
        st.markdown("""
            <style>
            .stApp { background-color:#0e1117; color:#ffffff; }

            /* TEXT */
            h1,h2,h3,h4,h5,h6,label,span,p,div {
                color:#ffffff !important;
            }

            /* Fade-in animation */
            .fade-in {
                animation: fadeIn 1.2s ease-in-out;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            /* Bounce animation */
            .bounce {
                animation: bounceIn 0.9s ease;
            }

            @keyframes bounceIn {
                0% { transform: scale(0.3); opacity: 0; }
                50% { transform: scale(1.05); opacity: 1; }
                70% { transform: scale(0.9); }
                100% { transform: scale(1); }
            }

            /* Fixed footer */
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: #111;
                padding: 10px;
                text-align: center;
                color: #ccc;
                font-size: 14px;
                border-top: 1px solid #444;
            }
            </style>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
            <style>
            .stApp { background-color:#ffffff; color:#000000; }

            h1,h2,h3,h4,h5,h6,label,span,p,div {
                color:#000000 !important;
            }

            .fade-in {
                animation: fadeIn 1.2s ease-in-out;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .bounce {
                animation: bounceIn 0.9s ease;
            }

            @keyframes bounceIn {
                0% { transform: scale(0.3); opacity: 0; }
                50% { transform: scale(1.05); opacity: 1; }
                70% { transform: scale(0.9); }
                100% { transform: scale(1); }
            }

            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: #f0f0f0;
                padding: 10px;
                text-align: center;
                color: #333;
                font-size: 14px;
                border-top: 1px solid #ccc;
            }
            </style>
        """, unsafe_allow_html=True)

apply_theme()

# ------------------------------------------------------------
# Load Models
# ------------------------------------------------------------
def load_models():
    models = {}
    if os.path.isdir("models"):
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
# Navigation
# ------------------------------------------------------------
def go_to(page):
    st.session_state.page = page

# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------
def home_page():

    # Logo (BIGGER)
    try:
        st.image("images/university_logo.jpg", width=350)
    except:
        pass

    st.markdown(f"<h3 class='fade-in'>{T('university')}</h3>", unsafe_allow_html=True)

    st.markdown(f"<h1 class='fade-in'>{T('title')}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='fade-in'>{T('description')}</p>", unsafe_allow_html=True)

    st.divider()

    # Sidebar
    with st.sidebar:

        st.button(T("theme"), on_click=update_theme)

        st.radio(
            T("language"),
            ["English", "العربية"],
            key="lang_selector",
            index=(0 if st.session_state.lang == "en" else 1),
            on_change=update_language
        )

        st.subheader(T("select_model"))
        if len(models) == 0:
            st.error("No models found.")
            model_name = None
        else:
            model_name = st.selectbox("", list(models.keys()))

    st.subheader(T("input_data"))

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
        st.image("images/university_logo.jpg", width=350)
    except:
        pass

    st.markdown(f"<h3>{T('university')}</h3>", unsafe_allow_html=True)

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
        f"<h1 class='bounce' style='text-align:center; font-size:60px;'>{result_label}</h1>",
        unsafe_allow_html=True,
    )

    st.divider()
    if st.button(T("back"), use_container_width=True):
        go_to("home")

# ------------------------------------------------------------
# FIXED FOOTER (BOTTOM BAR)
# ------------------------------------------------------------
footer_html = """
<div class='footer'>
    جامعة العرب للعلوم الطبية والتكنولوجيا<br>
    Arab University for Medical Sciences and Technology<br>
    <a href='https://armu.edu.ly' target='_blank'>https://armu.edu.ly</a><br>
    info@armu.edu.ly | +218 93-0600072<br>
    © 2025 Suliman & Ayob - All Rights Reserved<br>
    <a href='https://www.facebook.com/share/1FubzJnpDh/' target='_blank'>Facebook Page</a>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

# ------------------------------------------------------------
# RENDER
# ------------------------------------------------------------
if st.session_state.page == "home":
    home_page()
else:
    result_page()
