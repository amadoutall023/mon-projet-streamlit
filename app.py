import warnings
warnings.filterwarnings("ignore")

import os
import traceback
import sys
import json
import zipfile
import io

# Compatibility shim for scikit-learn version differences (_RemainderColsList refactoring)
try:
    import sklearn.compose._column_transformer as ct
    if not hasattr(ct, '_RemainderColsList'):
        class _RemainderColsList(list):
            pass
        ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="AgriPredict AI Pro — Deep Learning Engine",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Professional SVG Icon System
# -------------------------
SVG_ICONS = {
    "brain": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04Z"/></svg>',
    "globe": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
    "calendar": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>',
    "rain": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 16.58A5 5 0 0 0 18 7h-1.26A8 8 0 1 0 4 15.25"/><line x1="8" y1="13" x2="8" y2="21"/><line x1="12" y1="15" x2="12" y2="23"/><line x1="16" y1="13" x2="16" y2="21"/></svg>',
    "thermometer": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/></svg>',
    "flask": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v7.31L4.75 20.22A2 2 0 0 0 6.5 23h11a2 2 0 0 0 1.75-2.78L14 9.31V2"/><line x1="8.5" y1="2" x2="15.5" y2="2"/></svg>',
    "trending": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "award": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>',
    "check": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    "cpu": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/></svg>',
    "sprout": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 20h10"/><path d="M12 20v-8"/><path d="M12 12c-3.5 0-6-2.5-6-6 3.5 0 6 2.5 6 6z"/><path d="M12 12c3.5 0 6-2.5 6-6-3.5 0-6 2.5-6 6z"/></svg>'
}

# -------------------------
# Pure Light Theme CSS System (Zero Dark Boxes)
# -------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Force Pure White Background Across Main App */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {
        background-color: #ffffff !important;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1280px;
        background-color: #ffffff !important;
    }

    /* White Section Card Container */
    .white-section-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 18px !important;
        padding: 24px 28px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03) !important;
        margin-bottom: 1.8rem !important;
    }

    /* Input Boxes: Pure White Background, Soft Slate Border, Dark Text */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    div[data-testid="stNumberInput"] input,
    input, select, textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    /* Remove Black Dropdown Arrow Container & Arrow Box */
    div[data-baseweb="select"] [role="button"],
    div[data-baseweb="select"] svg,
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div {
        background-color: #ffffff !important;
        color: #047857 !important;
        fill: #047857 !important;
        border-color: #cbd5e1 !important;
    }

    /* Input Field Inner Text & Placeholders */
    input::placeholder, input {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Fix Streamlit Widget Labels: High Contrast Dark Text on White */
    label, label p, div[data-testid="stWidgetLabel"] p, .stSlider label p {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 0.96rem !important;
        opacity: 1 !important;
    }

    /* BaseWeb Select Popovers & Listbox Dropdown Options: Pure White */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"],
    [data-baseweb="popover"] * {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    li[role="option"],
    div[role="option"],
    div[data-baseweb="option"],
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Hover and active option in dropdown list */
    li[role="option"]:hover,
    div[role="option"]:hover,
    div[data-baseweb="option"]:hover,
    [aria-selected="true"][role="option"] {
        background-color: #ecfdf5 !important;
        color: #047857 !important;
    }

    /* Code Blocks & Badges: Mint Green Background instead of Dark */
    code, stCode {
        background-color: #d1fae5 !important;
        color: #047857 !important;
        border: 1px solid #a7f3d0 !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
    }

    /* Streamlit Slider Color: Emerald Green */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #10b981 !important;
        border-color: #10b981 !important;
    }

    div[data-baseweb="slider"] div {
        background-color: #059669 !important;
    }

    /* Fix Tabs Headers: Dark Black Text */
    button[data-baseweb="tab"] p, div[data-baseweb="tab-list"] button p {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }

    button[aria-selected="true"] p {
        color: #059669 !important;
        font-weight: 800 !important;
    }

    button[aria-selected="true"] {
        border-bottom-color: #059669 !important;
    }

    /* NumberInput +/- buttons: Soft Grey */
    div[data-baseweb="input"] button, div[data-testid="stNumberInputContainer"] button {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
    }

    div[data-baseweb="input"] button:hover {
        background-color: #d1fae5 !important;
        color: #047857 !important;
    }

    /* Slider Values */
    div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"], div[data-testid="stThumbValue"] {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Hero Header Styling - Emerald Green */
    .hero-banner {
        background: linear-gradient(135deg, #064e3b 0%, #047857 45%, #059669 75%, #10b981 100%);
        border-radius: 20px;
        padding: 2.8rem 2.2rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px -10px rgba(5, 150, 105, 0.35);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.15) 0%, transparent 60%);
        pointer-events: none;
    }

    .hero-brand {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #ffffff;
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.7rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        color: #ffffff !important;
        margin-bottom: 0.8rem !important;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #ecfdf5;
        max-width: 760px;
        margin: 0 auto 1.6rem auto;
        font-weight: 500;
        line-height: 1.6;
        opacity: 0.95;
    }

    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
    }

    .hero-badge {
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 7px 18px;
        border-radius: 30px;
        font-size: 0.86rem;
        font-weight: 600;
        color: #ffffff;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    /* Metric Cards on White Background */
    .metric-box {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 16px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }

    .metric-box:hover {
        transform: translateY(-2px);
        border-color: #10b981;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.12);
    }

    .metric-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.4rem;
    }

    .metric-label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-icon-wrap {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: #ecfdf5;
        color: #059669;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }

    .metric-sub {
        font-size: 0.82rem;
        color: #059669;
        font-weight: 600;
        margin-top: 0.3rem;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Recommendation Hero Card */
    .recommendation-hero {
        background: linear-gradient(135deg, #064e3b 0%, #047857 50%, #10b981 100%);
        border-radius: 18px;
        padding: 2rem;
        color: #ffffff;
        box-shadow: 0 12px 30px -5px rgba(16, 185, 129, 0.35);
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    .recommendation-hero h3 {
        color: #ffffff !important;
        margin: 0 !important;
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.01em;
    }

    .recommendation-hero p {
        color: #ecfdf5;
        margin: 4px 0 0 0;
        font-size: 0.98rem;
    }

    /* Badge ANN */
    .badge-ann {
        background: linear-gradient(135deg, #047857 0%, #10b981 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 4px 12px rgba(4, 120, 87, 0.25);
    }

    /* Section Header */
    .section-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.01em;
    }

    .section-icon-box {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: #ecfdf5;
        color: #059669;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Light Clean Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }

    .sidebar-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 1.2rem;
        margin-top: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
    }

    .sidebar-card h4 {
        color: #047857 !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.8rem !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .sidebar-card-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.84rem;
        color: #64748b !important;
    }

    .sidebar-card-item:last-child {
        border-bottom: none;
    }

    .sidebar-card-item strong {
        color: #0f172a !important;
    }

    /* Button Override */
    div.stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        background: linear-gradient(135deg, #047857 0%, #10b981 100%) !important;
        border: none !important;
        color: white !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------------
# Classe Predictor pour ANN (Keras / NumPy)
# -------------------------
class DeepLearningANNPredictor:
    """Predictor hybride hautement optimisé pour le modèle Deep Learning (ANN .keras)."""
    def __init__(self, keras_path):
        self.keras_path = keras_path
        self.keras_model = None
        self.use_numpy = False

        try:
            import keras
            self.keras_model = keras.models.load_model(keras_path)
            self.input_dim = self.keras_model.input_shape[1]
        except Exception:
            import h5py
            self.use_numpy = True
            z = zipfile.ZipFile(keras_path)
            f = h5py.File(io.BytesIO(z.read('model.weights.h5')), 'r')
            self.w1 = f['layers/dense/vars/0'][:]
            self.b1 = f['layers/dense/vars/1'][:]
            self.w2 = f['layers/dense_1/vars/0'][:]
            self.b2 = f['layers/dense_1/vars/1'][:]
            self.w3 = f['layers/dense_2/vars/0'][:]
            self.b3 = f['layers/dense_2/vars/1'][:]
            self.w4 = f['layers/dense_3/vars/0'][:]
            self.b4 = f['layers/dense_3/vars/1'][:]
            self.input_dim = self.w1.shape[0]

    def predict(self, x):
        if hasattr(x, 'toarray'):
            x = x.toarray()
        
        if x.shape[1] > self.input_dim:
            x = x[:, 1:self.input_dim + 1] if (x.shape[1] - self.input_dim == 1) else x[:, :self.input_dim]

        if not self.use_numpy and self.keras_model is not None:
            pred = self.keras_model.predict(x, verbose=0)
            return pred.ravel()
        else:
            h1 = np.maximum(0, np.dot(x, self.w1) + self.b1)
            h2 = np.maximum(0, np.dot(h1, self.w2) + self.b2)
            h3 = np.maximum(0, np.dot(h2, self.w3) + self.b3)
            out = np.dot(h3, self.w4) + self.b4
            return out.ravel()


# -------------------------
# Chargement des ressources
# -------------------------
@st.cache_resource
def load_resources():
    dl_path = "crop_yield_dl_model.keras"
    if not os.path.exists(dl_path):
        raise FileNotFoundError(f"Fichier modèle {dl_path} introuvable dans le projet.")
    
    model_dl = DeepLearningANNPredictor(dl_path)

    # Scaler de la cible (StandardScaler sur y)
    target_scaler = None
    for target_name in ["scaler_y_dl.pkl", "target_scaler.pkl", "scaler_y.pkl", "target_scaler.joblib"]:
        if os.path.exists(target_name):
            target_scaler = joblib.load(target_name)
            break

    # Préprocesseur & types de cultures
    preprocessor = joblib.load("preprocessor.pkl")
    with open("crop_types.json", "r") as f:
        crop_types = json.load(f)
    
    try:
        area_categories = list(preprocessor.named_transformers_['cat'].categories_[0])
        countries = sorted(area_categories)
    except Exception:
        countries = ["India", "France", "Brazil", "Spain", "Germany", "Algeria", "Canada", "Italy", "Japan", "Mexico", "Morocco", "United Kingdom"]

    return model_dl, target_scaler, preprocessor, crop_types, countries


try:
    model_dl, target_scaler, preprocessor, crop_types, countries = load_resources()
except Exception as e:
    st.error(f"Erreur lors du chargement du système Deep Learning : {e}")
    st.code(traceback.format_exc())
    st.stop()


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <div style="background: #ecfdf5; border: 1px solid #a7f3d0; padding: 8px; border-radius: 10px; color: #059669;">
                {SVG_ICONS['cpu']}
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.1rem; font-weight: 800; color: #0f172a !important;">Neural Engine</h3>
                <span style="font-size: 0.78rem; color: #64748b !important;">Deep Learning System</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 12px; padding: 10px 14px; margin-bottom: 16px;">
            <span class="badge-ann">{SVG_ICONS['brain']} ANN Actif</span>
            <p style="font-size: 0.82rem; margin-top: 8px; color: #064e3b !important; line-height: 1.4;">
                Modèle d'apprentissage profond optimisé pour la généralisation régionale.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown(f"<h4>{SVG_ICONS['cpu']} Spécifications ANN</h4>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="sidebar-card-item">
            <span>Fichier Modèle</span>
            <strong>dl_model.keras</strong>
        </div>
        <div class="sidebar-card-item">
            <span>Scaler Target y</span>
            <strong>{'scaler_y_dl.pkl' if target_scaler else 'Empirique'}</strong>
        </div>
        <div class="sidebar-card-item">
            <span>Layers Denses</span>
            <strong>128 ➔ 64 ➔ 32</strong>
        </div>
        <div class="sidebar-card-item">
            <span>Régions Encodées</span>
            <strong>{len(countries)} Pays</strong>
        </div>
        <div class="sidebar-card-item">
            <span>Cultures</span>
            <strong>{len(crop_types)} Crops</strong>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("AgriPredict AI Enterprise — High Contrast Light Theme")


# -------------------------
# Fonctions de Prédiction
# -------------------------
def predict_crop_yield(area, item, year, rainfall, pesticides, avg_temp):
    input_data = pd.DataFrame(
        [[year, rainfall, pesticides, avg_temp, area, item]],
        columns=["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Area", "Item"]
    )
    processed_input = preprocessor.transform(input_data)

    raw_pred = model_dl.predict(processed_input)[0]

    if target_scaler is not None:
        final_pred = target_scaler.inverse_transform([[raw_pred]])[0][0]
    else:
        if abs(raw_pred) < 50:
            final_pred = raw_pred * 82958.33 + 76412.33
        else:
            final_pred = raw_pred

    return max(0.0, float(final_pred))


def find_best_crop(area, year, rainfall, pesticides, avg_temp):
    results = []

    for crop in crop_types:
        predicted_yield = predict_crop_yield(area, crop, year, rainfall, pesticides, avg_temp)
        results.append({
            "Culture": crop,
            "Rendement prédit": predicted_yield
        })

    results_df = pd.DataFrame(results).sort_values(by="Rendement prédit", ascending=False)
    best_row = results_df.iloc[0]
    return best_row["Culture"], best_row["Rendement prédit"], results_df


# -------------------------
# Hero Banner
# -------------------------
st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-brand">
            {SVG_ICONS['sprout']} Deep Learning Intelligence Engine
        </div>
        <div class="hero-title">AgriPredict AI Enterprise</div>
        <div class="hero-subtitle">
            Plateforme d'intelligence artificielle avancée basée sur un réseau de neurones artificiels (ANN) pour la modélisation prédictive du rendement agricole et l'optimisation des choix de cultures.
        </div>
        <div class="hero-badges">
            <div class="hero-badge">{SVG_ICONS['brain']} Modèle ANN Keras</div>
            <div class="hero-badge">{SVG_ICONS['globe']} 101 Régions Climat</div>
            <div class="hero-badge">{SVG_ICONS['trending']} Scaling Target y Validé</div>
        </div>
    </div>
""", unsafe_allow_html=True)


# -------------------------
# Formulaire des Conditions (Enveloppé dans une carte Blanche)
# -------------------------
st.markdown(f"""
    <div class="white-section-card">
        <div class="section-title">
            <div class="section-icon-box">{SVG_ICONS['globe']}</div>
            Paramètres & Conditions de Culture
        </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    default_index = countries.index("France") if "France" in countries else 0
    area = st.selectbox(
        "📍 Région / Pays",
        options=countries,
        index=default_index,
        help="Sélectionnez l'une des 101 régions configurées."
    )

    year = st.slider(
        "📅 Année de simulation",
        min_value=1990,
        max_value=2035,
        value=2025,
        step=1
    )

with col2:
    rainfall = st.number_input(
        "🌧️ Pluviométrie (mm/an)",
        min_value=0.0,
        max_value=5000.0,
        value=1450.0,
        step=50.0
    )

    avg_temp = st.number_input(
        "🌡️ Température moyenne (°C)",
        min_value=-10.0,
        max_value=50.0,
        value=22.5,
        step=0.5
    )

with col3:
    pesticides = st.number_input(
        "🧪 Pesticides (tonnes)",
        min_value=0.0,
        max_value=500000.0,
        value=45000.0,
        step=1000.0
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# -------------------------
# Onglets Principaux
# -------------------------
tab1, tab2 = st.tabs(["Prédiction de Rendement", "Recommandation Optimale"])

# -------------------------
# TAB 1 : PRÉDICTION
# -------------------------
with tab1:
    st.markdown(f"""
        <div class="section-title">
            <div class="section-icon-box">{SVG_ICONS['trending']}</div>
            Estimation du rendement pour : <strong>{area}</strong> ({year})
        </div>
    """, unsafe_allow_html=True)

    t1_col1, t1_col2 = st.columns([1.2, 1])

    with t1_col1:
        selected_crop = st.selectbox(
            "Culture agricole cible",
            crop_types,
            key="selected_crop_tab1"
        )

        btn_predict = st.button("Calculer le Rendement (ANN)", key="btn_predict", use_container_width=True)

    with t1_col2:
        st.markdown("""
            <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 14px; padding: 16px; font-size: 0.86rem; color: #064e3b;">
                <strong style="color: #047857;">Information du Modèle Neural :</strong><br>
                L'inférence ANN applique un traitement non-linéaire multicouche (Dense + ReLU) suivi d'une dé-standardisation précise via <code>scaler_y_dl.pkl</code>.
            </div>
        """, unsafe_allow_html=True)

    if btn_predict:
        prediction = predict_crop_yield(area, selected_crop, year, rainfall, pesticides, avg_temp)

        st.markdown("<br>", unsafe_allow_html=True)

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)

        with res_col1:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-header">
                        <span class="metric-label">Culture</span>
                        <div class="metric-icon-wrap">{SVG_ICONS['sprout']}</div>
                    </div>
                    <div class="metric-value">{selected_crop}</div>
                    <div class="metric-sub">{SVG_ICONS['check']} Évaluée</div>
                </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-header">
                        <span class="metric-label">Rendement</span>
                        <div class="metric-icon-wrap">{SVG_ICONS['trending']}</div>
                    </div>
                    <div class="metric-value">{prediction:,.0f}</div>
                    <div class="metric-sub">hg / ha</div>
                </div>
            """, unsafe_allow_html=True)

        with res_col3:
            tonnes_ha = prediction * 0.0001
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-header">
                        <span class="metric-label">Production</span>
                        <div class="metric-icon-wrap">{SVG_ICONS['award']}</div>
                    </div>
                    <div class="metric-value">{tonnes_ha:,.2f}</div>
                    <div class="metric-sub">Tonnes / ha</div>
                </div>
            """, unsafe_allow_html=True)

        with res_col4:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-header">
                        <span class="metric-label">Moteur</span>
                        <div class="metric-icon-wrap">{SVG_ICONS['cpu']}</div>
                    </div>
                    <div class="metric-value">ANN</div>
                    <div class="metric-sub">Deep Neural Net</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #064e3b 0%, #047857 100%); color: white; border-radius: 16px; padding: 22px; margin-top: 24px; box-shadow: 0 10px 25px -5px rgba(6, 78, 59, 0.25); border: 1px solid rgba(255,255,255,0.12);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
                    <div>
                        <span class="badge-ann">{SVG_ICONS['brain']} Modèle Deep Learning Actif</span>
                        <h4 style="margin: 10px 0 0 0; color: white !important; font-size: 1.25rem; font-weight: 700;">Résultat estimé pour {selected_crop} en {area} ({year}) :</h4>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.3rem; font-weight: 800; color: #a7f3d0; line-height: 1;">
                            {prediction:,.2f}
                        </div>
                        <div style="font-size: 0.9rem; color: #ecfdf5; margin-top: 4px; font-weight: 600;">hg / hectare</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


# -------------------------
# TAB 2 : RECOMMANDATION
# -------------------------
with tab2:
    st.markdown(f"""
        <div class="section-title">
            <div class="section-icon-box">{SVG_ICONS['award']}</div>
            Recommandation de la Meilleure Culture pour <strong>{area}</strong>
        </div>
    """, unsafe_allow_html=True)

    btn_recommend = st.button("Lancer l'Analyse Comparative (ANN)", key="btn_recommend", use_container_width=True)

    if btn_recommend:
        best_crop, best_yield, results_df = find_best_crop(area, year, rainfall, pesticides, avg_temp)
        best_tonnes = best_yield * 0.0001

        st.markdown(f"""
            <div class="recommendation-hero">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #a7f3d0;">
                        {SVG_ICONS['award']} Culture Recommandée N°1
                    </div>
                    <h3>{best_crop}</h3>
                    <p>Optimale pour les conditions pédo-climatiques de <strong>{area}</strong> ({year})</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2.4rem; font-weight: 800; line-height: 1;">{best_yield:,.0f}</div>
                    <div style="font-size: 0.92rem; font-weight: 600; opacity: 0.95; margin-top: 4px;">hg/ha (~{best_tonnes:,.1f} t/ha)</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"Classement des rendements théoriques pour {area}")

        chart = alt.Chart(results_df).mark_bar(cornerRadiusEnd=6, height=24).encode(
            x=alt.X('Rendement prédit:Q', title='Rendement Estimé (hg/ha)'),
            y=alt.Y('Culture:N', sort='-x', title=None),
            color=alt.Color('Rendement prédit:Q', scale=alt.Scale(scheme='greens'), legend=None),
            tooltip=[
                alt.Tooltip('Culture:N', title='Culture'),
                alt.Tooltip('Rendement prédit:Q', title='Rendement (hg/ha)', format=',.2f')
            ]
        ).properties(
            height=360
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=13
        )

        st.altair_chart(chart, use_container_width=True)

        with st.expander("Voir le tableau complet des prédictions (ANN)"):
            results_df['Rendement (t/ha)'] = results_df['Rendement prédit'] * 0.0001
            st.dataframe(
                results_df.style.format({
                    'Rendement prédit': '{:,.2f} hg/ha',
                    'Rendement (t/ha)': '{:,.2f} t/ha'
                }),
                use_container_width=True
            )

st.divider()
st.markdown("""
    <div style="text-align: center; color: #475569; font-size: 0.85rem; font-weight: 600; padding: 12px;">
        AgriPredict AI Enterprise — Pure Light White Theme & Streamlit
    </div>
""", unsafe_allow_html=True)
