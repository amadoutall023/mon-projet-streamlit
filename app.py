import warnings
warnings.filterwarnings("ignore")

import os
import traceback
import streamlit as st
import pandas as pd
import joblib
import json
import altair as alt

# -------------------------
# Page Configuration & Styling
# -------------------------
st.set_page_config(
    page_title="Crop Yield AI 🌾",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern visual aesthetics
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header gradient banner */
    .hero-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #2e7d32 100%);
        padding: 2.2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    }
    
    .hero-header h1 {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
        color: #ffffff !important;
    }
    
    .hero-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-bottom: 0;
    }

    /* Result Banner */
    .result-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: #052c16;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.25);
    }
    </style>
""", unsafe_allow_html=True)


# -------------------------
# Chargement des fichiers
# -------------------------
@st.cache_resource
def load_resources():
    model_path = "crop_yield_model.joblib" if os.path.exists("crop_yield_model.joblib") else "crop_yield_model.pkl"
    model = joblib.load(model_path)
    preprocessor = joblib.load("preprocessor.pkl")
    with open("crop_types.json", "r") as f:
        crop_types = json.load(f)
    
    try:
        area_categories = list(preprocessor.named_transformers_['cat'].categories_[0])
        countries = sorted(area_categories)
    except Exception:
        countries = ["India", "France", "Brazil", "Spain", "Germany", "Algeria", "Canada", "Italy", "Japan", "Mexico", "Morocco", "United Kingdom"]

    return model, preprocessor, crop_types, countries


try:
    model, preprocessor, crop_types, countries = load_resources()
except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers : {e}")
    st.code(traceback.format_exc())
    st.stop()


# -------------------------
# Fonctions
# -------------------------

def predict_crop_yield(
    area,
    item,
    year,
    rainfall,
    pesticides,
    avg_temp
):
    input_data = pd.DataFrame(
        [[
            year,
            rainfall,
            pesticides,
            avg_temp,
            area,
            item
        ]],
        columns=[
            "Year",
            "average_rain_fall_mm_per_year",
            "pesticides_tonnes",
            "avg_temp",
            "Area",
            "Item"
        ]
    )

    processed_input = preprocessor.transform(input_data)

    prediction = model.predict(processed_input)

    return prediction[0]


def find_best_crop(
    area,
    year,
    rainfall,
    pesticides,
    avg_temp
):
    best_crop = None
    best_yield = -1

    results = []

    for crop in crop_types:

        predicted_yield = predict_crop_yield(
            area=area,
            item=crop,
            year=year,
            rainfall=rainfall,
            pesticides=pesticides,
            avg_temp=avg_temp
        )

        results.append({
            "Culture": crop,
            "Rendement prédit": predicted_yield
        })

        if predicted_yield > best_yield:
            best_yield = predicted_yield
            best_crop = crop

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Rendement prédit",
        ascending=False
    )

    return best_crop, best_yield, results_df


# -------------------------
# Interface Header
# -------------------------

st.markdown("""
    <div class="hero-header">
        <h1>🌾 Crop Yield AI</h1>
        <p>Application de prédiction du rendement agricole basée sur des données climatiques et agricoles</p>
    </div>
""", unsafe_allow_html=True)


# -------------------------
# Entrées utilisateur
# -------------------------

st.subheader("⚙️ Conditions agricoles")

col1, col2 = st.columns(2)

with col1:

    default_index = countries.index("France") if "France" in countries else 0
    area = st.selectbox(
        "Pays / Région (101 pays pris en charge)",
        options=countries,
        index=default_index,
        help="Choisissez parmi les 101 pays entraînés dans le modèle."
    )

    year = st.number_input(
        "Année",
        min_value=1990,
        max_value=2100,
        value=2025
    )

    rainfall = st.number_input(
        "Pluviométrie moyenne annuelle (mm)",
        min_value=0.0,
        value=1500.0
    )


with col2:

    pesticides = st.number_input(
        "Pesticides utilisés (tonnes)",
        min_value=0.0,
        value=50000.0
    )

    avg_temp = st.number_input(
        "Température moyenne (°C)",
        value=25.0
    )


st.divider()


# -------------------------
# Navigation par Onglets
# -------------------------
tab1, tab2 = st.tabs(["🔮 1. Prédiction du rendement", "🏆 2. Recommandation de culture"])

with tab1:
    st.subheader(f"1. Prédiction du rendement ({area})")

    selected_crop = st.selectbox(
        "Choisissez une culture",
        crop_types,
        key="selected_crop_tab1"
    )

    if st.button("Prédire le rendement", key="btn_predict"):

        prediction = predict_crop_yield(
            area=area,
            item=selected_crop,
            year=year,
            rainfall=rainfall,
            pesticides=pesticides,
            avg_temp=avg_temp
        )

        st.success(
            f"Rendement estimé : {prediction:,.2f} hg/ha"
        )
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(label="Culture", value=selected_crop)
        m_col2.metric(label="Rendement Estimé", value=f"{prediction:,.2f} hg/ha")
        m_col3.metric(label="Région", value=area)


with tab2:
    st.subheader(f"2. Recommandation de culture ({area})")

    if st.button("Trouver la meilleure culture", key="btn_recommend", type="primary"):

        best_crop, best_yield, results_df = find_best_crop(
            area=area,
            year=year,
            rainfall=rainfall,
            pesticides=pesticides,
            avg_temp=avg_temp
        )

        st.markdown(f"""
            <div class="result-card">
                🌟 Culture recommandée pour <strong>{area}</strong> : <strong>{best_crop}</strong> (Rendement : <strong>{best_yield:,.2f} hg/ha</strong>)
            </div>
        """, unsafe_allow_html=True)

        st.metric(
            label="Rendement estimé",
            value=f"{best_yield:,.2f} hg/ha"
        )

        st.subheader(f"Classement des cultures pour {area}")

        chart = alt.Chart(results_df.head(10)).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X('Rendement prédit:Q', title='Rendement prédit (hg/ha)'),
            y=alt.Y('Culture:N', sort='-x', title='Culture'),
            color=alt.Color('Rendement prédit:Q', scale=alt.Scale(scheme='greens'), legend=None),
            tooltip=['Culture', alt.Tooltip('Rendement prédit:Q', format=',.2f')]
        ).properties(
            height=300
        )
        st.altair_chart(chart, use_container_width=True)

        st.dataframe(
            results_df.head(10),
            use_container_width=True
        )

st.divider()
st.caption("🌾 Crop Yield AI — Propulsé par Streamlit & Scikit-Learn")
