import streamlit as st
import requests

API_URL = "https://home-credit-scoring.onrender.com/predict"

st.set_page_config(page_title="Home Credit Scoring", page_icon="🏦", layout="centered")
st.title("🏦 Home Credit — Scoring de Demande de Prêt")
st.markdown("Simulez l'évaluation d'une demande de crédit par notre modèle XGBoost.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Informations financières")
    income = st.number_input("Revenu annuel (€)", 10000, 1000000, 150000, step=5000)
    credit = st.number_input("Montant du crédit (€)", 5000, 4000000, 500000, step=10000)
    annuity = st.number_input("Annuité mensuelle (€)", 100, 200000, 25000, step=500)
    goods_price = st.number_input("Prix du bien (€)", 0, 4000000, 450000, step=10000)

with col2:
    st.subheader("👤 Profil du demandeur")
    age = st.slider("Âge", 18, 70, 35)
    years_employed = st.slider("Années d'ancienneté", 0, 40, 5)
    gender = st.selectbox("Genre", ["Homme", "Femme"])
    nb_children = st.number_input("Nombre d'enfants", 0, 10, 0)
    nb_docs = st.slider("Documents fournis", 0, 20, 5)

st.subheader("📊 Scores externes (0 à 1)")
col3, col4, col5 = st.columns(3)
with col3:
    ext1 = st.slider("Score externe 1", 0.0, 1.0, 0.5)
with col4:
    ext2 = st.slider("Score externe 2", 0.0, 1.0, 0.5)
with col5:
    ext3 = st.slider("Score externe 3", 0.0, 1.0, 0.5)

st.divider()

if st.button("🔍 Analyser la demande", use_container_width=True):
    payload = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "DAYS_BIRTH": -age * 365,
        "DAYS_EMPLOYED": -years_employed * 365,
        "CODE_GENDER": 1 if gender == "Homme" else 0,
        "CNT_CHILDREN": nb_children,
        "NB_DOCUMENTS_FOURNIS": nb_docs,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3
    }

    with st.spinner("Analyse en cours..."):
        try:
            res = requests.post(API_URL, json=payload, timeout=30).json()
            proba = res["probability_of_default"]

            st.divider()
            col6, col7, col8 = st.columns(3)
            with col6:
                st.metric("Probabilité de défaut", f"{proba*100:.1f}%")
            with col7:
                st.metric("Niveau de risque", res["risk_level"])
            with col8:
                st.metric("Seuil utilisé", f"{res['threshold_used']:.2f}")

            st.progress(proba)

            if res["decision"] == "REFUS":
                st.error(f"❌ Décision : **{res['decision']}** — Risque de défaut trop élevé")
            else:
                st.success(f"✅ Décision : **{res['decision']}** — Profil acceptable")

        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")