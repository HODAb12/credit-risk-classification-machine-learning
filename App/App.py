import streamlit as st
import pickle
import numpy as np

# ─── Configuration page ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyse de Risque de Crédit",
    page_icon="🏦",
    layout="wide"
)

# ─── CSS personnalisé ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 600;
        color: #185FA5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #888780;
        margin-bottom: 2rem;
    }
    .approved-box {
        background-color: #EAF3DE;
        border-left: 5px solid #639922;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .refused-box {
        background-color: #FCEBEB;
        border-left: 5px solid #E24B4A;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .approved-title {
        color: #3B6D11;
        font-size: 1.4rem;
        font-weight: 600;
    }
    .refused-title {
        color: #A32D2D;
        font-size: 1.4rem;
        font-weight: 600;
    }
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #185FA5;
        border-bottom: 2px solid #B5D4F4;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #F1EFE8;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Charger le modèle ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model_RL.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Fichier model_RL.pkl introuvable — placez-le dans le même dossier que app.py")

# ─── Titre ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏦 Analyse de Risque de Crédit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Modèle de Régression Logistique — Accuracy : 81.66%</div>', unsafe_allow_html=True)

# ─── Layout : 2 colonnes ──────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-header">👤 Informations personnelles</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Genre", ["Male", "Female"])
    with c2:
        married = st.selectbox("Marié(e)", ["Oui", "Non"])

    c3, c4 = st.columns(2)
    with c3:
        education = st.selectbox("Niveau d'études", ["Graduate", "Not Graduate"])
    with c4:
        self_employed = st.selectbox("Travailleur indépendant", ["Non", "Oui"])

    dependents = st.selectbox("Personnes à charge", ["0", "1", "2", "3+"])

    st.markdown('<div class="section-header">💰 Informations financières</div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        applicant_income = st.number_input("Revenu demandeur (MAD)", min_value=0, value=5000, step=500)
    with c6:
        coapplicant_income = st.number_input("Revenu co-emprunteur", min_value=0, value=0, step=500)

    c7, c8 = st.columns(2)
    with c7:
        loan_amount = st.number_input("Montant du prêt (milliers)", min_value=1, value=128, step=10)
    with c8:
        loan_term = st.selectbox("Durée du prêt (mois)", [360, 180, 120, 60, 480])

    st.markdown('<div class="section-header">🏠 Autres informations</div>', unsafe_allow_html=True)

    c9, c10 = st.columns(2)
    with c9:
        credit_history = st.selectbox("Historique de crédit", ["Conforme (1)", "Non conforme (0)"])
    with c10:
        property_area = st.selectbox("Zone du bien", ["Urban", "Semiurban", "Rural"])

    st.markdown("<br>", unsafe_allow_html=True)
    analyser = st.button("🔍 Analyser le client", use_container_width=True)

# ─── Prédiction ───────────────────────────────────────────────────────────────
with col2:
    st.markdown('<div class="section-header">📊 Résultat de l\'analyse</div>', unsafe_allow_html=True)

    if analyser and model_loaded:

        # Encodage
        gender_enc      = 1 if gender == "Male" else 0
        married_enc     = 1 if married == "Oui" else 0
        education_enc   = 1 if education == "Graduate" else 0
        self_emp_enc    = 1 if self_employed == "Oui" else 0
        credit_hist_enc = 1 if "1" in credit_history else 0

        dep_1 = 1 if dependents == "1" else 0
        dep_2 = 1 if dependents == "2" else 0
        dep_3 = 1 if dependents == "3+" else 0

        semi_urban = 1 if property_area == "Semiurban" else 0
        urban      = 1 if property_area == "Urban" else 0

        income_log = np.log1p(applicant_income)
        loan_log   = np.log1p(loan_amount)

        # Features
        input_data = np.array([[
            gender_enc, married_enc, education_enc, self_emp_enc,
            coapplicant_income, loan_term, credit_hist_enc,
            dep_1, dep_2, dep_3,
            semi_urban, urban,
            income_log, loan_log
        ]])

        # Prédiction
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        # ── Résultat ──
        if prediction == 1:
            prob_pct = probability[1] * 100
            st.markdown(f"""
            <div class="approved-box">
                <div class="approved-title">✅ Prêt APPROUVÉ</div>
                <p style="color:#3B6D11; margin-top:0.5rem; font-size:1rem;">
                    Probabilité d'approbation : <strong>{prob_pct:.1f}%</strong>
                </p>
                <p style="color:#3B6D11; font-size:0.9rem;">
                    Le profil de ce client présente un risque faible. 
                    Le crédit peut être accordé.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob_pct / 100)
        else:
            prob_pct = probability[0] * 100
            st.markdown(f"""
            <div class="refused-box">
                <div class="refused-title">❌ Prêt REFUSÉ</div>
                <p style="color:#A32D2D; margin-top:0.5rem; font-size:1rem;">
                    Probabilité de refus : <strong>{prob_pct:.1f}%</strong>
                </p>
                <p style="color:#A32D2D; font-size:0.9rem;">
                    Le profil de ce client présente un risque élevé. 
                    Le crédit ne peut pas être accordé.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob_pct / 100)

        # ── Récapitulatif ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📋 Récapitulatif du dossier</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Revenu demandeur", f"{applicant_income:,} MAD")
            st.metric("Historique crédit", "Conforme ✓" if credit_hist_enc == 1 else "Non conforme ✗")
        with r2:
            st.metric("Montant du prêt", f"{loan_amount}k MAD")
            st.metric("Zone du bien", property_area)
        with r3:
            st.metric("Durée", f"{loan_term} mois")
            st.metric("Personnes à charge", dependents)

        # ── Variables log ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔢 Variables transformées (log)</div>', unsafe_allow_html=True)

        l1, l2 = st.columns(2)
        with l1:
            st.metric("log(ApplicantIncome)", f"{income_log:.4f}")
        with l2:
            st.metric("log(LoanAmount)", f"{loan_log:.4f}")

    elif analyser and not model_loaded:
        st.error("Modèle non chargé — vérifiez le fichier model_RL.pkl")
    else:
        st.info("👈 Remplissez le formulaire et cliquez sur **Analyser le client**")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888780; font-size:0.85rem;'>"
    "Mini Projet ML — Régression Logistique | Accuracy : 81.66%"
    "</p>",
    unsafe_allow_html=True
)