"""FraudShield: India-focused Streamlit interface for fraud-risk screening."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
st.set_page_config(page_title="FraudShield | Transaction Check", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at top left, #14263b 0%, #0b1220 48%, #080d17 100%); color: #e8eef8; }
[data-testid="stHeader"] { background: rgba(8,13,23,.88); }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
.block-container { max-width: 1180px; padding-top: 1.5rem; padding-bottom: 2.5rem; }
.hero { padding: 1.55rem 1.8rem; border-radius: 20px; background: linear-gradient(115deg,#132b46,#164e63 65%,#17615e); border: 1px solid #28566b; margin-bottom: 1.25rem; box-shadow: 0 14px 36px rgba(0,0,0,.22); }
.hero h1 { margin: 0; color: #f8fbff; font-size: 2.2rem; letter-spacing: -.03em; }
.hero p { margin: .45rem 0 0; color: #c6d8e9; font-size: 1rem; }
h2,h3,h4,p,label,.stMarkdown { color: #e8eef8; }
[data-testid="stForm"] { background: rgba(17,27,43,.88); border: 1px solid #293a52; border-radius: 16px; padding: 1.25rem 1.35rem; }
[data-testid="stMetric"], [data-testid="stAlert"] { background: #111c2d; border: 1px solid #2a3b52; border-radius: 13px; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background: #0c1626 !important; border-color: #34465f !important; }
input,textarea { color: #edf4ff !important; }
[data-testid="stNumberInput"] button { background: #17263a; color: #e8eef8; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button { background: linear-gradient(90deg,#0d9488,#0f766e); color: white; border: 1px solid #2dd4bf; border-radius: 10px; font-weight: 700; min-height: 2.9rem; }
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { background: #115e59; color: white; border-color: #5eead4; }
hr { border-color: #293a52; }
.hint { color: #9fb1c8; font-size: .88rem; }
[data-testid="stProgress"] > div > div { background: #2dd4bf; }
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>🛡️ FraudShield</h1><p>Credit card transaction screening ·</p></div>""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.warning("The trained model has not been created yet. Train it first from the project folder:")
    st.code("python train_model.py", language="powershell")
    st.info("After training creates models/fraud_model.joblib, refresh this page.")
    st.stop()

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

artifact = load_model(str(MODEL_PATH))
pipeline = artifact["pipeline"]
features = artifact.get("features", [])
threshold = float(artifact.get("threshold", 0.5))

with st.form("transaction_form", clear_on_submit=False):
    st.subheader("Transaction details")
    st.caption("Enter the transaction and account signals available to you.")
    st.markdown("#### Purchase information")
    c1,c2,c3 = st.columns(3)
    with c1:
        transaction_amount = st.number_input("Transaction amount (₹)", min_value=0.0, value=1250.0, step=100.0, help="This India-focused demo displays transaction amounts in rupees.")
        merchant_category = st.selectbox("Merchant category", ["Online Retail","Restaurant & Dining","Grocery & Supermarket","Gas Station","Travel & Airlines","Electronics & Technology","Clothing & Apparel","Healthcare & Pharmacy","Hotel & Lodging","Entertainment & Streaming","Home Improvement","Auto & Transportation","Utilities & Telecom","Education","Financial Services"])
        merchant_country = st.selectbox("Merchant country", ["India","USA","UK","Canada","Australia","France","Germany","Italy","Spain","Netherlands","Brazil","China","Japan","Mexico","Nigeria","Romania","Russia","South Africa","South Korea","UAE"], index=0)
    with c2:
        transaction_hour = st.slider("Transaction hour (24-hour clock)",0,23,14)
        transaction_channel = st.selectbox("Transaction channel",["Online","In-store","Mobile App","ATM"])
        device_type = st.selectbox("Device type",["Mobile","Desktop","POS Terminal","ATM"])
    with c3:
        card_present = st.selectbox("Card physically present?",["No","Yes"])
        international_transaction = st.selectbox("International transaction?",["No","Yes"])
        used_chip = st.selectbox("Chip used?",["No","Yes"])
        used_pin = st.selectbox("PIN used?",["No","Yes"])
    st.divider()
    st.markdown("#### Customer and account signals")
    a1,a2,a3 = st.columns(3)
    with a1:
        customer_age = st.number_input("Customer age",min_value=18,max_value=120,value=35,step=1)
        account_age_days = st.number_input("Account age (days)",min_value=0,max_value=50000,value=900,step=30)
        credit_score = st.number_input("Credit score",min_value=300,max_value=900,value=680,step=5)
    with a2:
        credit_limit = st.number_input("Credit limit (₹)",min_value=0.0,value=100000.0,step=5000.0)
        transactions_last_1h = st.number_input("Transactions in last hour",min_value=0,max_value=1000,value=1,step=1)
        transactions_last_24h = st.number_input("Transactions in last 24 hours",min_value=0,max_value=1000,value=3,step=1)
    with a3:
        amount_last_24h = st.number_input("Total transaction amount in last 24 hours (₹)",min_value=0.0,value=5500.0,step=500.0)
        avg_transaction_amount_30d = st.number_input("Average transaction amount (30 days, ₹)",min_value=0.0,value=1800.0,step=100.0)
        num_declined_last_30d = st.number_input("Declined transactions in last 30 days",min_value=0,max_value=1000,value=0,step=1)
    st.markdown("#### Location signal")
    loc1,loc2 = st.columns([2,3])
    with loc1:
        distance_not_available = st.checkbox("Distance from home is N/A",value=False,help="Use this when the distance is not known or is not applicable to the transaction.")
        distance_from_home_km = st.number_input("Distance from home (km)",min_value=0.0,max_value=50000.0,value=8.0,step=1.0,disabled=distance_not_available,help="Distance is measured in kilometres. Select N/A above if it is unavailable.")
    with loc2:
        if distance_not_available:
            st.info("Distance will be treated as missing. The trained preprocessing pipeline will impute its training-set median.")
       
            
    submitted = st.form_submit_button("Check transaction risk",use_container_width=True)

if submitted:
    row = {
        "transaction_amount":float(transaction_amount), "merchant_category":merchant_category, "merchant_country":merchant_country,
        "transaction_hour":int(transaction_hour), "card_present":int(card_present=="Yes"), "customer_age":int(customer_age),
        "account_age_days":int(account_age_days), "credit_limit":float(credit_limit), "credit_score":int(credit_score),
        "transactions_last_1h":int(transactions_last_1h), "transactions_last_24h":int(transactions_last_24h),
        "amount_last_24h":float(amount_last_24h), "distance_from_home_km":np.nan if distance_not_available else float(distance_from_home_km),
        "international_transaction":int(international_transaction=="Yes"), "transaction_channel":transaction_channel,
        "device_type":device_type, "used_chip":int(used_chip=="Yes"), "used_pin":int(used_pin=="Yes"),
        "avg_transaction_amount_30d":float(avg_transaction_amount_30d), "num_declined_last_30d":int(num_declined_last_30d),
    }
    input_df = pd.DataFrame([row],columns=features if features else None)
    try:
        probability = float(pipeline.predict_proba(input_df)[0][1])
        predicted_fraud = probability >= threshold
        st.divider()
        st.subheader("Screening result")
        r1,r2 = st.columns([1,2])
        with r1:
            if predicted_fraud:
                st.error("Potentially suspicious")
                st.write("The model flagged this transaction for further review.")
            else:
                st.success("Lower predicted risk")
                st.write("The model did not flag this transaction at the current threshold.")
        with r2:
            st.metric("Estimated fraud score",f"{probability:.1%}")
            st.progress(min(max(probability,0.0),1.0))
            st.caption(f"Decision threshold: {threshold:.0%}. This is a model estimate, not a guarantee or a calibrated real-world probability.")
        st.session_state.setdefault("history",[]).insert(0,{
            "Transaction amount":f"₹{transaction_amount:,.2f}","Merchant category":merchant_category,"Merchant country":merchant_country,
            "International":"Yes" if international_transaction=="Yes" else "No","Estimated fraud score":f"{probability:.1%}",
            "Screening result":"Potentially suspicious" if predicted_fraud else "Lower predicted risk",
        })
        st.session_state["history"] = st.session_state["history"][:10]
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

if st.session_state.get("history"):
    st.divider()
    st.subheader("Recent checks")
    st.dataframe(pd.DataFrame(st.session_state["history"]),use_container_width=True,hide_index=True)
    if st.button("Clear recent checks"):
        st.session_state["history"] = []
        st.rerun()

st.markdown('<p class="hint">Educational prototype only. Do not use it to approve or decline real financial transactions. The source dataset may not use verified INR-normalized values; validate currency assumptions before interpreting scores. Production use requires validation, monitoring, privacy controls, and human review.</p>',unsafe_allow_html=True)
