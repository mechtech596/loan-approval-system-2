import pickle
import streamlit as st
import numpy as np

@st.cache_resource
def load_all():
    with open("pt.pkl", "rb") as f:
        pt = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("loan_approval_model.pkl", "rb") as f:
        model = pickle.load(f)
    return pt, scaler, model

pt, scaler, model = load_all()

st.title("Smart Loan Approval Predictor")
st.markdown(
    "Welcome! This smart tool helps you find out if your loan request is likely to be approved — "
    "based on your financial profile. Let’s get started!"
)

# Initialize session state for form visibility
if "show_form" not in st.session_state:
    st.session_state.show_form = False

# Start button sets show_form True
if st.button("Start"):
    st.session_state.show_form = True

# Show the form if start clicked
if st.session_state.show_form:
    st.header("Applicant Information")

    no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.radio("Self Employed?", ["Yes", "No"])
    income_annum = st.number_input("Annual Income (₹)", min_value=0)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
    loan_term = st.number_input("Loan Term (months)", min_value=0)
    cibil_score = st.slider("CIBIL Score", 300, 900, 700)
    res_asset_val = st.number_input("Residential Assets Value (₹)", min_value=0)
    com_asset_val = st.number_input("Commercial Assets Value (₹)", min_value=0)
    lux_asset_val = st.number_input("Luxury Assets Value (₹)", min_value=0)
    bank_asset_val = st.number_input("Bank Asset Value (₹)", min_value=0)

    if st.button("Predict Loan Approval"):
        education_num = 1 if education == "Graduate" else 0
        self_employed_num = 1 if self_employed == "Yes" else 0

        input_features = [
            no_of_dependents,
            education_num,
            self_employed_num,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            res_asset_val,
            com_asset_val,
            lux_asset_val,
            bank_asset_val
        ]

        input_array = np.array(input_features).reshape(1, -1)

        # Apply PowerTransformer then StandardScaler before prediction
        input_pt = pt.transform(input_array)
        input_scaled = scaler.transform(input_pt)

        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.success("Congratulations! Your loan is likely to be approved.")
        else:
            st.error("Sorry, your loan may not be approved based on the current details.")



