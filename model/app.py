import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt   # ✅ added

# ==========================================
# LOAD MODEL (Safe Path Loading)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "jod.pkl")

with open(model_path, "rb") as f:
    binary_model, severity_model, scaler = pickle.load(f)

# ==========================================
# PAGE SETTINGS
# ==========================================
st.set_page_config(page_title="Social Media Addiction Predictor")

st.title("📱 Social Media Addiction Prediction System")
st.write("Enter one value at a time and press Enter.")

st.markdown("---")

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.answers = []

# ==========================================
# QUESTIONS (MUST MATCH TRAINING ORDER)
# ==========================================
questions = [
    "Daily Usage Hours",
    "Weekend Usage Hours",
    "Phone Checks Per Day",
    "Screen Time Before Bed (hrs)",
    "Sleep Hours",
    "Academic Performance (0-100)"
]

# ==========================================
# ✅ PROGRESS BAR (ADDED)
# ==========================================
st.progress(st.session_state.step / len(questions))

# ==========================================
# STEP-BY-STEP INPUT FLOW
# ==========================================
if st.session_state.step < len(questions):

    current_question = questions[st.session_state.step]

    value = st.text_input(current_question, key=f"input_{st.session_state.step}")

    if value:
        try:
            number = float(value)

            st.session_state.answers.append(number)
            st.session_state.step += 1
            st.rerun()

        except ValueError:
            st.error("Please enter a valid numeric value.")

# ==========================================
# PREDICTION SECTION (Same Logic As Notebook)
# ==========================================
else:

    st.success("All inputs received ✅")

    # Prepare input
    user_input = np.array(st.session_state.answers).reshape(1, -1)
    user_scaled = scaler.transform(user_input)

    # STEP 1: Addicted or Not
    addicted = binary_model.predict(user_scaled)[0]
    prob_addicted = binary_model.predict_proba(user_scaled)[0][1] * 100

    st.subheader("🔍 Prediction Result")

    # STEP 2: Output Logic (EXACT notebook logic)
    if addicted == 0:
        st.success("Status: Not Addicted ✅")
        st.info(f"Future Addiction Chance: {prob_addicted:.2f}%")
    else:
        severity = severity_model.predict(user_scaled)[0]

        if severity == 0:
            st.warning("Status: Addicted (Medium) ⚠️")
        else:
            st.error("Status: Addicted (High) 🚨")

    # ==========================================
    # ✅ PIE CHART (ADDED)
    # ==========================================
    st.subheader("📊 Addiction Breakdown")

    labels = ['Not Addicted', 'Addicted']
    sizes = [100 - prob_addicted, prob_addicted]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.set_title("Addiction Probability Split")

    st.pyplot(fig)

    # ==========================================
    # RESET BUTTON
    # ==========================================
    if st.button("Start Over"):
        st.session_state.step = 0
        st.session_state.answers = []
        st.rerun()