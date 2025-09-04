import streamlit as st
import pickle
import numpy as np

st.title("🩺 Diabetes Disease Prediction")

# Load model and show basic debug info
try:
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Make sure diabetes_model.pkl is in the app folder.")
    st.stop()

st.write("Model type:", type(model))
# If pipeline, show steps
if hasattr(model, "named_steps"):
    st.write("Pipeline steps:", list(model.named_steps.keys()))
else:
    st.write("Warning: loaded object is not a pipeline. Ensure you saved scaler + model together.")

# Input sliders (order MUST match training order)
pregnancies = st.slider("Pregnancies", 0, 17, 1)
glucose = st.slider("Glucose", 0, 200, 100)
bp = st.slider("Blood Pressure", 0, 122, 70)
skin = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 846, 79)
bmi = st.slider("BMI", 0.0, 70.0, 25.0, step=0.01)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)
age = st.slider("Age", 21, 100, 33)

input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], dtype=float)

if st.button("Predict"):
    # Show raw input to verify order/values
    st.write("Input array:", input_data.tolist())

    # If model is not a pipeline, you MUST scale input similarly to training
    try:
        pred = model.predict(input_data)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Show probability if available
    prob_text = ""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        prob_text = f" (probabilities: class 0 = {probs[0]:.2f}, class 1 = {probs[1]:.2f})"
    elif hasattr(model, "decision_function"):
        score = model.decision_function(input_data)[0]
        prob_text = f" (decision score: {score:.3f})"

    if pred == 1:
        st.error(f"⚠️ This person has Diabetes{prob_text}")
    else:
        st.success(f"✅ This person does not have Diabetes{prob_text}")
