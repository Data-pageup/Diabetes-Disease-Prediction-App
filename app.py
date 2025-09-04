import streamlit as st
import pickle
import numpy as np

# load saved model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🩺 Diabetes Disease Prediction")

# input sliders
pregnancies = st.slider("Pregnancies", 0, 17, 1)
glucose = st.slider("Glucose", 0, 200, 100)
bp = st.slider("Blood Pressure", 0, 122, 70)
skin = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 846, 79)
bmi = st.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 21, 100, 33)

# prediction button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ This person has Diabetes")
    else:
        st.success("✅ This person does not have Diabetes")
