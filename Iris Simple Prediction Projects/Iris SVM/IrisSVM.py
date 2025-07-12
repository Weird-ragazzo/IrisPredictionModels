
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("IrisSVM.pkl")
scaler = joblib.load("IrisSVM_scaler.pkl")

target_names = ['Setosa 🌸', 'Versicolor 🌺', 'Virginica 🌷']

st.set_page_config(page_title="Iris SVM Predictor", page_icon="🌷")

st.title("🌷 **Iris Flower Species Predictor (SVM)** 🌷")
st.markdown("""
Provide flower measurements to predict its **species** 🌸🌺🌷.
""")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("🌿 Sepal Length (cm)", min_value=0.0, format="%.2f")
    petal_length = st.number_input("🌿 Petal Length (cm)", min_value=0.0, format="%.2f")

with col2:
    sepal_width = st.number_input("🌿 Sepal Width (cm)", min_value=0.0, format="%.2f")
    petal_width = st.number_input("🌿 Petal Width (cm)", min_value=0.0, format="%.2f")

if st.button("🔮 Predict Species (SVM)"):
    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=['sepal length (cm)', 'sepal width (cm)',
                                     'petal length (cm)', 'petal width (cm)'])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    st.success(f"🌷 **Predicted Species:** {target_names[pred]}")
    st.balloons()
