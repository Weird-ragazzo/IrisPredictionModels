
import streamlit as st
import pandas as pd
import joblib

# Load saved Decision Tree model and scaler
model = joblib.load("IrisDT.pkl")
scaler = joblib.load("IrisDT_scaler.pkl")  # Use your existing scaler file

# Iris target names (0: Setosa, 1: Versicolor, 2: Virginica)
target_names = ['Setosa 🌸', 'Versicolor 🌺', 'Virginica 🌷']

# Page config
st.set_page_config(page_title="Iris Decision Tree Predictor", page_icon="🌳")

# App title and description
st.title("🌳 **Iris Flower Species Predictor (Decision Tree)** 🌳")
st.markdown("""
Welcome to the **Iris Flower Classifier App (Decision Tree Model)**!  
Provide the flower's measurements below to find out its **species** 🌸🌺🌷.
""")

st.markdown("---")

# Input fields in a two-column layout
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("🌿 Sepal Length (cm)", min_value=0.0, format="%.2f")
    petal_length = st.number_input("🌿 Petal Length (cm)", min_value=0.0, format="%.2f")

with col2:
    sepal_width = st.number_input("🌿 Sepal Width (cm)", min_value=0.0, format="%.2f")
    petal_width = st.number_input("🌿 Petal Width (cm)", min_value=0.0, format="%.2f")

st.markdown("---")

# Prediction button
if st.button("🌳 Predict Species (Decision Tree)"):
    # Prepare input data
    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=['sepal length (cm)', 'sepal width (cm)',
                                     'petal length (cm)', 'petal width (cm)'])
    
    # Scale input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    pred = model.predict(input_scaled)[0]
    species_name = target_names[pred]

    # Display result
    st.success(f"🌳 **Predicted Species:** {species_name}")

    st.balloons()

# Footer
st.markdown("""
---
🌳 *Built with ❤️ using Streamlit and Decision Tree Algorithm*
""")
