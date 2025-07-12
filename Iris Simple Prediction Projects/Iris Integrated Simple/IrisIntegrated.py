# iris_multi_model_app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ----------------------------
# Function to load models
# ----------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression 🌸": joblib.load("IrisLogRegModel.pkl"),
        "KNN 🌺": joblib.load("IrisKNN.pkl"),
        "SVM 🌷": joblib.load("IrisSVM.pkl"),
        "Decision Tree 🌳": joblib.load("IrisDT.pkl")
    }
    scaler = joblib.load("IrisLogRegModel_scaler.pkl")
    return models, scaler

# ----------------------------
# Function to predict species
# ----------------------------
def predict_species(model, input_scaled):
    pred = model.predict(input_scaled)[0]
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(input_scaled)[0]
    else:
        probas = None
    return pred, probas

# ----------------------------
# Function to display result
# ----------------------------
def display_result(pred, probas, model_name):
    species_name = target_names[pred]
    st.success(f"✨ **Predicted Species:** {species_name}")
    st.info(f"🔍 **Model Used:** {model_name}")

    # Show confidence scores if available
    if probas is not None:
        st.markdown("#### 🔢 **Confidence Scores:**")
        for i, prob in enumerate(probas):
            st.write(f"{target_names[i]}: {prob:.2%}")

    # Show image
    st.image(species_images[species_name], caption=species_name, use_container_width=True)
    st.balloons()

# ----------------------------
# Load models and scaler
# ----------------------------
models, scaler = load_models()

# Target names with emojis
target_names = ['Setosa 🌸', 'Versicolor 🌺', 'Virginica 🌷']

# Species images dictionary
species_images = {
    'Setosa 🌸': 'plant.jpg',
    'Versicolor 🌺': 'plant.jpg',
    'Virginica 🌷': 'plant.jpg'
}

# Model accuracies (update with your real results)
model_accuracies = {
    "Logistic Regression 🌸": "97%",
    "KNN 🌺": "100%",
    "SVM 🌷": "100%",
    "Decision Tree 🌳": "100%"
}

# Page config
st.set_page_config(page_title="Iris Multi-Model Predictor", page_icon="🌸", layout="centered")

# App title
st.title("🌸 **Iris Flower Species Predictor (Modern Edition)** 🌸")
st.markdown("""
Welcome to the **Iris Flower Multi-Model Classifier App**!  
Select an algorithm, provide flower measurements, and view predictions with **confidence scores and visuals** 🌷🌺🌸.
""")

# Tabs for modern UI
tab1, tab2, tab3 = st.tabs(["🏷️ Select Model", "✏️ Input Features", "📊 Prediction Result"])

# ----------------------------
# Tab 1: Model Selection
# ----------------------------
with tab1:
    model_choice = st.selectbox(
        "🔍 Select ML Model:",
        tuple(models.keys())
    )
    st.write(f"✅ **Selected Model Accuracy:** {model_accuracies[model_choice]}")

# ----------------------------
# Tab 2: Input Features
# ----------------------------
with tab2:
    st.header("🌿 Enter Flower Measurements")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.35)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

    # Prepare input data
    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=['sepal length (cm)', 'sepal width (cm)',
                                     'petal length (cm)', 'petal width (cm)'])
    
    input_scaled = scaler.transform(input_df)

# ----------------------------
# Tab 3: Prediction and Visuals
# ----------------------------
with tab3:
    st.header("📊 Prediction Result")

    if st.button("✨ Predict Species"):
        model = models[model_choice]
        pred, probas = predict_species(model, input_scaled)
        display_result(pred, probas, model_choice)

        # Plotting input vs dataset
        iris = load_iris(as_frame=True)
        df = iris.frame
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)',
                        hue=iris.target_names[df['target']], palette='Set2', s=60)
        ax.scatter(petal_length, petal_width, color='black', s=150, label='Your Input', edgecolor='yellow')
        ax.legend()
        ax.set_title("Petal Length vs Width with Your Input")
        st.pyplot(fig)

# Footer
st.markdown("""
---
🌸 *Built with ❤️ using Streamlit and multiple ML algorithms by Dhruv Raghav*
""")
