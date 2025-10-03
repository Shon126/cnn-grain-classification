# ---------------- IMPORTS ----------------
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Grain Identifier", layout="wide")

MODEL_PATH = "grains_mobilenetv2_cleaned.keras"
LABELS_PATH = "class_labels.json"

# Grain info dictionary (update as per your data)
grains_info = {
    "maize": {"Protein": "9 g per 100g", "Carbs": "74 g per 100g", "Uses": "Cornmeal, porridge, popcorn"},
    "proso": {"Protein": "11 g per 100g", "Carbs": "70 g per 100g", "Uses": "Bird feed, porridge"},
    "red kidney beans": {"Protein": "24 g per 100g", "Carbs": "60 g per 100g", "Uses": "Curries, soups"},
    "chickpea": {"Protein": "19 g per 100g", "Carbs": "61 g per 100g", "Uses": "Hummus, curries"},
    "soybean": {"Protein": "36 g per 100g", "Carbs": "30 g per 100g", "Uses": "Tofu, soy milk, flour"},
    "lentils": {"Protein": "26 g per 100g", "Carbs": "60 g per 100g", "Uses": "Soups, curries"},
    "jowar": {"Protein": "11 g per 100g", "Carbs": "72 g per 100g", "Uses": "Flatbreads, porridges"},
    "sesame": {"Protein": "18 g per 100g", "Carbs": "23 g per 100g", "Uses": "Oil, seeds in cooking"},
    "rice": {"Protein": "7 g per 100g", "Carbs": "80 g per 100g", "Uses": "Staple food, flour"},
    "ragi": {"Protein": "7 g per 100g", "Carbs": "72 g per 100g", "Uses": "Porridge, flour"},
    "bajra": {"Protein": "11 g per 100g", "Carbs": "67 g per 100g", "Uses": "Flatbreads, porridge"},
    "foxtail millet": {"Protein": "12 g per 100g", "Carbs": "65 g per 100g", "Uses": "Porridge, flour"},
    "wheat": {"Protein": "13 g per 100g", "Carbs": "71 g per 100g", "Uses": "Bread, pasta, flour"}
}

# ---------------- LOAD MODEL & LABELS ----------------
@st.cache_resource(show_spinner=True)
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        class_labels = json.load(f)
    return model, class_labels

model, class_labels = load_model_and_labels()

# ---------------- TITLE ----------------
st.title("🌾 Grain Identifier")
st.write("Upload a grain image or take a photo to identify it.")

# ---------------- IMAGE INPUT ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # ---------------- PREPROCESS ----------------
    IMG_SIZE = (128, 128)
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # ---------------- PREDICT ----------------
    pred_prob = model.predict(img_array)[0]
    idx = int(np.argmax(pred_prob))
    label = class_labels[str(idx)] if str(idx) in class_labels else class_labels[idx]
    confidence = pred_prob[idx] * 100

    # ---------------- DISPLAY RESULTS ----------------
    st.subheader("🔹 Prediction")
    st.write(f"{label}** — {confidence:.2f}% confidence")
    info = grains_info.get(label, {})
    if info:
        st.write(f"*Protein:* {info['Protein']}")
        st.write(f"*Carbs:* {info['Carbs']}")
        st.write(f"*Uses:* {info['Uses']}")