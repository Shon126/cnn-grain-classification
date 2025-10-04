import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Grain Identifier", layout="wide")

# ---------------- PATHS ----------------
MODEL_PATH = "grains_model.h5"           # Your trained model
LABELS_PATH = "class_labels.json"        # Class labels mapping
GRAINS_INFO_PATH = "grains_info.json"    # Grain info (protein, carbs, uses)

# ---------------- LOAD MODEL & LABELS ----------------
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        class_labels = json.load(f)
    with open(GRAINS_INFO_PATH, "r") as f:
        grains_info = json.load(f)
    return model, class_labels, grains_info

model, class_labels, grains_info = load_model_and_labels()

# ---------------- TITLE ----------------
st.title("🌾 Grain Identifier")

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload an image of the grain", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for model
    img_array = np.array(img.resize((128,128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------- PREDICTION ----------------
    pred_prob = model.predict(img_array)
    idx = np.argmax(pred_prob)
    label = class_labels[str(idx)]
    confidence = pred_prob[0][idx] * 100

    # Grain info
    info = grains_info.get(label, {"Protein":"N/A", "Carbs":"N/A", "Uses":"N/A"})

    # ---------------- DISPLAY ----------------
    st.subheader("🔹 Prediction")
    st.markdown(f"{label}** — {confidence:.2f}% confidence")
    st.markdown(f"*Protein:* {info['Protein']}")
    st.markdown(f"*Carbs:* {info['Carbs']}")
    st.markdown(f"*Uses:* {info['Uses']}")