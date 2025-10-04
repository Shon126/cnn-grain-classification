import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
import os
import gdown

st.set_page_config(page_title="Grain Identifier", layout="wide")

# ------------------ Paths ------------------ #
MODEL_FILE = "grains_mobilenetv2_cleaned.h5"
CLASS_LABELS_FILE = "class_labels.json"
GRAINS_INFO_FILE = "grains_info.json"

# Google Drive link for the model (replace YOUR_FILE_ID)
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"

# ------------------ Download model if not exists ------------------ #
if not os.path.exists(MODEL_FILE):
    st.info("Downloading model…")
    gdown.download(MODEL_DRIVE_URL, MODEL_FILE, quiet=False)
    st.success("Model downloaded!")

# ------------------ Load model and JSONs ------------------ #
@st.cache_data
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_FILE)
    with open(CLASS_LABELS_FILE, "r") as f:
        class_labels = json.load(f)
    with open(GRAINS_INFO_FILE, "r") as f:
        grains_info = json.load(f)
    return model, class_labels, grains_info

model, class_labels, grains_info = load_model_and_labels()

# ------------------ App Title ------------------ #
st.title("🌾 Grain Identifier")

# ------------------ Image Upload ------------------ #
uploaded_file = st.file_uploader("Upload an image of a grain", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    # Preprocess for model
    img_resized = cv2.resize(img_rgb, (224, 224))  # assuming MobileNetV2 input
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # ------------------ Prediction ------------------ #
    try:
        pred_prob = model.predict(img_array)[0]
        idx = np.argmax(pred_prob)
        label = class_labels[str(idx)]
        confidence = pred_prob[idx] * 100

        # Grain info
        info = grains_info.get(label, {})

        st.markdown(f"### 🔹 Prediction: {label} — {confidence:.2f}% confidence")
        st.markdown(f"*Protein:* {info.get('Protein', 'N/A')} per 100g")
        st.markdown(f"*Carbs:* {info.get('Carbs', 'N/A')} per 100g")
        st.markdown(f"*Uses:* {info.get('Uses', 'N/A')}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")