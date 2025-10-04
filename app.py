import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
from PIL import Image

# ---------------------
# CONFIG
# ---------------------
st.set_page_config(page_title="Grain Identifier", layout="wide")

MODEL_PATH = "grains_mobilenetv2_cleaned.h5"
CLASS_LABELS_PATH = "class_labels.json"
GRAINS_INFO_PATH = "grains_info.json"
IMG_SIZE = (224, 224)  # Model input size

# ---------------------
# LOAD MODEL & LABELS
# ---------------------
@st.cache_resource(show_spinner=True)
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_LABELS_PATH, "r") as f:
        class_labels = json.load(f)
    with open(GRAINS_INFO_PATH, "r") as f:
        grains_info = json.load(f)
    return model, class_labels, grains_info

model, class_labels, grains_info = load_model_and_labels()

# ---------------------
# PAGE UI
# ---------------------
st.title("🌾 Grain Identifier")
st.write("Upload an image of a grain, and this app will identify it and provide nutritional info.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image for model
    img_array = np.array(image)
    img_array = cv2.resize(img_array, IMG_SIZE)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    try:
        pred_prob = model.predict(img_array)[0]
        idx = np.argmax(pred_prob)
        label = class_labels[str(idx)]
        confidence = pred_prob[idx] * 100

        st.subheader("🔹 Prediction:")
        st.markdown(f"{label}** — {confidence:.2f}% confidence")

        # Show nutritional info
        info = grains_info.get(label, {})
        if info:
            st.markdown(f"*Protein:* {info.get('Protein', 'N/A')} g per 100g")
            st.markdown(f"*Carbs:* {info.get('Carbs', 'N/A')} g per 100g")
            st.markdown(f"*Uses:* {info.get('Uses', 'N/A')}")
        else:
            st.write("No additional info available.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")