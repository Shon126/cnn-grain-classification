# ================= APP.PY =================
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

st.set_page_config(page_title="Grain Identifier", layout="wide")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("grains_mobilenetv2_cleaned.h5")
    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)  # list of class names
    with open("grains_info.json", "r") as f:
        grains_info = json.load(f)
    return model, class_labels, grains_info

model, class_labels, grains_info = load_model_and_labels()
IMG_SIZE = 128  # model input size

# ---------------- App UI ----------------
st.title("🌾 Grain Identifier")
st.write("Upload a grain image or take a photo to identify it and learn its nutritional info and uses.")

# File uploader or camera input
option = st.radio("Choose input method:", ["Upload Image", "Take Photo"])

img = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload your grain image", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
elif option == "Take Photo":
    img = st.camera_input("Take a photo of the grain")

# ---------------- Prediction ----------------
if img is not None:
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image (same as training)
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    if img_array.shape != (IMG_SIZE, IMG_SIZE, 3):
        st.error(f"⚠ Image shape mismatch: {img_array.shape}. Please upload an RGB image.")
    else:
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

        # Predict top 1
        pred_prob = model.predict(img_array)
        top_idx = np.argmax(pred_prob[0])
        label = class_labels[top_idx]
        confidence = pred_prob[0][top_idx]*100

        st.markdown("### 🔹 Prediction:")
        st.markdown(f"{label}** — {confidence:.2f}% confidence")

        # Display nutritional info & uses
        if label in grains_info:
            info = grains_info[label]
            st.markdown(f"- *Protein:* {info.get('protein','N/A')} g per 100g")
            st.markdown(f"- *Carbs:* {info.get('carbs','N/A')} g per 100g")
            st.markdown(f"- *Uses:* {info.get('uses','N/A')}")

        # ---------------- Optional Debug ----------------
        st.markdown("### 🔹 Raw Prediction Probabilities (Debug)")
        for i, p in enumerate(pred_prob[0]):
            st.write(f"{class_labels[i]}: {p*100:.2f}%")