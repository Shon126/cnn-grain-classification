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
        class_labels = json.load(f)
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
    # Preprocess image
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    # Predict
    pred_prob = model.predict(img_array)
    top_idx = pred_prob[0].argsort()[-3:][::-1]  # top 3 predictions

    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.markdown("### 🔹 Predictions:")
    for i, idx in enumerate(top_idx):
        label = class_labels[str(idx)]
        confidence = pred_prob[0][idx]*100
        st.markdown(f"{i+1}. {label}** — {confidence:.2f}% confidence")

        # Display info
        if label in grains_info:
            info = grains_info[label]
            st.markdown(f"- *Protein:* {info.get('protein','N/A')} g per 100g")
            st.markdown(f"- *Carbs:* {info.get('carbs','N/A')} g per 100g")
            st.markdown(f"- *Uses:* {info.get('uses','N/A')}")
        st.markdown("---")