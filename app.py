import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image

# ---------------------------
# Paths
MODEL_PATH = "grain_model.keras"
CLASS_LABELS_PATH = "class_labels.json"
GRAINS_INFO_PATH = "grains_info.json"  # create a JSON file with protein, carbs, uses

# ---------------------------
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_LABELS_PATH, "r") as f:
        class_labels = json.load(f)
    with open(GRAINS_INFO_PATH, "r") as f:
        grains_info = json.load(f)
    return model, class_labels, grains_info

model, class_labels, grains_info = load_model_and_labels()

# ---------------------------
st.title("🌾 Grain Identifier")
st.write("Upload a grain image or take a photo, and get its name, confidence, and nutritional info.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img = img.resize((224,224))  # adjust to your model input
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    pred_prob = model.predict(img_array)[0]
    top_idx = np.argmax(pred_prob)
    top_label = class_labels[str(top_idx)]
    confidence = pred_prob[top_idx]*100
    
    st.subheader("🔹 Prediction")
    st.write(f"{top_label}** — {confidence:.2f}% confidence")
    
    # Info
    info = grains_info.get(top_label, {})
    if info:
        st.write(f"*Protein:* {info.get('protein','N/A')} g per 100g")
        st.write(f"*Carbs:* {info.get('carbs','N/A')} g per 100g")
        st.write(f"*Uses:* {info.get('uses','N/A')}")