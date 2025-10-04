import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Paths to your model and label files
MODEL_PATH = "grains_mobilenetv2_cleaned.h5"
CLASS_LABELS_PATH = "class_labels.json"
GRAINS_INFO_PATH = "grains_info.json"  # protein, carbs, uses

st.set_page_config(page_title="Grain Identifier", layout="wide")

@st.cache_resource
def load_model_and_labels():
    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load class labels
    with open(CLASS_LABELS_PATH, "r") as f:
        class_labels = json.load(f)
    
    # Load grain info
    with open(GRAINS_INFO_PATH, "r") as f:
        grains_info = json.load(f)
    
    return model, class_labels, grains_info

model, class_labels, grains_info = load_model_and_labels()

st.title("🌾 Grain Identifier")
st.write("Upload an image of a grain and get its prediction!")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_container_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0  # normalize if trained that way
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    pred_prob = model.predict(img_array)[0]
    idx = np.argmax(pred_prob)
    label = class_labels.get(str(idx)) or class_labels.get(idx)
    confidence = pred_prob[idx] * 100

    # Show top prediction
    st.markdown(f"### 🔹 Prediction: *{label}* — {confidence:.2f}% confidence")
    
    # Show nutrition info and uses
    info = grains_info.get(label, {})
    if info:
        st.write(f"*Protein:* {info.get('protein', 'N/A')} g per 100g")
        st.write(f"*Carbs:* {info.get('carbs', 'N/A')} g per 100g")
        st.write(f"*Uses:* {info.get('uses', 'N/A')}")