import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# --- Load model and labels ---
MODEL_PATH = "grains_model.keras"
LABELS_PATH = "class_labels.json"
INFO_PATH = "grains_info.json"

model = load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)

with open(INFO_PATH, 'r') as f:
    grains_info = json.load(f)

# --- Streamlit App ---
st.set_page_config(page_title="Grain Classifier 🌾", page_icon="🌾")
st.title("Grain Classifier 🌾")
st.write("Upload an image of a grain, and I'll tell you what it is!")

# File uploader
uploaded_file = st.file_uploader("Choose a grain image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    pred_prob = model.predict(img_array)[0]
    pred_index = np.argmax(pred_prob)
    pred_label = class_labels[str(pred_index)]
    confidence = pred_prob[pred_index] * 100
    
    # Display
    st.image(img, use_container_width=True)
    st.markdown(f"*Prediction:* {pred_label} ({confidence:.2f}% confident)")
    st.markdown(f"*Info:* {grains_info[pred_label]}")
else:
    st.write("Upload an image to get started!")

st.markdown("---")
st.write("Made with 💖 by Shon and Babyberry’s assistant 😉")