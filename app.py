# ---------------- IMPORTS ----------------
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="🌾 Grain Identifier", layout="centered")
st.title("🌾 Grain Identifier")
st.write("Upload an image or take a photo, and identify the grain!")

# ---------------- LOAD MODEL & DATA ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("grains_mobilenetv2_cleaned.h5")

@st.cache_data
def load_class_labels():
    with open("class_labels.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_grains_info():
    with open("grains_info.json", "r") as f:
        return json.load(f)

model = load_model()
class_labels = load_class_labels()
grains_info = load_grains_info()

# ---------------- INPUT METHOD ----------------
option = st.radio("Select input method:", ["Upload Image", "Take Photo"])
img = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose a grain image", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
elif option == "Take Photo":
    img = st.camera_input("Take a photo of the grain")

# ---------------- PREDICTION ----------------
if img is not None:
    st.image(img, caption="Uploaded/Clicked Image", use_column_width=True)
    
    # Preprocess image
    IMG_SIZE = (96,96)
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    pred_prob = model.predict(img_array)
    pred_index = np.argmax(pred_prob)
    pred_label = class_labels[pred_index]
    confidence = float(pred_prob[0][pred_index]*100)
    
    # Display main result
    st.success(f"*Predicted Grain:* {pred_label} ({confidence:.2f}% confidence)")
    
    # Display full info
    info = grains_info.get(pred_label, {})
    st.markdown(f"*Protein:* {info.get('protein','-')}")
    st.markdown(f"*Carbs:* {info.get('carbs','-')}")
    st.markdown(f"*Uses:* {info.get('uses','-')}")
    
    # Top 3 predictions
    top_indices = np.argsort(pred_prob[0])[::-1][:3]
    st.markdown("### 🔹 Top 3 Predictions:")
    for i in top_indices:
        st.write(f"{class_labels[i]}: {pred_prob[0][i]*100:.2f}%")