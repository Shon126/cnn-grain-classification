import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import json
import os

st.set_page_config(page_title="Grain Classifier", page_icon="🌾", layout="centered")

st.title("🌾 Grain Classifier App")
st.write("Upload an image of a grain and the model will classify it for you.")

# -------------------------------
# 1️⃣ Set paths
# -------------------------------
MODEL_PATH = "grains_mobilenetv2_cleaned.keras"  # your model file
LABELS_PATH = "class_labels.json"                # your labels file

# -------------------------------
# 2️⃣ Load model
# -------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------------
# 3️⃣ Load labels
# -------------------------------
if not os.path.exists(LABELS_PATH):
    st.error(f"❌ Labels file not found: {LABELS_PATH}")
    st.stop()

with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)

# Convert to list if dict:
if isinstance(class_labels, dict):
    # sometimes saved as {"0": "rice", "1": "wheat", ...}
    class_labels = list(class_labels.values())

st.write(f"Classes: {class_labels}")

# -------------------------------
# 4️⃣ Upload Image
# -------------------------------
uploaded_file = st.file_uploader("Upload a grain image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # 5️⃣ Preprocess Image
    # -------------------------------
    img = image.load_img(uploaded_file, target_size=(224,224))  # change if your model input differs
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------------
    # 6️⃣ Predict
    # -------------------------------
    try:
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_class = class_labels[predicted_index]
        confidence = np.max(predictions) * 100

        st.subheader("🔎 Prediction")
        st.success(f"Class: *{predicted_class}* ({confidence:.2f}% confidence)")
    except Exception as e:
        st.error(f"Prediction error: {e}")