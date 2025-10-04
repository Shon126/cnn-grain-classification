import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import json
import os

# -------------------------------
# 1️⃣ App title
# -------------------------------
st.set_page_config(page_title="Grain Classifier", page_icon="🌾")
st.title("🌾 Grain Classification App")
st.write("Upload an image of a grain, and the AI will tell you what it is!")

# -------------------------------
# 2️⃣ Load model and class labels
# -------------------------------
MODEL_PATH = "grains_mobilenetv2_cleaned.keras"
LABELS_PATH = "class_labels.json"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found! Check your path: {MODEL_PATH}")
else:
    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")

if not os.path.exists(LABELS_PATH):
    st.error(f"Class labels file not found! Check your path: {LABELS_PATH}")
else:
    with open(LABELS_PATH, 'r') as f:
        class_labels = json.load(f)
    st.write("Class labels loaded:", list(class_labels.values()))

# -------------------------------
# 3️⃣ Image upload
# -------------------------------
uploaded_file = st.file_uploader("Choose a grain image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # -------------------------------
    # 4️⃣ Preprocess image
    # -------------------------------
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension

    # -------------------------------
    # 5️⃣ Predict
    # -------------------------------
    prediction = model.predict(img_array)
    pred_class_index = np.argmax(prediction, axis=1)[0]
    pred_class_label = class_labels[str(pred_class_index)]
    pred_confidence = prediction[0][pred_class_index]

    st.success(f"Predicted Grain: *{pred_class_label}*")
    st.write(f"Confidence: {pred_confidence*100:.2f}%")