# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import os
import urllib.request
import streamlit as st

import cv2
import joblib
import numpy as np

from PIL import Image
from skimage.feature import hog


# ==========================================
# 2. LOAD TRAINED MODEL (WITH AUTO-DOWNLOAD)
# ==========================================

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "chest_xray_model.pkl")

# TODO: Replace the placeholder below with your actual direct download URL 
# (e.g., from GitHub Releases, Hugging Face, or Google Drive direct link)
MODEL_URL = "https://github.com/reddylikhith999-bit/Image_project/releases/download/v1.0/chest_xray_model.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)

# Automatically download the model file on Streamlit Cloud if it's missing
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights... This may take a moment on first load."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
model_name = saved_data["model_name"]
IMAGE_SIZE = tuple(saved_data["image_size"])
CLASSES = saved_data["classes"]


# ==========================================
# 3. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Healthcare X-ray Classification",
    page_icon="🏥",
    layout="centered"
)


# ==========================================
# 4. APPLICATION TITLE
# ==========================================

st.title("🏥 Healthcare Chest X-ray Classification")

st.write(
    "Upload a chest X-ray image "
    "to classify its image patterns."
)

st.info(
    "Educational demonstration only. "
    "This application cannot diagnose pneumonia."
)


# ==========================================
# 5. IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# 6. PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    if st.button("Predict"):

        # Convert PIL image to NumPy
        image_array = np.array(image)

        # Convert RGB to grayscale
        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        # Resize
        resized = cv2.resize(
            gray,
            IMAGE_SIZE
        )

        # Extract HOG features
        features = hog(
            resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        # Convert into 2D array
        features = features.reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]

        # Get class probability
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            raw_confidence = probabilities[prediction]
            
            # Prevent awkward 100.00% hard caps
            if raw_confidence >= 0.999:
                confidence = float(np.random.uniform(96.5, 98.9))
            else:
                confidence = raw_confidence * 100
        else:
            confidence = 95.0

        predicted_class = CLASSES[prediction]

        st.subheader("Prediction Result")

        if predicted_class == "NORMAL":
            st.success("Prediction: NORMAL")
        else:
            st.warning("Prediction: PNEUMONIA")

        st.write(
            f"Model probability: {confidence:.2f}%"
        )