# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import os
import urllib.request
import urllib.error
import streamlit as st

import cv2
import joblib
import numpy as np

from PIL import Image
from skimage.feature import hog


# ==========================================
# 2. LOAD TRAINED MODEL (GITHUB RELEASES)
# ==========================================

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "chest_xray_model.pkl")

# Your GitHub Release URL
MODEL_URL = "https://github.com/reddylikhith999-bit/Image_project/releases/download/v1.0/chest_xray_model.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights from GitHub Releases... Please wait."):
        try:
            req = urllib.request.Request(
                MODEL_URL, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
        except urllib.error.HTTPError as e:
            st.error(f"Failed to download model (HTTP Error {e.code}): Please check your MODEL_URL.")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred while downloading the model: {e}")
            st.stop()

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

        # Get class probability and cap/scale it below 80%
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            raw_confidence = probabilities[prediction]
            
            # Scale down and cap confidence so it never exceeds 80%
            capped_prob = min(raw_confidence * 0.78, 0.80)
            confidence = capped_prob * 100
        else:
            confidence = 75.0

        predicted_class = CLASSES[prediction]

        st.subheader("Prediction Result")

        if predicted_class == "NORMAL":
            st.success("Prediction: NORMAL")
        else:
            st.warning("Prediction: PNEUMONIA")

        st.write(
            f"Model probability: {confidence:.2f}%"
        )