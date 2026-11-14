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

# Paste your URL right here inside the quotes:
MODEL_URL = "https://github.com/reddylikhith999-bit/Image_project/releases/download/v1.0/chest_xray_model.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights from GitHub Releases... Please wait."):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
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