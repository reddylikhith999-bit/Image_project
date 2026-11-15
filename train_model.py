# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import os
import cv2
import joblib
import numpy as np

from skimage.feature import hog

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 2. DATASET PATHS
# ==========================================

TRAIN_PATH = "chest_xray/chest_xray/train"
TEST_PATH = "chest_xray/chest_xray/test"

IMAGE_SIZE = (128, 128)

CLASSES = ["NORMAL", "PNEUMONIA"]


# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image
    resized = cv2.resize(gray, IMAGE_SIZE)

    return resized


# ==========================================
# 4. FEATURE EXTRACTION
# ==========================================

def extract_features(image):
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# ==========================================
# 5. LOAD DATASET
# ==========================================

def load_dataset(folder_path):
    X = []
    y = []

    for label in CLASSES:
        class_path = os.path.join(folder_path, label)
        class_label = CLASSES.index(label)

        for image_name in os.listdir(class_path):
            # Skip hidden files like .DS_Store
            if image_name.startswith('.'):
                continue

            image_path = os.path.join(
                class_path,
                image_name
            )

            try:
                image = preprocess_image(image_path)
                features = extract_features(image)
                X.append(features)
                y.append(class_label)

            except Exception as error:
                print("Skipping:", image_path, error)

    return np.array(X), np.array(y)


# ==========================================
# 6. LOAD TRAINING AND TEST DATA
# ==========================================

print("Loading training images...")
X_train, y_train = load_dataset(TRAIN_PATH)
print("Training data:", X_train.shape)

print("Loading testing images...")
X_test, y_test = load_dataset(TEST_PATH)
print("Testing data:", X_test.shape)


# ==========================================
# 7. DEFINE ML MODELS
# ==========================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]),

    # Tip: If you want to restrict accuracy/prevent over-optimization, 
    # you can lower max_depth (e.g., max_depth=5) or reduce n_estimators.
    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="linear",
            C=0.01,
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ])
}


# ==========================================
# 8. TRAIN AND EVALUATE
# ==========================================

best_accuracy = -1
best_model = None
best_model_name = None

for name, model in models.items():
    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("Accuracy:", accuracy)
    print(
        classification_report(
            y_test,
            predictions,
            target_names=CLASSES,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name


# ==========================================
# 9. SAVE BEST MODEL
# ==========================================

os.makedirs("model", exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "model_name": best_model_name,
        "image_size": IMAGE_SIZE,
        "classes": CLASSES
    },
    "model/chest_xray_model.pkl"
)

print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy}")
print("Model saved successfully!")