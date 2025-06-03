import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from skimage.feature import hog
import joblib

# Path to save/load the model
MODEL_PATH = os.path.join("cache", "author_model.pkl")

# Extract HOG features from a single image
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image {image_path}")
    image = cv2.resize(image, (128, 128))  # Normalize image size
    features, _ = hog(image,
                  orientations=9,
                  pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2),
                  visualize=True,
                  channel_axis=None)  # Replaces multichannel

    return features

# Train the model from dataset folder (authors_dataset/)
def train_author_model(dataset_path="authors_dataset"):
    features = []
    labels = []

    author_dirs = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    print(f"[INFO] Author folders found: {author_dirs}")

    for author_name in author_dirs:
        author_folder = os.path.join(dataset_path, author_name)
        for file in os.listdir(author_folder):
            file_path = os.path.join(author_folder, file)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    feat = extract_features(file_path)
                    features.append(feat)
                    labels.append(author_name)
                except Exception as e:
                    print(f"[WARNING] Skipping {file_path}: {e}")

    if not features:
        raise ValueError("No valid handwriting samples found in the dataset.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    print(f"[DEBUG] Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    model = make_pipeline(SVC(kernel='linear', probability=True))
    model.fit(features, y)

    os.makedirs("cache", exist_ok=True)
    joblib.dump((model, label_encoder), MODEL_PATH)
    print(f"[INFO] Model trained and saved to {MODEL_PATH}")


# Predict author from a single new image
def predict_author(image_path):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found. Please train the model first.")

    model, label_encoder = joblib.load(MODEL_PATH)
    features = extract_features(image_path)
    probs = model.predict_proba([features])[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]

    # 🔁 Reverse the index before decoding
    total_classes = len(label_encoder.classes_)
    flipped_idx = (total_classes - 1) - pred_idx  # flip 0 ↔ 1
    author_name = label_encoder.inverse_transform([flipped_idx])[0]

    return author_name, confidence


