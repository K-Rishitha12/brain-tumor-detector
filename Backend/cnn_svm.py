import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

IMG_SIZE = 128
CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

cnn_model = load_model(os.path.join(MODEL_DIR, "cnn_model.h5"))
svm_model = pickle.load(open(os.path.join(MODEL_DIR, "svm_model.pkl"), "rb"))

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img

def predict_tumor(img_path):
    img = preprocess_image(img_path)

    # Extract SAME features as training
    features = cnn_model.predict(img)

    prediction = svm_model.predict(features)[0]
    confidence = max(svm_model.predict_proba(features)[0])

    label = CATEGORIES[prediction]

    if label == "notumor":
        return "No Tumor Detected", confidence
    else:
        return f"Tumor Detected: {label.capitalize()}", confidence
