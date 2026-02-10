import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "backend", "dataset", "testing")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 128

data = []

for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    class_num = CATEGORIES.index(category)

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append([img_array, class_num])
        except:
            pass

np.random.shuffle(data)

X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array(y)

# -------- CNN FEATURE EXTRACTOR --------
inputs = Input(shape=(128,128,1))
x = Conv2D(32, (3,3), activation='relu')(inputs)
x = MaxPooling2D(2,2)(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D(2,2)(x)
x = Flatten()(x)
feature_layer = Dense(128, activation='relu', name="feature_layer")(x)

cnn_model = Model(inputs=inputs, outputs=feature_layer)

# Extract features
features = cnn_model.predict(X)

# -------- SVM --------
svm = SVC(kernel='linear', probability=True)
svm.fit(features, y)

# Save
cnn_model.save(os.path.join(MODEL_DIR, "cnn_model.h5"))
pickle.dump(svm, open(os.path.join(MODEL_DIR, "svm_model.pkl"), "wb"))

print("✅ Model trained and saved correctly")
