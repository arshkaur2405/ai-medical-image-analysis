import cv2
import numpy as np
import tensorflow as tf

load_model = tf.keras.models.load_model

IMG_SIZE = 224

model = load_model("models/model.h5")

def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return "Image not found"

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    pred = model.predict(img)

    if pred > 0.5:
        return "PNEUMONIA"
    else:
        return "NORMAL"