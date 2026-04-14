import os
import cv2
import numpy as np

IMG_SIZE = 224

def load_data(data_dir):
    data = []
    labels = []

    for category in ["NORMAL", "PNEUMONIA"]:
        path = os.path.join(data_dir, category)
        label = 0 if category == "NORMAL" else 1

        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0

                data.append(img)
                labels.append(label)
            except:
                continue

    return np.array(data), np.array(labels)