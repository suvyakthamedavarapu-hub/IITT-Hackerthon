import pickle
import cv2
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load sample image
img = cv2.imread("sample.jpg")

# Preprocess (same as training)
img = cv2.resize(img, (64, 64))
img = img.flatten()
img = np.array([img])

# Predict
prediction = model.predict(img)

# Output
print("Predicted Feature:", prediction)