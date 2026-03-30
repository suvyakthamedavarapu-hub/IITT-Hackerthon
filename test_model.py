import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras

# Step 1: Load model
model = keras.models.load_model("trained_model.h5")

# Step 2: Load image
img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)

# Step 3: Resize to 28x28
img = cv2.resize(img, (28, 28))

# Step 4: Normalize (VERY IMPORTANT)
img = img / 255.0

# Step 5: Reshape to match model input
img = np.reshape(img, (1, 28, 28))

# Step 6: Predict
prediction = model.predict(img)

# Step 7: Get digit
predicted_digit = np.argmax(prediction)

# Step 8: Output
print("Predicted Digit:", predicted_digit)
