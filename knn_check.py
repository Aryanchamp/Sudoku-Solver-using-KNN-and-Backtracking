import joblib
import cv2
import numpy as np

model = joblib.load('KNN.sav')
img = cv2.imread('cell_1_0_pred_0_conf_0.20.jpg', cv2.IMREAD_GRAYSCALE)  # Use a clear image of a digit (e.g., 5)
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
resized = cv2.resize(thresh, (28, 28))
normalized = resized / 255.0
flattened = normalized.flatten().reshape(1, -1)
print(model.predict(flattened))