import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load model
new_model = tf.keras.models.load_model('digit_CNN.model')
# Predicting your own image
img = cv2.imread('number.png') # Load image
plt.imshow(img)
plt.show()
print(img.shape)

convertToGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(convertToGray.shape)

resized = cv2.resize(convertToGray, (28, 28), interpolation = cv2.INTER_AREA)

print(resized.shape)
plt.imshow(resized)
plt.show()
# Image normalization before entering the neural network (convert from 0-255 to 0-1)
normalizeImage = tf.keras.utils.normalize(resized, axis = 1)
normalizeImage = np.array(normalizeImage).reshape(-1, 28, 28, 1)

newPrediction = new_model.predict(normalizeImage)
print(np.argmax(newPrediction))