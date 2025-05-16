import joblib
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.datasets import mnist
import numpy as np

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_flat, y_train)

# Evaluate
print("Accuracy:", knn.score(x_test_flat, y_test))

# Save model
joblib.dump(knn, 'KNN.sav')