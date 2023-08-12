import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from PIL import Image


# Assuming you have a list of image paths and corresponding labels
image_paths = ['C:\Downloads\blood.jpg']
labels = [0]  # List of corresponding labels

num_samples = len(image_paths)
image_width, image_height = 128, 128
num_channels = 3  # Assuming RGB images

# Create empty arrays to store the preprocessed images and labels
images = np.empty((num_samples, image_width, image_height, num_channels), dtype=np.float32)

# Load and preprocess the images
for i, image_path in enumerate(image_paths):
    try:

     image = Image.open('blood.jpg').resize((image_width, image_height))
     image_array = np.array(image, dtype=np.float32)
     image_array /= 255.0
     images[i] = image_array
    except FileNotFoundError:
     print(f"Error: File not found at path: {image_path}")

# Convert the labels to a numpy array
labels = np.array(labels, dtype=np.int32)

#Split the data into training and testing sets
#train_images, test_images, train_labels, test_labels = train_test_split(
    #images, labels, test_size=0.0, random_state=42)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, num_channels)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Assuming binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(images, labels, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(images,labels, verbose=2)
print('Test accuracy:', test_acc)