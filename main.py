import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Load your model
model = tf.keras.models.load_model(os.path.join(os.getcwd(), "my_model.h5"))


# Preprocess the image
def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Load and preprocess your image
image_path = os.path.join(os.getcwd(), "data/photo/3.jpg")
print(image_path)
input_data = preprocess_image(image_path)

# Make predictions
predictions = model.predict(input_data)

# Get the predicted class (for classification)
predicted_class = np.argmax(predictions, axis=1)
print(f'Predicted class: {predicted_class}')
