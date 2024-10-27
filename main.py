import os
import numpy as np
import tensorflow as tf

from utils.data_processor import preprocess_image


if __name__ == "__main__":

    def main():
        # Load your model
        model = tf.keras.models.load_model(os.path.join(os.getcwd(), "/data/model.h5"))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Load and preprocess your image
        image_path = os.path.join(os.getcwd(), "data/photo/test/3.jpg")
        input_data = preprocess_image(image_path)

        # Make predictions
        predictions = model.predict(input_data)

        class_output_key = "class_output"
        class_output = predictions[class_output_key]

        # Define the labels for the 5 classes
        labels = {"Shoes": 0, "Trousers": 1, "Suit": 2, "Hair": 3, "Shirt": 4}
        num_classes = len(labels)

        # Get predicted class index and probabilities
        predicted_class_index = np.argmax(class_output)
        predicted_probabilities = class_output[0][
            :num_classes
        ]  # Only the relevant probabilities for the first image

        # Reverse the labels dictionary
        labels_reversed = {v: k for k, v in labels.items()}

        # Get the predicted class label and confidence
        predicted_class_label = labels_reversed[predicted_class_index]
        predicted_confidence = (
            predicted_probabilities[predicted_class_index] * 100
        )  # Convert to percentage

        # Print results
        print(f"Predicted class index: {predicted_class_index}")
        print(f"Predicted class label: {predicted_class_label}")
        print(f"Confidence: {predicted_confidence:.2f}%")

        # Display only the class probabilities for the specified labels
        print("Class probabilities for specified labels:")
        for label, index in labels.items():
            prob = predicted_probabilities[index] * 100
            print(f"{label}: {prob:.2f}%")

    main()
