import os
import numpy as np
import tensorflow as tf
from utils.data_processor import preprocess_image

if __name__ == "__main__":

    def main():
        # Load your model
        model = tf.keras.models.load_model(os.path.join(os.getcwd(), "data/model.h5"))
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

        # Define the labels for the classes
        labels = {
            # (same labels as before)
        }

        # Reverse the labels dictionary for display
        labels_reversed = {v: k for k, v in labels.items()}

        num_classes = len(labels)

        # Get predicted class index and probabilities
        predicted_class_index = np.argmax(class_output)
        predicted_probabilities = class_output[0][:num_classes]

        # Calculate sum of probabilities for fashion and common classes
        fashion_prob_sum = sum(predicted_probabilities[i] for label, i in labels.items() if label.startswith("fashion_"))
        common_prob_sum = sum(predicted_probabilities[i] for label, i in labels.items() if label.startswith("common_"))

        # Normalize probabilities
        total_prob_sum = fashion_prob_sum + common_prob_sum
        if total_prob_sum > 0:
            fashion_prob_sum /= total_prob_sum
            common_prob_sum /= total_prob_sum

        # Calculate the fashionability index
        fashionability_index = fashion_prob_sum / (fashion_prob_sum + common_prob_sum) if (fashion_prob_sum + common_prob_sum) > 0 else 0

        # Print results
        print(f"Predicted class index: {predicted_class_index}")
        print(f"Normalized Fashion probability: {fashion_prob_sum:.4f}")
        print(f"Normalized Common probability: {common_prob_sum:.4f}")
        print(f"Fashionability Index: {fashionability_index:.4f}")

        # Create a sorted list of class probabilities
        sorted_probabilities = sorted(
            [(labels_reversed[i], prob) for i, prob in enumerate(predicted_probabilities)],
            key=lambda x: x[1],
            reverse=True
        )

        # Display class probabilities in descending order
        print("Class probabilities in descending order:")
        for label, prob in sorted_probabilities:
            print(f"{label}: {prob * 100:.2f}%")

    main()
