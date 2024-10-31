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
            "fashion_Jacket": 0,
            "fashion_Trousers": 1,
            "fashion_Bag": 2,
            "fashion_Sweater": 3,
            "fashion_Evening dress": 4,
            "fashion_Boots": 5,
            "fashion_Leggings": 6,
            "fashion_Heels": 7,
            "fashion_Jewelry": 8,
            "fashion_Sandals": 9,
            "fashion_Blouse": 10,
            "fashion_Skirt": 11,
            "fashion_Midi dress": 12,
            "fashion_Belt": 13,
            "fashion_Casual dress": 14,
            "fashion_Shorts": 15,
            "fashion_Tank top": 16,
            "fashion_Loafers": 17,
            "fashion_Gloves": 18,
            "fashion_Scarf": 19,
            "fashion_Trench coat": 20,
            "fashion_Sneakers": 21,
            "fashion_Cardigan": 22,
            "fashion_Coat": 23,
            "fashion_Shirt": 24,
            "fashion_Hoodie": 25,
            "fashion_T-shirt": 26,
            "fashion_Jeans": 27,
            "common_Shirt": 28,
            "common_Jeans": 29,
            "common_Shorts": 30,
            "common_Sandals": 31,
            "common_T-shirt": 32,
            "common_Loafers": 33,
            "common_Sneakers": 34,
            "common_Sweater": 35,
            "common_Trousers": 36,
            "common_Tank top": 37,
            "common_Belt": 38,
            "common_Boots": 39,
            "common_Cardigan": 40,
            "common_Jacket": 41,
            "common_Scarf": 42,
            "common_Skirt": 43,
            "common_Heels": 44,
            "common_Blouse": 45,
            "common_Flats": 46,
            "common_Bag": 47,
            "common_Evening dress": 48,
            "common_Casual dress": 49,
        }

        # Reverse the labels dictionary for display
        labels_reversed = {v: k for k, v in labels.items()}

        num_classes = len(labels)

        # Get predicted class index and probabilities
        predicted_class_index = np.argmax(class_output)
        predicted_probabilities = class_output[0][:num_classes]

        # Calculate sum of probabilities for fashion and common classes
        fashion_prob_sum = sum(
            predicted_probabilities[i]
            for label, i in labels.items()
            if label.startswith("fashion_")
        )
        common_prob_sum = sum(
            predicted_probabilities[i]
            for label, i in labels.items()
            if label.startswith("common_")
        )

        # Normalize probabilities
        total_prob_sum = fashion_prob_sum + common_prob_sum
        if total_prob_sum > 0:
            fashion_prob_sum /= total_prob_sum
            common_prob_sum /= total_prob_sum

        # Calculate the fashionability index
        fashionability_index = (
            fashion_prob_sum / (fashion_prob_sum + common_prob_sum)
            if (fashion_prob_sum + common_prob_sum) > 0
            else 0
        )

        # Print results
        print(f"Predicted class index: {predicted_class_index}")
        print(f"Normalized Fashion probability: {fashion_prob_sum:.4f}")
        print(f"Normalized Common probability: {common_prob_sum:.4f}")
        print(f"Fashionability Index: {fashionability_index:.4f}")

        # Create a sorted list of class probabilities
        sorted_probabilities = sorted(
            [
                (labels_reversed[i], prob)
                for i, prob in enumerate(predicted_probabilities)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        # Display class probabilities in descending order
        print("Class probabilities in descending order:")
        for label, prob in sorted_probabilities:
            print(f"{label}: {prob * 100:.2f}%")

    main()
