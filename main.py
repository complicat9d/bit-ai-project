import os
import numpy as np
import tensorflow as tf
from utils.data_processor import preprocess_image
from utils.model_train import compile_model
from utils.log import logger
from config import settings

if __name__ == "__main__":

    def main():
        # Each time main.py is executed the model is recompiled to ensure consistency with results
        compile_model()

        model = tf.keras.models.load_model(os.path.join(os.getcwd(), "data/model.h5"))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Load and preprocess your image
        image_path = os.path.join(os.getcwd(), "data/photo/test/10.jpg")
        input_data = preprocess_image(image_path)

        predictions = model.predict(input_data)

        class_output_key = "class_output"
        class_output = predictions[class_output_key]
        labels = settings.LABELS

        # Reverse the labels dictionary for display
        labels_reversed = {v: k for k, v in labels.items()}

        num_classes = len(labels)

        # Get predicted class index and probabilities
        predicted_class_index = np.argmax(class_output)
        predicted_probabilities = class_output[0][:num_classes]

        # Create a dictionary to hold the best probabilities for each unique postfix
        best_items_by_postfix = {}

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
        predicted_label = labels_reversed[predicted_class_index]
        print(f"Predicted clothing item: {predicted_label}")
        logger.info(
            f"Normalized Fashion probability: {fashion_prob_sum:.4f}; Normalized Common probability: {common_prob_sum:.4f}"
        )
        print(f"Fashionability Index: {fashionability_index:.4f}")

        # Iterate through predictions to find the best item for each postfix
        for i in range(num_classes):
            label = labels_reversed[i]
            prob = predicted_probabilities[i]
            postfix = label.split("_")[1]  # Get the postfix (e.g., Jacket, Trousers)

            if (
                postfix not in best_items_by_postfix
                or prob > best_items_by_postfix[postfix][1]
            ):
                best_items_by_postfix[postfix] = (
                    label,
                    prob,
                )  # Store the best item for this postfix

        # Create a list of the best items
        best_items_list = list(best_items_by_postfix.values())

        # Sort the best items by probability
        sorted_best_items = sorted(best_items_list, key=lambda x: x[1], reverse=True)

        # Get the top 5 items by probability
        top_5_items = sorted_best_items[:5]

        # Print top 5 items with their probabilities
        print("Top 5 clothing items by probability:")
        top_5_items_probs = [prob for _, prob in top_5_items]

        # Normalize the top 5 item probabilities
        sum_probs = sum(top_5_items_probs)
        if sum_probs > 0:
            normalized_top_5_probs = [prob / sum_probs for prob in top_5_items_probs]
        else:
            normalized_top_5_probs = top_5_items_probs  # In case all are zero

        # Print normalized top 5 items with their probabilities
        for (label, _), normalized_prob in zip(top_5_items, normalized_top_5_probs):
            print(f"{label}: {normalized_prob * 100:.2f}%")

    main()
