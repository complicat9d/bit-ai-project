import tensorflow as tf

from utils.data_processor import json_to_generator, generator_to_dataset


def create_model(input_shape=(256, 256, 3), num_classes=58):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=input_shape
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(
                num_classes, activation="softmax"
            ),  # Change activation for regression
        ]
    )
    return model


if __name__ == "__main__":
    gen = json_to_generator()
    dataset = generator_to_dataset(gen, batch_size=32, image_size=(256, 256))

    # Create the model
    model = create_model(
        input_shape=(256, 256, 3), num_classes=58
    )  # Adjust num_classes as needed

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # Use 'categorical_crossentropy' for one-hot encoded labels
        metrics=["accuracy"],
    )

    # Fit the model
    model.fit(dataset, epochs=10)  # Adjust the number of epochs based on your needs

    model.save("my_model.h5")
