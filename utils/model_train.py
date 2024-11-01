import tensorflow as tf

from utils.data_processor import json_to_generator, generator_to_dataset
from config import settings


def create_model(input_shape=(256, 256, 3), num_classes=58):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)

    # Output layer for class probabilities
    class_output = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="class_output"
    )(x)

    # Output layer for bounding box coordinates
    bbox_output = tf.keras.layers.Dense(4, activation="linear", name="bbox_output")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={"class_output": class_output, "bbox_output": bbox_output},
    )

    return model


def compile_model(epochs: int = 20, batch_size: int = 32):
    gen = json_to_generator()
    dataset = generator_to_dataset(gen, batch_size=batch_size, image_size=(256, 256))

    # Create the model
    model = create_model(
        input_shape=(256, 256, 3), num_classes=58
    )  # Adjust num_classes as needed

    # Compile the model
    model.compile(
        optimizer="adam",
        loss={
            "class_output": "sparse_categorical_crossentropy",
            "bbox_output": "mean_squared_error",
        },
        metrics={
            "class_output": "accuracy",
            "bbox_output": "mean_absolute_error",  # You can use another metric for bounding boxes
        },
    )

    # Fit the model
    model.fit(dataset, epochs=epochs)  # Adjust the number of epochs based on your needs

    model.save(settings.MODEL_PATH)


if __name__ == "__main__":
    compile_model()
