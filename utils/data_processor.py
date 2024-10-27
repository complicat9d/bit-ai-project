import os
import json
from typing import Generator, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def json_to_generator(
    filepath: str = "data/json/.json",
) -> Generator[Tuple[str, float, float, float, float, int], None, None]:
    with open(os.path.join(os.getcwd(), filepath), "r") as file:
        data = json.load(file)

    # Create label mapping
    label_mapping = {}
    current_label_index = 0

    for item in data:
        image_path = os.path.join(
            os.getcwd(), "data/photo/train/{}.jpg".format(item["id"])
        )

        for annotation in item["annotations"]:
            for result in annotation["result"]:
                box = result["value"]
                label = box["rectanglelabels"][0]

                if label not in label_mapping:
                    label_mapping[label] = current_label_index
                    current_label_index += 1

                yield (
                    image_path,
                    float(box["x"]),  # Convert to float
                    float(box["y"]),  # Convert to float
                    float(box["width"]),  # Convert to float
                    float(box["height"]),  # Convert to float
                    label_mapping[label],  # Ensure this is an int
                )


def generator_to_dataset(
    generator: Generator, batch_size: int = 32, image_size: Tuple[int, int] = (256, 256)
) -> tf.data.Dataset:
    # Create a TensorFlow dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),  # Image path
            tf.TensorSpec(shape=(), dtype=tf.float32),  # x
            tf.TensorSpec(shape=(), dtype=tf.float32),  # y
            tf.TensorSpec(shape=(), dtype=tf.float32),  # width
            tf.TensorSpec(shape=(), dtype=tf.float32),  # height
            tf.TensorSpec(shape=(), dtype=tf.int32),  # label
        ),
    )

    # Map the dataset to extract the annotations and process images
    def process_image(
        image_path: tf.Tensor,
        x: tf.Tensor,
        y: tf.Tensor,
        width: tf.Tensor,
        height: tf.Tensor,
        label: tf.Tensor,
    ):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        return image, {
            "class_output": label,  # class label
            "bbox_output": tf.stack([x, y, width, height]),  # bounding box coordinates
        }

    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# Preprocess the image
def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


if __name__ == "__main__":
    gen = json_to_generator()
    dataset = generator_to_dataset(gen)
    print(dataset)
    for images, annotations in dataset.take(2):
        print(images.shape, annotations)
