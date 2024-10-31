import os
from rembg import remove


def remove_background(input_image_path: str, output_image_path: str):
    # Open the input image
    with open(input_image_path, "rb") as input_file:
        input_image = input_file.read()

    # Remove the background
    output_image = remove(input_image)

    # Save the output image
    with open(output_image_path, "wb") as output_file:
        output_file.write(output_image)


def process_images_background(input_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{filename}")
            remove_background(input_path, output_path)
            print(f"Processed: {input_path} -> {output_path}")


input_folder = os.path.join(os.getcwd(), "data/photo/train")
output_folder = os.path.join(os.getcwd(), "output")
process_images_background(input_folder, output_folder)
