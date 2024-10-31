import os
from rembg import remove


def remove_background(input_image_path, output_image_path):
    # Open the input image
    with open(input_image_path, "rb") as input_file:
        input_image = input_file.read()

    # Remove the background
    output_image = remove(input_image)

    # Save the output image
    with open(output_image_path, "wb") as output_file:
        output_file.write(output_image)


# Example usage
input_path = os.path.join(
    os.getcwd(), "data/photo/test/2.jpg"
)  # Change to your input image path
output_path = os.path.join(
    os.getcwd(), "output.jpg"
)  # Change to your desired output path
remove_background(input_path, output_path)
