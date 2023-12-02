import csv
import numpy as np
import cv2


def csv_to_image(csv_path, image_path, width, height):
    # Read CSV data
    with open(csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        flattened_pixels = [float(pixel) for pixel in next(reader)]

    # Reshape the flattened pixels into the original image shape
    image_data = np.array(flattened_pixels).reshape((height, width, 3))

    # Convert pixel values back to uint8 (assuming 8-bit per channel)
    image_data = np.clip(image_data, 0, 255).astype(np.uint8)

    # Save the image
    cv2.imwrite(image_path, image_data)


if __name__ == "__main__":
    # Replace 'path/to/your/input.csv' with the actual path to your CSV file
    csv_path = "Leo_testing_folder/output_images/O_test.csv"

    # Replace 'path/to/your/output_image.png' with the desired output path and file name
    image_path = "Leo_testing_folder/out.png"

    # Set the width and height of the image (adjust as needed)
    width, height = 128, 128

    # Call the function to convert CSV to image and save
    csv_to_image(csv_path, image_path, width, height)
