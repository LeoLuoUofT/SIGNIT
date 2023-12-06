from PIL import Image


def image_to_text(image_path, text_file_path):
    # Open the image file
    img = Image.open(image_path)

    # Convert the image to text (pixel values)
    pixel_values = list(img.getdata())

    # Save the pixel values to a text file
    with open(text_file_path, "w") as text_file:
        for pixel in pixel_values:
            text_file.write(",".join(map(str, pixel)) + "\n")


def text_to_image(text_file_path, output_image_path, image_size):
    # Read pixel values from the text file
    with open(text_file_path, "r") as text_file:
        pixel_values = [tuple(map(int, line.strip().split(","))) for line in text_file]

    # Create a new image using the pixel values
    img = Image.new("RGB", image_size)
    img.putdata(pixel_values)

    # Save the image to a file
    img.save(output_image_path)

    # Display the image
    img.show()


if __name__ == "__main__":
    # Path to the input image
    input_image_path = "stream_inputs/B_test.jpg"

    # Path to the text file to save the pixel values
    text_file_path = "stream_inputs/B_test.txt"

    # Path to the output image
    output_image_path = "stream_inputs/B_test_new.jpg"

    # Size of the image (width, height)
    image_size = (200, 200)  # Specify the actual width and height of the image

    # Convert image to text file
    image_to_text(input_image_path, text_file_path)

    # Convert text file back to image and display
    text_to_image(text_file_path, output_image_path, image_size)
