import pandas as pd
import numpy as np
import cv2

# Load the CSV file into a DataFrame
csv_file = "byproducts\stream_sanity_check\part-00000-b02513e7-59bf-4543-aeb7-809a494637a9-c000.csv"

# Read the CSV file
df = pd.read_csv(csv_file, encoding="utf-8", engine="python")

# Extract pixel values from the 'pixel' column
pixels = df["Pixel"].values

# Convert the flattened pixel values to a NumPy array
pixel_array = np.array(pixels, dtype=np.uint8)

num = pixel_array.size // 10000
# Reshape the pixel array to match the dimensions of the image (100x100x5)
image_shape = (100, 100, num)
# Explicitly specify the order as 'F' (Fortran-order) to match OpenCV's expectation
image = pixel_array.reshape(image_shape, order="F")
image = np.rot90(image, k=-1)
image = np.flip(image, axis=1)

# Convert the pixel values to uint8 (0-255) for image display
for i in range(num):
    nimage = image[:, :, i : i + 1]  # Extract the i-th channel

    # Display the image using OpenCV
    cv2.imshow(f"Channel {i+1}", nimage)
    cv2.waitKey(0)

cv2.destroyAllWindows()
