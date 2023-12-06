import numpy as np
import cv2


def convert_raw_to_image(input_file, output_file, width, height):
    # Read raw video data from the binary file
    with open(input_file, "rb") as file:
        raw_data = file.read()

    # Convert the raw data to NumPy array
    video_frames = np.frombuffer(raw_data, dtype=np.uint8)

    # Reshape the array to the original video dimensions (assuming BGR24 format)
    video_frames = video_frames.reshape((height, width, 3))

    # Normalize pixel values to the range [0, 255]
    video_frames = video_frames.astype(np.float32) * (255.0 / video_frames.max())

    # Convert from BGR to RGB
    video_frames_rgb = cv2.cvtColor(video_frames.astype(np.uint8), cv2.COLOR_BGR2RGB)

    # Save the video frames as an image
    cv2.imwrite(output_file, video_frames_rgb)


if __name__ == "__main__":
    input_file = (
        "stream_inputs/Al_Jazeera_timestamp.txt"  # Adjust the filename accordingly
    )
    output_file = "output_image.png"  # Adjust the output filename and format as needed
    width = 1920  # Adjust the width of the video frames
    height = 1080  # Adjust the height of the video frames

    convert_raw_to_image(input_file, output_file, width, height)
