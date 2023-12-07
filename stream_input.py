import subprocess
import time
import datetime
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd
import mediapipe as mp
from SIGNIT_convert import crop_hands
import math
import sys


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def draw_landmarks(image, filename, intermediate_images=False):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Convert the image to BGR format for Mediapipe
    image_rgb = image

    # Detect hand landmarks using mediapipe
    results = hands.process(image_rgb)

    hand_center = [0, 0]
    max_distance = 0
    height, width, _ = image.shape
    num_centers = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the average position of all landmarks (hand center)
            x_sum, y_sum = 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * width), int(landmark.y * height)
                x_sum += x
                y_sum += y
                if intermediate_images:
                    cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

            num_landmarks = len(hand_landmarks.landmark)
            hand_center[0] += int(x_sum / num_landmarks)
            hand_center[1] += int(y_sum / num_landmarks)
            num_centers += 1.0

        hand_center = [int(hi / num_centers) for hi in hand_center]
        if intermediate_images:
            cv2.circle(
                image, (int(hand_center[0]), int(hand_center[1])), 5, (0, 255, 0), -1
            )  # Draw green dot at the center

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmark_point = (int(landmark.x * width), int(landmark.y * height))
                distance = calculate_distance(hand_center, landmark_point)
                if distance > max_distance:
                    max_distance = distance
    else:
        # Set it to the center of the image if no hands are found
        center_x = width // 2
        center_y = height // 2

        hand_center = (center_x, center_y)
        max_distance = width

    if intermediate_images:
        saved = f"byproducts/intermediate/{filename}_intermediate1.png"
        cv2.imwrite(saved, image_rgb)

    return hand_center, max_distance


def image_to_parquet(
    current_time,
    filenames,
    image_data_list,
    parquet_file_path,
    intermediate_images=False,
):
    df_list = []  # List to store individual DataFrames for each image
    for filename, image_data in zip(filenames, image_data_list):
        # Open the image from in-memory data
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        # Convert the image to a NumPy array
        img_array = np.array(img)

        # Draw landmarks on the image
        center, max_distance = draw_landmarks(
            img_array, filename, intermediate_images=intermediate_images
        )

        coutput = crop_hands(img_array, center, max_distance)

        if intermediate_images:
            cv2.imwrite(
                f"byproducts/intermediate/{filename}_intermediate2.png", coutput
            )

        pixel_values = list(coutput.flatten())

        # Convert pixel values to a Pandas DataFrame
        df = pd.DataFrame(
            {
                "filename": filenames[0],
                "time": current_time,
                "Pixels": [pixel_values],
            }
        )
        df_list.append(df)

    # Concatenate individual DataFrames into a single DataFrame
    final_df = pd.concat(df_list, ignore_index=True)

    # from IPython.display import display
    # display(final_df)

    # Save the DataFrame to a Parquet file
    final_df.to_parquet(parquet_file_path, index=False)


def get_stream_url(url):
    try:
        result = subprocess.run(["youtube-dl", "-g", url], stdout=subprocess.PIPE)
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        print(f"Error fetching stream URL: {e}")
        return None


def pull_frames(stream_url, name, num_frames=5):
    try:
        # Get the current timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Use the name and timestamp in the filenames
        filename_parquet = f"byproducts/stream_inputs/{name}_{current_time}.parquet"

        image_data_list = []  # List to store image data for multiple frames

        # Use ffmpeg to extract raw video frames and capture the image data in-memory
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                stream_url,
                "-vf",
                f"fps={num_frames}",
                "-t",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ],
            stdout=subprocess.PIPE,
        )

        # Capture the image data for each frame
        for _ in range(num_frames):
            image_data_list.append(result.stdout)

        # Save the images as a parquet
        new_time = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")
        image_to_parquet(
            new_time,
            [f"{name}_{current_time}_{i}" for i in range(num_frames)],
            image_data_list,
            filename_parquet,
            intermediate_images=False,
        )

    except Exception as e:
        print(f"Error downloading frames: {e}")


def process_video(url, name):
    stream_url = get_stream_url(url)
    if stream_url:
        pull_frames(stream_url, name)


if __name__ == "__main__":
    # Check if enough command-line arguments are provided
    # Parse command-line arguments
    url = sys.argv[1]
    name = sys.argv[2]
    seconds_to_stop = int(sys.argv[3])

    start_time = time.time()

    while True:
        process_video(url, name)
        time.sleep(1)

        elapsed_time = time.time() - start_time
        if elapsed_time >= seconds_to_stop:
            print(f"Stopping after {seconds_to_stop} seconds.")
            break

    # URL_Name_List = [
    #     ("https://www.youtube.com/live/gCNeDWCI0vo?si=AAxtZcpFBL26CbEG", "Al"),
    #     # ("https://www.twitch.tv/test0251", "Test_Stream"),
    # ]

    # while True:
    #     with ThreadPoolExecutor() as executor:
    #         # Process each URL concurrently
    #         executor.map(lambda args: process_video(*args), URL_Name_List)

    #     time.sleep(1)
