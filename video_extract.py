import cv2
import sys
import numpy as np
import time
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    BinaryType,
)


def extract_frame_data(frame):
    # Extract metadata from the frame
    height, width, channels = frame.shape
    metadata = {"Height": height, "Width": width, "Channels": channels}

    # Flatten the pixel data
    pixel_data = cv2.resize(frame, (28, 28)).flatten().tobytes()
    return {"Metadata": str(metadata), "PixelData": pixel_data}


def process_video(video_path):
    # Create a Spark session
    spark = SparkSession.builder.appName("VideoFrameData").getOrCreate()

    # Define the schema for the DataFrame
    schema = StructType(
        [
            StructField("FrameNumber", IntegerType(), True),
            StructField("Data", StringType(), True),
            StructField("PixelData", BinaryType(), True),
        ]
    )

    # Read video frames using OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    # Initialize an empty list to store data for each frame
    frame_data_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract metadata and pixel data for each frame
        data = extract_frame_data(frame)

        # Append frame data to the list
        frame_data_list.append((frame_number, str(data["Metadata"]), data["PixelData"]))

        frame_number += 1

    # Release the video capture object
    cap.release()

    # Create a DataFrame from the frame data list
    frame_data_rdd = spark.sparkContext.parallelize(frame_data_list)
    frame_data_df = spark.createDataFrame(frame_data_rdd, schema)

    # Show the DataFrame
    print(frame_data_df.head())

    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    input = sys.argv[1]
    process_video(input)
