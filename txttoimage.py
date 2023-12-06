import urllib.parse
import numpy as np
import cv2
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
from pyspark.sql.types import StructType, StructField, StringType

# Define the schema for the DataFrame
schema = StructType([StructField("pixels", StringType())])

# Create a Spark session
spark = SparkSession.builder.appName("ImageConversionApp").getOrCreate()


# Define the convert_raw_to_image function
def convert_raw_to_image(input_file):
    # Parse the URL-encoded file path
    parsed_path = urllib.parse.unquote(input_file)

    # Read raw video data from the binary file
    with open(parsed_path, "rb") as file:
        raw_data = file.read()

    # Convert the raw data to NumPy array
    video_frames = np.frombuffer(raw_data, dtype=np.uint8)

    # Reshape the array to the original video dimensions (assuming BGR24 format)
    video_frames = video_frames.reshape((1080, 1920, 3))

    # Normalize pixel values to the range [0, 255]
    video_frames = video_frames.astype(np.float32) * (255.0 / video_frames.max())

    # Convert from BGR to RGB
    video_frames_rgb = cv2.cvtColor(video_frames.astype(np.uint8), cv2.COLOR_BGR2RGB)

    # Convert the array of integers to a string
    pixels_str = ",".join(map(str, video_frames_rgb.flatten().tolist()))

    return pixels_str


# Input and output folders
input_folder = "stream_inputs"
output_folder = "stream_outputs"

# Define the input schema for the streaming DataFrame
streaming_df = (
    spark.readStream.format("text")
    .option("maxFilesPerTrigger", 1)  # Process one file per trigger
    .schema(schema)
    .load(input_folder)
)

# Apply the convert_raw_to_image UDF to create the DataFrame with pixel values
convert_udf = spark.udf.register(
    "convert_raw_to_image", convert_raw_to_image, StringType()
)
processed_df = streaming_df.withColumn("pixels", convert_udf(input_file_name()))

# Define the query to write the DataFrame to the output folder as text files
query = (
    processed_df.writeStream.outputMode("append")
    .format("text")
    .option("checkpointLocation", "checkpoint")  # Specify a checkpoint directory
    .start(output_folder)
)

# Await termination of the query
query.awaitTermination()
