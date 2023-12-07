from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, udf, col, to_timestamp
from pyspark.sql.types import (
    StructType,
    StructField,
    ArrayType,
    IntegerType,
    StringType,
)
import numpy as np
from tensorflow.keras.models import load_model
import sys

# Initialize Spark session
spark = SparkSession.builder.appName("ImageStream").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
no_sanity = False

# Define the schema for the Parquet files
parquet_schema = StructType(
    [
        StructField("Filename", StringType()),
        StructField("Time", StringType()),
        StructField("Pixels", ArrayType(IntegerType())),
    ]
)

# Define the input and output paths for structured streaming
input_path = "byproducts/stream_inputs/"
output_path = "byproducts/stream_outputs/"

# Read new Parquet files as a streaming DataFrame
streaming_df = spark.readStream.schema(parquet_schema).parquet(input_path)

class_names = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "Del",
    "N",
    "Space",
]


def sanity_check(processed_df):
    query2 = processed_df.writeStream.outputMode("update").format("console").start()

    # Explode the processed pixels
    processed_df = processed_df.select(explode(col("Pixels")).alias("Pixel"))

    # Display the processed DataFrame
    sanity_check = "byproducts/stream_sanity_check"
    query3 = (
        processed_df.writeStream.format("csv")
        .option("header", "true")
        .option(
            "checkpointLocation", "byproducts/checkpoint"
        )  # Add a checkpoint location
        .option("path", sanity_check)  # Specify the output path
        .start()
    )
    return query2, query3


# Define a user-defined function (UDF) to process pixels
def process_pixels(pixel_array):
    # Convert the pixel value to a NumPy array
    pixel_array = np.array(pixel_array, dtype=np.uint8)

    # Reshape the array to the desired shape (assuming square image)
    image_size = (1, 100, 100, 1)
    pixel_array = pixel_array.reshape(image_size)

    model = load_model("model_small.keras")
    predictions = model.predict(pixel_array)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_class_names = [class_names[i] for i in predicted_indices]

    # Return the flattened array
    return predicted_class_names[0]


# Register the UDF
process_pixels_udf = udf(lambda z: process_pixels(z), StringType())

# Process the pixels before exploding
processed_df = (
    streaming_df.withColumn("Signed_Letter", process_pixels_udf(col("Pixels")))
    .withColumn("timestamp", to_timestamp("Time", "yyyyMMdd HH:mm:ss"))
    .select(["Filename", "timestamp", "Signed_Letter", "Pixels"])
)

final_df = (
    processed_df.withWatermark("timestamp", "10 seconds")
    .groupBy("Filename")
    .agg({"Signed_Letter": "max"})
)

if no_sanity:
    query2, query3 = sanity_check(processed_df)
query = final_df.writeStream.outputMode("update").format("console").start()

# query = (
#     exploded_df.writeStream.outputMode("append")
#     .format("console")
#     .start()
# )
seconds = int(sys.argv[1])
try:
    # Keep the script running to continuously process new data
    query.awaitTermination(timeout=seconds)
    if no_sanity:
        query2.awaitTermination()
        query3.awaitTermination()

except KeyboardInterrupt:
    # Stop the streaming query if interrupted
    query.stop()

finally:
    # Stop the Spark session
    spark.stop()
