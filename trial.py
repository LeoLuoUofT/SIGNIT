from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
import sys

def process_new_files(df, epoch_id):
    # Filter only new files
    new_files_df = df.select("value", input_file_name().alias("file_name"))

    # Show the names of new text files
    new_files_df.show(truncate=False)

if __name__ == "__main__":
    # Initialize a Spark session
    spark = SparkSession.builder \
        .appName("StructuredFileStream") \
        .getOrCreate()

    if len(sys.argv) != 2:
        print("Usage: python script.py <input_folder>")
        sys.exit(1)

    # Specify the path to the folder containing text files
    input_folder = sys.argv[1]

    # Define a streaming DataFrame that represents data from a text file
    lines = spark.readStream.text(input_folder)

    # Process new files in the streaming DataFrame once per second
    query = lines.writeStream \
        .outputMode("append") \
        .foreachBatch(process_new_files) \
        .trigger(processingTime="1 seconds").start()

    # Await termination
    query.awaitTermination()
