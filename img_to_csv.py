from pyspark import SparkContext, SparkConf
import numpy as np
import cv2
import os
import mediapipe as mp
import math
import re
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType
from tensorflow.keras.models import load_model
import sys

alphabet_mapping = {
    chr(i): i - ord("A") if "A" <= chr(i) <= "Z" else i - ord("a")
    for i in range(ord("A"), ord("Z") + 1)
}

# Adding 'nothing', 'space', and 'del'
alphabet_mapping["del"] = 26
alphabet_mapping["nothing"] = 27
alphabet_mapping["space"] = 28


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def hand_detection(pair):
    # Extract the file name and binary content from the RDD pair
    file_name, binary_content = pair

    image_array = np.frombuffer(binary_content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Specify the output path to save the image/csv

    # output_path = os.path.join(
    #     output_folder, f"{file_name.split('/')[-1].split('.')[0]}.csv"
    # )

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Process the image and detect hand landmarks
    results = hands.process(image_rgb)

    height, width, _ = image.shape

    # Calculate the center of the hand and maximum distance to any landmark
    hand_center = [0, 0]
    num_centers = 0.0
    max_distance = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the average position of all landmarks (hand center)
            x_sum, y_sum = 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * width), int(landmark.y * height)
                x_sum += x
                y_sum += y

            num_landmarks = len(hand_landmarks.landmark)
            hand_center[0] += int(x_sum / num_landmarks)
            hand_center[1] += int(y_sum / num_landmarks)
            num_centers += 1.0

        hand_center = [int(hi / num_centers) for hi in hand_center]

        # Calculate the maximum distance to any landmark
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

    # Draw the center on the image and save the result to a file
    # cv2.circle(image, hand_center, 10, (0, 255, 0), -1)
    # cv2.imwrite(output_path, image)

    label = re.split(r"[0-9_]+", file_name.split("/")[11])[0]

    coutput = crop_hands(image, hand_center, max_distance)
    flattened_pixels = np.reshape(coutput, [1, coutput.size])

    # save as csv file
    # np.savetxt(output_folder, flattened_pixels, fmt='%d', delimiter=",")
    # print(flattened_pixels.shape)

    if label in alphabet_mapping:
        label = alphabet_mapping[label]
    else:
        label = 27

    return label, flattened_pixels


def create_data(pair):
    file_name, binary_content = pair

    image_array = np.frombuffer(binary_content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    label = re.split(r"[0-9_]+", file_name.split("/")[12])[0]

    target_size = (100, 100)
    resized_image = cv2.resize(image, target_size)
    recolored = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    coutput = recolored
    flattened_pixels = np.reshape(coutput, [1, coutput.size])

    if label in alphabet_mapping:
        label = alphabet_mapping[label]
    else:
        label = 27

    return label, flattened_pixels


def crop_hands(image, hand_center, max_distance):
    # Calculate the bounding box dimensions
    x_min = max(0, int(hand_center[0] - 2 * max_distance))
    y_min = max(0, int(hand_center[1] - 2 * max_distance))
    x_max = min(int(image.shape[1]), int(hand_center[0] + 2 * max_distance))
    y_max = min(int(image.shape[0]), int(hand_center[1] + 2 * max_distance))

    # Behavior of none square image
    width = x_max - x_min
    height = y_max - y_min

    if width > height:
        diff = (width - height) // 2
        x_min = x_min + diff
        x_max = x_max - diff
    elif height > width:
        diff = height - width
        y_min = y_min + diff
        y_max = y_max - diff

    # Crop the image around the hands
    cropped_image = image[y_min:y_max, x_min:x_max]
    target_size = (100, 100)
    resized_image = cv2.resize(cropped_image, target_size)
    recolored = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Save the cropped image to a file
    # cv2.imwrite(output_path, resized_image)
    return recolored


def create_dataset_csv(rdd):
    # for each in binary_rdd.collect():
    #     numpy_arrays.append(np.frombuffer(each[1], np.uint8))

    labels = rdd.map(lambda x: x[0]).collect()
    numpy_arrays = labels_and_arrays.map(lambda x: x[1]).collect()
    labels = np.array(labels).reshape(len(labels), 1)

    # Concatenate the NumPy arrays into one
    result_array = np.concatenate((labels, np.concatenate(numpy_arrays)), axis=1)

    return result_array


if __name__ == "__main__":
    conf = SparkConf().setAppName("BinaryFile")
    sc = SparkContext(conf=conf)

    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <output_folder>")
        sys.exit(1)

    # Specify the path to the folder containing binary files
    image_path = sys.argv[1]
    output_folder = sys.argv[2]

    # Read binary files into an RDD
    binary_rdd = sc.binaryFiles(image_path)

    # labels_and_arrays = binary_rdd.map(hand_detection)

    # just for dataset
    labels_and_arrays = binary_rdd.map(create_data)

    # create dataset
    # result_array = create_dataset_csv(labels_and_arrays)
    # np.savetxt(output_folder, result_array, fmt="%d", delimiter=",")

    schema = StructType(
        [
            StructField("label", IntegerType(), True),
            StructField("pixels", ArrayType(IntegerType()), True),
        ]
    )

    spark = SparkSession.builder.appName("Input Dataframe").getOrCreate()
    rdd_rows = labels_and_arrays.map(
        lambda x: Row(label=x[0], pixels=list(x[1].flatten().astype(int).tolist()))
    )

    # Explode the pixels array into separate columns
    df = spark.createDataFrame(rdd_rows, schema=schema)
    df = df.select(
        "label",
        *[
            col("pixels")[i].cast(IntegerType()).alias(f"pixel_{i+1}")
            for i in range(len(df.select("pixels").first()[0]))
        ],
    )

    # input_data = df.toPandas().values

    # model = load_model("model.keras")
    # predictions = model.predict(input_data)

    df.write.csv(output_folder, header=False, mode="overwrite")

    # Stop the SparkContext when done
    sc.stop()
