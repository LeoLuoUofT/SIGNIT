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


def hand_detection(pair, intermediateimages=False):
    # Extract the file name and binary content from the RDD pair
    file_name, binary_content = pair

    image_array = np.frombuffer(binary_content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
                if intermediateimages:
                    cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

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

    # Specify the output path to save the image/csv
    if intermediateimages:
        output_path = os.path.join(
            output_folder, f"{file_name.split('/')[-1].split('.')[0]}.jpg"
        )

        # Draw the center on the image and save the result to a file
        cv2.circle(image, hand_center, 10, (0, 255, 0), -1)
        cv2.imwrite(output_path, image)

    coutput = crop_hands(image, hand_center, max_distance)
    flattened_pixels = np.reshape(coutput, [1, coutput.size])

    # save as csv file
    # np.savetxt(output_folder, flattened_pixels, fmt='%d', delimiter=",")
    # print(flattened_pixels.shape)

    return label_re(flattened_pixels, file_name)


def label_re(flattened_pixels, file_name, labels=False):
    if labels:
        # write label for training data
        label = os.path.basename(file_name).split(".")[0].split("_")[0]
        if label in alphabet_mapping:
            label = alphabet_mapping[label]
        else:
            label = 27
        return label, flattened_pixels
    else:
        sequence = os.path.basename(file_name).split(".")[0]
        return int(sequence), flattened_pixels


def raw_data(pair):
    file_name, binary_content = pair

    image_array = np.frombuffer(binary_content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # label = re.split(r"[0-9_]+", file_name.split("/")[12])[0]

    target_size = (100, 100)
    resized_image = cv2.resize(image, target_size)
    recolored = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    coutput = recolored
    flattened_pixels = np.reshape(coutput, [1, coutput.size])

    # if label in alphabet_mapping:
    #     label = alphabet_mapping[label]
    # else:
    #     label = 27

    return label_re(flattened_pixels, file_name)


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


def write_dataset_csv(rdd):
    schema = StructType(
        [
            StructField("label", IntegerType(), True),
            StructField("pixels", ArrayType(IntegerType()), True),
        ]
    )

    spark = SparkSession.builder.appName("Input Dataframe").getOrCreate()
    df = spark.createDataFrame(rdd, schema=schema)
    df = df.select(
        "label",
        *[
            col("pixels")[i].cast(IntegerType()).alias(f"pixel_{i}")
            for i in range(len(df.select("pixels").first()[0]))
        ],
    )

    df.write.csv(output_folder, header=False, mode="overwrite")


def output_predicts(rdd):
    schema = StructType(
        [
            StructField("sequence", IntegerType(), True),
            StructField("pixels", ArrayType(IntegerType()), True),
        ]
    )

    spark = SparkSession.builder.appName("Input Dataframe").getOrCreate()
    df = spark.createDataFrame(rdd, schema=schema)
    df = df.select(
        "sequence",
        *[
            col("pixels")[i].cast(IntegerType()).alias(f"pixel_{i}")
            for i in range(len(df.select("pixels").first()[0]))
        ],
    )

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
    df_pandas = df.toPandas()

    input_data = np.array(df_pandas.iloc[:, 1:]).reshape(df.count(), 100, 100, 1)
    model = load_model("model_small.keras")
    predictions = model.predict(input_data)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_class_names = [class_names[i] for i in predicted_indices]

    # in case sequence wasn't preserved runs faster without it
    # predictions_df = spark.createDataFrame(
    #     zip(df_pandas["sequence"].tolist(), predicted_class_names),
    #     ["sequence", "predictions"],
    # ).sort("sequence", ascending=True)

    return predicted_class_names


if __name__ == "__main__":
    from pyspark import SparkConf, SparkContext
    from pyspark.streaming import StreamingContext
    from pyspark.sql import Row, SparkSession

    conf = SparkConf().setAppName("Folder ")
    sc = SparkContext(conf=conf)

    if len(sys.argv) != 2:
        print("Usage: spark-submit script.py <input_path>")
        sys.exit(1)

    # Specify the path to the folder containing binary files
    image_path = sys.argv[1]
    output_folder = "byproducts/intermediate"

    binary_rdd = sc.binaryFiles(image_path)

    # rdd to image type
    img_arrays = binary_rdd.map(hand_detection)
    rdd_rows = img_arrays.map(
        lambda x: Row(sequence=x[0], pixels=list(x[1].flatten().astype(int).tolist())),
    )

    # create dataset
    # write_dataset_csv(img_arrays)

    prd = output_predicts(rdd_rows)
    print(prd)
    print("Saving to byproducts/output.txt")
    prd_str = "".join(prd)

    # Save the string to a file
    file_path = "byproducts/output.txt"
    with open(file_path, "a") as file:
        file.write("\n"+prd_str)

    # Stop the SparkContext when done
    sc.stop()
