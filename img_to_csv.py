from pyspark import SparkContext, SparkConf
import numpy as np
import cv2
import os
import mediapipe as mp
import math


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def hand_detection(pair):
    # Extract the file name and binary content from the RDD pair
    file_name, binary_content = pair

    image_array = np.frombuffer(binary_content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Specify the output path to save the image/csv

    output_path = os.path.join(
        output_folder, f"{file_name.split('/')[-1].split('.')[0]}.csv"
    )

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

    coutput = crop_hands(image, hand_center, max_distance)
    flattened_pixels = np.reshape(coutput, [1, coutput.size])

    # save as csv file
    np.savetxt(output_path, flattened_pixels, fmt='%d', delimiter=",")

    return flattened_pixels


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
    target_size = (128, 128)
    resized_image = cv2.resize(cropped_image, target_size)

    # Save the cropped image to a file
    # cv2.imwrite(output_path, resized_image)
    return resized_image


if __name__ == "__main__":
    conf = SparkConf().setAppName("BinaryFile")
    sc = SparkContext(conf=conf)

    # Specify the path to the folder containing binary files
    image_path = "Leo_testing_folder/imgtest"
    output_folder = "Leo_testing_folder/output_images"

    # Read binary files into an RDD
    binary_rdd = sc.binaryFiles(image_path)

    binary_rdd.map(hand_detection)

    # Stop the SparkContext when done
    sc.stop()
