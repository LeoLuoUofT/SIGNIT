import cv2
import mediapipe as mp
import math


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def detect_hand_center_and_max_distance(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Process the image and detect hand landmarks
    results = hands.process(image_rgb)

    # Calculate the center of the hand and maximum distance to any landmark
    hand_center = None
    max_distance = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the average position of all landmarks (hand center)
            x_sum, y_sum = 0, 0
            for landmark in hand_landmarks.landmark:
                height, width, _ = image.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                x_sum += x
                y_sum += y

            num_landmarks = len(hand_landmarks.landmark)
            hand_center = (int(x_sum / num_landmarks), int(y_sum / num_landmarks))

            # Calculate the maximum distance to any landmark
            for landmark in hand_landmarks.landmark:
                landmark_point = (int(landmark.x * width), int(landmark.y * height))
                distance = calculate_distance(hand_center, landmark_point)
                if distance > max_distance:
                    max_distance = distance

                # Draw landmarks on the image
                cv2.circle(image, landmark_point, 5, (255, 0, 0), -1)

            # Draw the center on the image
            cv2.circle(image, hand_center, 10, (0, 255, 0), -1)

    # Save the result to a file
    cv2.imwrite(output_path, image)

    return hand_center, max_distance


def crop_hands(image_path, hand_center, max_distance, output_path):
    # Calculate the bounding box dimensions
    image = cv2.imread(image_path)
    x_min = max(0, int(hand_center[0] - 2 * max_distance))
    y_min = max(0, int(hand_center[1] - 2 * max_distance))
    x_max = min(int(image.shape[1]), int(hand_center[0] + 2 * max_distance))
    y_max = min(int(image.shape[0]), int(hand_center[1] + 2 * max_distance))

    # Crop the image around the hands
    cropped_image = image[y_min:y_max, x_min:x_max]
    target_size = (128, 128)
    resized_image = cv2.resize(cropped_image, target_size)

    # Save the cropped image to a file
    cv2.imwrite(output_path, resized_image)


if __name__ == "__main__":
    # Replace 'path/to/your/image.jpg' with the actual path to your image file
    image_path = "data/asl_alphabet_test/F_test.jpg"
    image_path = "Leo_testing_folder/WIN_20231201_20_44_55_Pro.jpg"
    output_image_path = "Leo_testing_folder/wcropped.jpg"
    # convert_to_grayscale(image_path, output_image_path)
    hand_center, max_distance = detect_hand_center_and_max_distance(
        image_path, "result_image.jpg"
    )
    crop_hands(image_path, hand_center, max_distance, output_image_path)
