import os
import shutil


def remove_files_in_folder_a(folder_a, folder_b):
    # Iterate through subfolders in folder A
    for root, dirs, files in os.walk(folder_a):
        for file in files:
            file_path_a = os.path.join(root, file)

            # Check if the file in folder A exists in folder B
            file_path_b = os.path.join(folder_b, file)
            if os.path.exists(file_path_b):
                # Remove the file in folder A
                os.remove(file_path_a)
                print(f"Removed: {file_path_a}")


# Specify the paths to folders A and B
folder_a_path = "data\\asl_alphabet_train_wo_test"
folder_b_path = "data\\testing_images"

# Call the function to remove files in folder A that exist in folder B
remove_files_in_folder_a(folder_a_path, folder_b_path)
