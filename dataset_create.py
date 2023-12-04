import os
import shutil

# Source directory containing subfolders with images
source_directory = "data/asl_alphabet_train"

# Destination directory to store selected images
destination_directory = "data/training_images"

# Ensure the destination directory exists, create if not
os.makedirs(destination_directory, exist_ok=True)


# Function to copy every 100th image from each subfolder
def copy_every_100th_image(subfolder_path):
    files = os.listdir(subfolder_path)
    selected_files = files[::5]  # Get every 100th file
    for file in selected_files:
        source_path = os.path.join(subfolder_path, file)
        destination_path = os.path.join(destination_directory, file)
        shutil.copy(source_path, destination_path)


# Iterate through subfolders in the source directory
# for subfolder in os.listdir(source_directory):
#     subfolder_path = os.path.join(source_directory, subfolder)
#     if os.path.isdir(subfolder_path):
#         copy_every_100th_image(subfolder_path)

subfolder_path = os.path.join(source_directory)
if os.path.isdir(subfolder_path):
    copy_every_100th_image(subfolder_path)

print("Image copying completed.")
