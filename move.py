import os
import shutil

# Define the base directory
base_dir = "adversarial_evaluation/adversarial_examples"

# Iterate through each folder in the base directory
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):  # Check if it's a folder
        gamma_path = os.path.join(folder_path, "gamma")
        if os.path.exists(gamma_path) and os.path.isdir(gamma_path):
            target_dir = os.path.join(gamma_path, "50")
            
            # Move files that are not folders into the "50" folder
            files_moved = 0
            for item in os.listdir(gamma_path):
                item_path = os.path.join(gamma_path, item)
                if os.path.isfile(item_path):  # Check if it's a file
                    shutil.move(item_path, os.path.join(target_dir, item))
                    files_moved += 1
            
            # Print the number of files moved
            print(f"Moved {files_moved} files from {gamma_path} to {target_dir}.")