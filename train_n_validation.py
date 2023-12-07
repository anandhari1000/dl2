import os
import random
import shutil

# Define directories


main_dir = r"C:/Users/Admin/Downloads/Dataset_Celebrities"
train_dir = r"C:/Users/Admin/Downloads/Dataset_Celebrities/cropped/lionel_messi"
validation_dir = r"C:/Users/Admin/Downloads/Dataset_Celebrities/cropped/roger_federer"


# Define validation set percentage
validation_percentage = 0.2

# Create training and validation directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Iterate through each celebrity folder
for celebrity_folder in os.listdir(main_dir):
    celebrity_path = os.path.join(main_dir, celebrity_folder)

    # List all files in the celebrity folder
    files = []
    for root, dirs, filenames in os.walk(celebrity_path):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    # Shuffle the files randomly
    random.shuffle(files)

    # Determine the split index
    split_index = int(len(files) * (1 - validation_percentage))

    # Divide files into training and validation sets
    train_files = files[:split_index]
    validation_files = files[split_index:]

    # Move training files to the training directory
    for file in train_files:
        new_path = os.path.join(train_dir, celebrity_folder, os.path.relpath(file, celebrity_path))
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.move(file, new_path)

    print(f"Moved {len(train_files)} files to the training set for {celebrity_folder}.")

    # Move validation files to the validation directory
    for file in validation_files:
        new_path = os.path.join(validation_dir, celebrity_folder, os.path.relpath(file, celebrity_path))
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.move(file, new_path)

    print(f"Moved {len(validation_files)} files to the validation set for {celebrity_folder}.")
