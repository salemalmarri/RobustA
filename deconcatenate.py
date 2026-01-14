import numpy as np
import os

# Define the source directory containing .npy files and the output directory
source_dir = "./XD-Violence"
output_dir = "./XD-Violence-10"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all .npy files in the source directory
for file in os.listdir(source_dir):
    if file.endswith(".npy"):
        print(f"Processing file: {file}")  # Print the file name before loading
        filepath = os.path.join(source_dir, file)
        data = np.load(filepath)  # Load the .npy file

        # Extract the video name from the filename (e.g., `v=xxxx_#xx-xx`)
        video_name = file.split("_#")[0]  # Adjust as per your naming convention
        print(f"Video name: {video_name}")

        # Ensure the shape is compatible (n, 10, 2048)
        if len(data.shape) == 3 and data.shape[1] == 10:
            base_name = os.path.splitext(file)[0]  # Remove extension from the filename

            # Iterate over the 10 crops
            for crop_idx in range(data.shape[1]):
                # Save each crop
                crop_data = data[:, crop_idx, :]  # Extract data for this crop
                new_filename = f"{base_name}__{crop_idx}_rgb.npy"
                save_path = os.path.join(output_dir, new_filename)
                np.save(save_path, crop_data)

        else:
            print(f"Skipping {file}: Shape is not (n, 10, 2048).")

print("Deconcatenation and renaming completed.")
