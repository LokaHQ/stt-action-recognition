import os

def remove_npy_files(directory):
    count = 0

    # Loop through the files in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file has a .npy extension
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                os.remove(file_path)  # Remove the file
                count += 1

    return count

if __name__ == "__main__":
    directory = "dataset_cleaned/dataset/examples"  # Specify the path to the files
    removed_files_count = remove_npy_files(directory)
    print(f"{removed_files_count} .npy files were removed.")