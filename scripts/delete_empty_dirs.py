import os
import shutil


def delete_empty_directories(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            current_dir = os.path.join(root, dir_name)
            if not os.listdir(current_dir):  # Check if the directory is empty
                process_csv_path = os.path.join(current_dir, "progress.csv")
                if (
                    os.path.exists(process_csv_path)
                    and os.path.getsize(process_csv_path) == 0
                ):
                    print(f"Deleting empty directory: {current_dir}")
                    shutil.rmtree(current_dir)


def main():
    target_directory = "D:\\Afstuderen\\uncertainty"

    if os.path.exists(target_directory):
        delete_empty_directories(target_directory)
        print("Empty directories with an empty 'progress.csv' file deleted.")
    else:
        print("Invalid directory path. Please provide a valid directory path.")


if __name__ == "__main__":
    main()
