import os


def rename_files(directory: str):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "progress-filled-bayes.csv":
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, "progress-fill-bayes.csv")
                os.rename(old_path, new_path)
                print(f"Renamed: {file} to progress-filled-bayes.csv")


def main(directory: str):
    if os.path.exists(directory):
        rename_files(directory)
        print("File renaming completed.")
    else:
        print("Directory not found. Please enter a valid directory path.")


if __name__ == "__main__":
    main(
        "D:\\Afstuderen\\embedding-consumption\\embedding-fifty-logs\\logs\\meta\\obs-0\\rnn-0\\updates-0.05"
    )
