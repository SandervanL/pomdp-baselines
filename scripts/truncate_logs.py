import os
from typing import Optional


def truncate_file(file_path: str, lines_to_keep: int) -> None:
    """Truncate the given file to keep only the first 'lines_to_keep' lines."""
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        with open(file_path, "w") as file:
            file.writelines(lines[:lines_to_keep])

        print(f"Truncated {file_path} to {lines_to_keep} lines.")
    except Exception as e:
        print(f"Error truncating {file_path}: {e}")


def truncate_logs(directory_path: str, lines_to_keep: Optional[int] = 1000) -> None:
    """Traverse the directory tree and truncate all '*.log' files to 'lines_to_keep' lines."""
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith(".log"):
                file_path = os.path.join(root, file_name)
                truncate_file(file_path, lines_to_keep)


def main() -> None:
    """Main function to execute the log truncation."""
    # Specify the directory path to start the traversal
    # directory_path = "D:\\Afstuderen\\baseline\\"
    directory_path = "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Logs\\embedding-model\\"

    # Specify the number of lines to keep in each log file
    lines_to_keep = 1000

    # Call the function to truncate log files
    truncate_logs(directory_path, lines_to_keep)


if __name__ == "__main__":
    main()
