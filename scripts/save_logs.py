import os


def find_csvs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        csv_file = None
        yaml_file = None

        for filename in filenames:
            if filename.endswith("csv"):
                joined_path = os.path.join(dirpath, filename)
                if "task-" in joined_path:
                    yield joined_path


def main(root_dir: str):
    for csv in find_csvs(root_dir):
        print(f"Processing {csv}")
        with open(csv, "r+") as file:
            num_keys = 0
            file.seek(0)
            lines = file.readlines()
            file.seek(0)
            for index, line in enumerate(lines):
                if index == 0:
                    num_keys = line.count(",")
                    # Count how many commas are in the line
                    file.write(line)
                    continue
                commas_to_add = num_keys - line.count(",")
                if commas_to_add == 14:
                    file.write(line.strip() + "," * commas_to_add + "\n")
                elif commas_to_add == 9:
                    file.write("," * commas_to_add + line.strip() + "\n")
                elif commas_to_add != 0:
                    print(f"Unknown number of commas to add: {commas_to_add}")


if __name__ == "__main__":
    main(
        "C:\\Users\\Sander\\Documents\\Courses\\2022-2023\\Afstuderen\\Logs\\embedding-type\\embedding-type-logs"
    )
