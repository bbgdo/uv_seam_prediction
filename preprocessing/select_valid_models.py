import os


def clean_directory(target_folder, whitelist_file):
    base_path = os.path.dirname(os.path.abspath(__file__))

    folder_path = os.path.join(base_path, target_folder)
    whitelist_path = os.path.join(base_path, whitelist_file)

    if not os.path.exists(whitelist_path):
        print(f"Error: File {whitelist_file} is not found!")
        return

    with open(whitelist_path, 'r', encoding='utf-8') as f:
        valid_files = {line.strip() for line in f if line.strip()}

    if not os.path.exists(folder_path):
        print(f"Error: Dir {target_folder} is not found!")
        return

    print(f"Clearing: {target_folder}")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            if filename not in valid_files:
                try:
                    os.remove(file_path)
                    print(f"Removed: {filename}")
                except Exception as e:
                    print(f"Failed to remove {filename}: {e}")
            else:
                print(f"Kept: {filename}")


if __name__ == "__main__":
    TARGET_DIR = "Mesh_Files_Cleaned"
    WHITELIST = "valid_files.txt"

    clean_directory(TARGET_DIR, WHITELIST)