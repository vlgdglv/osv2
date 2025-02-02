import os
import shutil
import argparse

def merge_files_by_prefix(source_dir, file_prefix, dest_file):
    """
    Merge chunked files of a specific file (by prefix) into a single large file.

    :param source_dir: Path to the directory containing chunked files.
    :param file_prefix: Prefix of the chunked files to be merged (e.g., 'file').
    :param dest_file: Path to the final merged file.
    """
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return

    # Filter and sort chunk files based on the prefix and part number
    chunk_files = sorted(
        [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.startswith(file_prefix) and '.part' in f],
        key=lambda x: int(x.split('.part')[-1])  # Sort by chunk number
    )

    if not chunk_files:
        print(f"No chunk files found with prefix '{file_prefix}' in directory: {source_dir}")
        return

    print(f"Found {len(chunk_files)} chunk files for prefix '{file_prefix}'. Starting merge into {dest_file}.")

    # Merge chunks into the destination file
    with open(dest_file, 'wb') as dest:
        for chunk_file in chunk_files:
            print(f"Merging {chunk_file}...")
            with open(chunk_file, 'rb') as src:
                shutil.copyfileobj(src, dest)

    print(f"Merge completed. Merged file saved at: {dest_file}")

    # Optional: Clean up chunk files
    for chunk_file in chunk_files:
        os.remove(chunk_file)
        print(f"Deleted chunk file: {chunk_file}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_directory', type=str, default=None)
    parser.add_argument("--file_prefix", type=str, required=False)
    parser.add_argument("--destination_file", type=str, required=False)
    args = parser.parse_args()


    # source_directory = "bs_SimANS_36k_8/"  # Replace with your chunked files directory
    # file_prefix = "SPTAGFullList.bin"  # Replace with the specific prefix of your file
    # destination_file = "bs_SimANS_36k_8/SPTAGFullList.bin.merged"  # Replace with your desired merged file path

    merge_files_by_prefix(args.source_directory, args.file_prefix, args.destination_file)
