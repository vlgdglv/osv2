import os
import shutil
import argparse

def split_and_move_file(source_file, dest_dir, chunk_size, skip_chunks=0):
    """
    Split a large file into chunks and move to destination directory.

    :param source_file: Path to the source file.
    :param dest_dir: Path to the destination directory (should exist or be creatable).
    :param chunk_size: Size of each chunk in bytes (e.g., 1GB = 1 * 1024 * 1024 * 1024).
    """
    if not os.path.exists(source_file):
        print(f"Source file does not exist: {source_file}")
        return

    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    file_size = os.path.getsize(source_file)
    num_chunks = (file_size + chunk_size - 1) // chunk_size

    with open(source_file, 'rb') as src:
        for i in range(num_chunks):
            if i < skip_chunks:
                _ = src.read(chunk_size)
                print(f"Skipping chunk {i+1}")
                continue
            chunk_file_name = os.path.join(dest_dir, f"{os.path.basename(source_file)}.part{i+1}")
            print(f"Processing chunk {i+1}/{num_chunks}: {chunk_file_name}")

            # Read and write chunk
            with open(chunk_file_name, 'wb') as dest:
                data = src.read(chunk_size)
                dest.write(data)

            # Verify chunk and delete from source if successful
            if os.path.exists(chunk_file_name) and os.path.getsize(chunk_file_name) == len(data):
                print(f"Chunk {i+1} successfully moved.")
            else:
                print(f"Error occurred in moving chunk {i+1}. Aborting...")
                break

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, default=None)
    parser.add_argument('--destination_dir', type=str, default=None)
    parser.add_argument("--chunk_size_gb", type=int, required=False, default=10)
    args = parser.parse_args()

    # source_file_path = "/home/aiscuser/osv2/spann/bs_SimANS_36k_8/HeadIndex/vectors.bin"  # Replace with your file path
    # destination_directory = "/datacosmos/local/User/baoht/onesparse2/marcov2/spann/bs_SimANS_36k_8/HeadIndex"  # Replace with your destination path
    chunk_size = args.chunk_size_gb * 1024 * 1024 * 1024  # Convert to bytes
    # skip_chunks = 70
    split_and_move_file(args.source_path, args.destination_dir, chunk_size)
