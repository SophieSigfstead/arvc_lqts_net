import os
import subprocess

# Define the base directory where the ARVC, control, and LQTS subdirectories are located
base_dir = "./"

# Function to process each subdirectory using extract_ecg_xml.py
def process_directory(source_dir):
    dest_dir = source_dir + "_csv"
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Construct the subprocess command
    import sys

    command = [
        "python3",
        "extract_ecg_xml.py",
        "--source_dir", source_dir,
        "--dest_dir", dest_dir,
        "--samples", "2500"
    ]

    # Run the command and capture the output
    try:
        subprocess.run(command, check=True)
        print(f"Processed: {source_dir} -> {dest_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to process {source_dir}: {e}")

# Walk through the entire directory structure starting from base_dir
for root, dirs, files in os.walk(base_dir):
    # Skip the base directory itself
    if root == base_dir:
        continue

    # Process the current directory if it contains XML files
    if any(file.endswith(".xml") for file in files):
        process_directory(root)
