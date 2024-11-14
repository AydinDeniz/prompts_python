import os
import time
import shutil
from datetime import datetime

# Define the source and destination directories
source_directory = "path/to/source/directory"  # Replace with your source directory
destination_directory = "path/to/destination/directory"  # Replace with your destination directory
log_file = "file_operations_log.txt"

# Function to write logs
def write_log(message):
    with open(log_file, "a") as log:
        log.write(f"{datetime.now()} - {message}\n")

# Function to monitor, rename, and move files
def monitor_and_move_files():
    processed_files = set()  # Keep track of files that have been processed

    while True:
        for filename in os.listdir(source_directory):
            source_path = os.path.join(source_directory, filename)

            # Skip if it’s already processed or if it’s a directory
            if filename in processed_files or os.path.isdir(source_path):
                continue

            # Generate a new filename with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            new_filename = f"{timestamp}_{filename}"
            destination_path = os.path.join(destination_directory, new_filename)

            # Move the file to the destination directory
            shutil.move(source_path, destination_path)
            processed_files.add(filename)  # Mark as processed

            # Log the operation
            write_log(f"Moved file {filename} to {destination_path}")

        # Wait for 5 minutes before checking again
        time.sleep(300)  # 300 seconds = 5 minutes

# Start monitoring
monitor_and_move_files()
