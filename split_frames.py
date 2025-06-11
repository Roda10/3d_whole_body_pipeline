# File: split_ehf_frames.py

import os
from pathlib import Path

def create_frame_splits():
    """
    Scans the EHF data directory, finds all frame IDs, sorts them,
    and splits them into two files: frames_gpu0.txt and frames_gpu1.txt.
    """
    print("--- Starting frame splitting process ---")
    
    try:
        # Define the path to your EHF dataset relative to the script's location
        ehf_path = Path("data/EHF")
        
        if not ehf_path.exists() or not ehf_path.is_dir():
            print(f"Error: EHF data directory not found at '{ehf_path.resolve()}'")
            print("Please make sure you are running this script from the project root directory.")
            return

        # Find all image files and extract the frame ID
        frames = []
        for img_file in sorted(ehf_path.glob("*_img.jpg")):
            frame_id = img_file.stem.replace("_img", "")
            frames.append(frame_id)

        if not frames:
            print(f"Error: No frames found in '{ehf_path.resolve()}'.")
            print("Check that the EHF data is correctly populated.")
            return
            
        print(f"Found {len(frames)} total EHF frames.")

        # Split the list into two halves
        midpoint = len(frames) // 2
        frames_for_gpu0 = frames[:midpoint]
        frames_for_gpu1 = frames[midpoint:]

        # Write the first file
        with open("frames_gpu0.txt", "w") as f:
            f.write("\n".join(frames_for_gpu0))
        print(f"Created frames_gpu0.txt with {len(frames_for_gpu0)} frames.")

        # Write the second file
        with open("frames_gpu1.txt", "w") as f:
            f.write("\n".join(frames_for_gpu1))
        print(f"Created frames_gpu1.txt with {len(frames_for_gpu1)} frames.")

        print("--- Frame splitting complete ---")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    create_frame_splits()