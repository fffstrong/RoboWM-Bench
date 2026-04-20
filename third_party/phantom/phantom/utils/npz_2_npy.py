import numpy as np
import os

def convert_npz_to_npy(npz_file_path, output_dir=None):
    """
    Extract arrays from a .npz file and save them as independent .npy files.
    
    Args:
        npz_file_path (str): The path to the .npz file.
        output_dir (str, optional): The directory to save the generated .npy files.
                                    If None, they will be saved in the same directory as the .npz file.
    """
    # Check if the file exists
    if not os.path.exists(npz_file_path):
        print(f"Error: File not found '{npz_file_path}'")
        return

    # Set the output directory
    if output_dir is None:
        output_dir = os.path.dirname(npz_file_path)
        if output_dir == "":
            output_dir = "."
            
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Load the .npz file
        print(f"Loading: {npz_file_path}")
        with np.load(npz_file_path) as data:
            # data.files contains the names of all arrays within the archive
            array_names = data.files
            print(f"Found {len(array_names)} arrays: {array_names}")

            # Iterate through and save each array
            for name in array_names:
                array_data = data[name]
                # Build the output file path, e.g., 'array_name.npy'
                output_path = os.path.join(output_dir, f"{name}.npy")
                
                # Save as .npy
                np.save(output_path, array_data)
                print(f"Saved: {output_path} (Shape: {array_data.shape})")
                
        print("Conversion complete!")

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


# ==========================================
# Usage Example
# ==========================================
if __name__ == "__main__":
    # Replace with your own .npz file path
    source_file = "data/raw/hand_dataset/pour_water/0_veo_depth.npz"

    # Replace with the folder path where you want to save the .npy files (optional)
    # If left empty, it defaults to the same directory as the source_file
    target_folder = "data/raw/hand_dataset/pour_water"

    # Execute the conversion
    convert_npz_to_npy(source_file, target_folder)