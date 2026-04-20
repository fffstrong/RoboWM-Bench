import numpy as np
from pathlib import Path
import os
import cv2


def normalize_depth_for_display(data):
    """
    Normalize depth data to 0-255 range for display or saving as video.
    """
    # Handle invalid values (like NaN or Inf), replacing them with 0
    data = np.nan_to_num(data, copy=True)

    # Get max and min values of the data
    d_min = data.min()
    d_max = data.max()

    if d_max == d_min:
        # If all values are the same, return a completely black image
        return np.zeros(data.shape, dtype=np.uint8)

    # Normalization formula: (x - min) / (max - min) * 255
    norm_data = (data - d_min) / (d_max - d_min) * 255
    
    # Convert to uint8
    return norm_data.astype(np.uint8)

def save_as_mp4(data, output_path, fps=15):
    """
    Save numpy array as mp4 video
    """
    # Check data dimensions
    # Assuming data format is (Frames, Height, Width) or (Frames, Height, Width, Channels)
    if data.ndim not in [3, 4]:
        print(f"Cannot generate video: data dimension is {data.ndim}, requires (T, H, W) or (T, H, W, C)")
        return

    frames_count = data.shape[0]
    height = data.shape[1]
    width = data.shape[2]

    # Initialize video writer
    # mp4v is the encoding format for mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    # Determine if it is grayscale (ndim=3) or color (ndim=4)
    is_color = True 
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), is_color)

    print(f"Saving video to: {output_path} ...")
    print(f"Video resolution: {width}x{height}, Frames: {frames_count}")

    # Normalize data (depth maps are usually float or uint16, need to convert to 0-255)
    # Note: Here we apply uniform normalization across the entire video sequence; it can also be done per frame
    norm_data = normalize_depth_for_display(data)

    for i in range(frames_count):
        frame = norm_data[i]

        # To make the depth map clearer, a pseudocolor (Colormap) is usually applied
        # Convert grayscale to pseudocolor (e.g., JET mode: blue is near/far, red is far/near)
        color_frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        
        video_writer.write(color_frame)

    video_writer.release()
    print(f"Video saved successfully!")

def inspect_npy_file(file_path):
    """
    Read .npy file and print its shape, data type, and content.
    If it is a depth file, additionally save it as a video.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        return

    try:
        data = np.load(file_path, allow_pickle=True)

        print(f"--- File: {file_path} ---")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"NDim: {data.ndim}")
        
        print("-" * 30)
        print("--- Statistics ---")
        
        # Count the number of NaNs
        nan_count = np.isnan(data).sum()
        nan_ratio = (nan_count / data.size) * 100
        
        # Count the number of 0s
        zero_count = np.sum(data == 0)
        zero_ratio = (zero_count / data.size) * 100

        # If all are NaN, prevent error
        if nan_count == data.size:
            print("Warning: The entire array is NaN!")
        else:
            print(f"Min: {np.nanmin(data)}")
            print(f"Max: {np.nanmax(data)}")
            print(f"Mean: {np.nanmean(data):.4f}")
            print(f"Median: {np.nanmedian(data)}")
            print(f"Std: {np.nanstd(data):.4f}")
        
        print(f"Zero count: {zero_count} (Ratio: {zero_ratio:.2f}%)")
        print(f"NaN/Invalid count: {nan_count} (Ratio: {nan_ratio:.2f}%)")
        print("-" * 30)
        
        if data.ndim >= 3 and data.shape[0] > 0:
            first_frame = data[0]  # Extract the first frame
            save_path = file_path.replace('.npy', '_frame0.npz') # Generate new filename
            np.savez(save_path, depths=first_frame) # Save as .npz file
            print(f"💾 Extracted first frame depth map (Shape: {first_frame.shape}) and saved to:\n   {save_path}")
            print("-" * 30)
        
        filename = os.path.basename(file_path)
        if "depth" in filename.lower() and data.ndim >= 3:
            print(">> Depth data detected, preparing to generate video...")
            
            # Generate output filename: replace .npy with .mp4
            dir_name = os.path.dirname(file_path)
            video_name = filename.replace('.npy', '_npy_vis.mp4')
            output_video_path = os.path.join(dir_name, video_name)
            
            save_as_mp4(data, output_video_path, fps=15)
            print("-" * 30)
        # ==========================================

        print("Data content preview (first items):")
        # Flatten and print the first few items to avoid flooding the screen
        print(data.flatten()[:10]) 

    except Exception as e:
        print(f"Error occurred while processing file: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# Modify your filename here
# ==========================================
target_file = 'data/raw/hand_dataset/pour_water/0_human_depth.npy'

if __name__ == "__main__":
    inspect_npy_file(target_file)
