import numpy as np
import os

def compare_npz_files(file1_path, file2_path):
    print(f"📁 Loading file 1: {file1_path}")
    print(f"📁 Loading file 2: {file2_path}")
    print("-" * 60)
    
    # Check if files exist
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        print("❌ Error: Specified file(s) not found, please check the paths.")
        return

    try:
        data1 = np.load(file1_path)
        data2 = np.load(file2_path)
        
        keys = ['ee_pts', 'ee_oris', 'ee_widths']
        
        # Check if all required keys exist
        for key in keys:
            if key not in data1 or key not in data2:
                print(f"❌ Error: Missing key array '{key}'.")
                return
                
        # Get the minimum number of frames to align trajectories (in case sequence lengths differ)
        num_frames1 = data1['ee_pts'].shape[0]
        num_frames2 = data2['ee_pts'].shape[0]
        min_frames = min(num_frames1, num_frames2)
        
        if num_frames1 != num_frames2:
            print(f"⚠️ Warning: Number of frames in the two files is inconsistent (File1: {num_frames1}, File2: {num_frames2}).")
            print(f"Will truncate to the first {min_frames} frames for comparison.")
            print("-" * 60)

        # 1. Compare ee_pts (Shape: N, 3) - Calculate Euclidean distance
        pts1 = data1['ee_pts'][:min_frames]
        pts2 = data2['ee_pts'][:min_frames]
        # np.linalg.norm(..., axis=1) calculates the L2 norm for each row
        pts_diff = np.linalg.norm(pts1 - pts2, axis=1) 
        
        # 2. Compare ee_oris (Shape: N, 3, 3) - Calculate Frobenius norm
        oris1 = data1['ee_oris'][:min_frames]
        oris2 = data2['ee_oris'][:min_frames]
        # Calculate the Frobenius norm of the difference for each 3x3 matrix
        oris_diff = np.linalg.norm(oris1 - oris2, axis=(1, 2))
        
        # 3. Compare ee_widths (Shape: N,) - Calculate absolute error
        widths1 = data1['ee_widths'][:min_frames]
        widths2 = data2['ee_widths'][:min_frames]
        widths_diff = np.abs(widths1 - widths2)

        # Print statistical results
        print("📊 Error Statistics (Calculated frame-by-frame based on aligned frames)")
        print("=" * 60)
        
        metrics = [
            ("ee_pts (Euclidean Distance L2-Norm)", pts_diff),
            ("ee_oris (Matrix Frobenius Norm)", oris_diff),
            ("ee_widths (Absolute Error L1-Norm)", widths_diff)
        ]
        
        for name, diff_array in metrics:
            print(f"🔹 {name}:")
            print(f"   • Mean (Average error): {np.mean(diff_array):.6f}")
            print(f"   • Max  (Maximum error): {np.max(diff_array):.6f} (Found at frame {np.argmax(diff_array)})")
            print(f"   • Min  (Minimum error): {np.min(diff_array):.6f} (Found at frame {np.argmin(diff_array)})")
            print(f"   • Std  (Standard deviation): {np.std(diff_array):.6f}")
            print("-" * 60)
            
        print("✅ Comparison complete.")

    except Exception as e:
        print(f"❌ Error occurred while processing files: {e}")


if __name__ == "__main__":
    # Replace with your actual file paths
    file_0 = "data/processed_with_depth/pick_and_place/0/smoothing_processor/smoothed_actions_left_single_arm.npz"
    file_1 = "data/processed_with_depth/pick_and_place/1/smoothing_processor/smoothed_actions_left_single_arm.npz"

    compare_npz_files(file_0, file_1)