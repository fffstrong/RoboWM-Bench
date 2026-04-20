import numpy as np
import argparse
import sys
import plotly.graph_objects as go

def inspect_and_visualize(file_path):
    try:
        # Load npz file
        data = np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"❌ Error: File not found '{file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot read file: {e}")
        sys.exit(1)

    print(f"📂 Analyzing file: {file_path}")
    print("=" * 60)

    # Get all keys
    keys = list(data.keys())
    print(f"Contains a total of {len(keys)} arrays: {keys}\n")

    # --- Part 1: Print statistical information ---
    for i, key in enumerate(keys):
        arr = data[key]
        
        print(f"🔹 [{i+1}] Key: '{key}'")
        print(f"   • Shape (Shape): {arr.shape}")
        print(f"   • NDim  (Dimensions): {arr.ndim}")
        print(f"   • Dtype (Type): {arr.dtype}")
        print(f"   • Size  (Elements): {arr.size}")

        if np.issubdtype(arr.dtype, np.number):
            if arr.size > 0:
                # Count the number of NaNs
                nan_count = np.isnan(arr).sum()
                nan_ratio = (nan_count / arr.size) * 100
                
                # Count the number of 0s
                zero_count = np.sum(arr == 0)
                zero_ratio = (zero_count / arr.size) * 100

                print(f"   • Stats (Statistics):")
                # If all are NaN, prevent error
                if nan_count == arr.size:
                    print("       Warning: The entire array is NaN!")
                else:
                    print(f"       Min: {np.nanmin(arr)}")
                    print(f"       Max: {np.nanmax(arr)}")
                    print(f"       Mean: {np.nanmean(arr):.4f}")
                    print(f"       Median: {np.nanmedian(arr)}")
                    print(f"       Std: {np.nanstd(arr):.4f}")
                
                print(f"       Zero count: {zero_count} (Ratio: {zero_ratio:.2f}%)")
                print(f"       NaN/Invalid count: {nan_count} (Ratio: {nan_ratio:.2f}%)")
            else:
                print("   • Stats (Statistics): [Empty array]")
        else:
            print("   • Stats (Statistics): Non-numeric data (Skipping statistics)")

        if arr.size > 0:
            preview_vals = arr.flatten()[:5]
            preview_str = ", ".join([str(x) for x in preview_vals])
            if arr.size > 5:
                preview_str += ", ..."
            print(f"   • Data  (Preview): [{preview_str}]")
        else:
            print("   • Data  (Preview): [Empty array]")

        print("-" * 60)
    
    if 'depths' in keys and data['depths'].ndim >= 3 and data['depths'].shape[0] > 0:
        first_frame = data['depths'][0]  # Extract the first frame
        save_path = file_path.replace('.npz', '_frame0.npz') # Generate new filename
        np.savez(save_path, depths=first_frame) # Save as new .npz file
        print(f"💾 Extracted first frame depth map (Shape: {first_frame.shape}) and saved to:\n   {save_path}")
        print("-" * 60)
    
    print("✅ Analysis complete.\n")


if __name__ == "__main__":
    filename = "data/processed/epic/0/smoothing_processor/smoothed_actions_right_shoulders.npz"
    inspect_and_visualize(filename)