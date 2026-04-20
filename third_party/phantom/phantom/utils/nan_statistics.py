import numpy as np
import matplotlib.pyplot as plt


def inspect_nan_distribution(npy_path: str):
    """Check the distribution of NaNs in the depth map"""
    depth_data = np.load(npy_path)

    # Get the first frame
    frame = depth_data[0]
    nan_mask = np.isnan(frame)

    print(f"NaN Statistics:")
    print(f"  Total pixels: {frame.size}")
    print(f"  NaN pixels: {np.sum(nan_mask)}")
    print(f"  NaN ratio: {100 * np.sum(nan_mask) / frame.size:.2f}%")

    # Visualization
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.imshow(frame, cmap='jet')
    plt.colorbar()
    plt.title("Depth Map")

    plt.subplot(122)
    plt.imshow(nan_mask, cmap='gray')
    plt.colorbar()
    plt.title("NaN Distribution")

    plt.tight_layout()
    plt.savefig("data/raw/hand_dataset/pick_place/nan_distribution.png")
    print(f"Visualization saved to data/raw/hand_dataset/pick_place/nan_distribution.png")


if __name__ == "__main__":
    inspect_nan_distribution("data/raw/hand_dataset/pick_place/0_human_depth.npy")
