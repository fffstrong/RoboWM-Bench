import cv2
import os


def crop_center_square(input_path, output_path):
    """
    Read a 1920x1080 video and crop the center to 1080x1080
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    
    # Get original video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing: {input_path}")
    print(f"Original size: {orig_width}x{orig_height}, FPS: {fps}")

    # Target size
    target_size = 1080

    # Calculate crop coordinates
    # No cropping needed vertically (1080 -> 1080)
    # Center crop horizontally
    x_start = (orig_width - target_size) // 2  # (1920-1080)/2 = 420
    x_end = x_start + target_size              # 420 + 1080 = 1500

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_size, target_size))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Core cropping code: frame[y_start:y_end, x_start:x_end]
        # Take full height, take middle width
        cropped_frame = frame[:, x_start:x_end]
        
        out.write(cropped_frame)
        
        count += 1
        if count % 30 == 0:
            print(f"Progress: {count}/{total_frames} frames...", end='\r')

    cap.release()
    out.release()
    print(f"\nDone! Saved to: {output_path}")


if __name__ == "__main__":
    # ================= Configuration Area =================
    # Enter the path of the video you want to crop here
    # Assuming you want to crop the 1st recorded RGB video
    input_video = "output_data/1_rgb.mp4" 

    # Automatically generate output filename (e.g.: 1_rgb_cropped.mp4)
    file_name, ext = os.path.splitext(input_video)
    output_video = f"{file_name}_cropped{ext}"
    # ===========================================

    crop_center_square(input_video, output_video)