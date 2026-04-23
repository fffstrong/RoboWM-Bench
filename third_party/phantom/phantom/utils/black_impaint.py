import cv2
import argparse
import os

parser = argparse.ArgumentParser(description="Cover half of the video to keep only one hand.")
parser.add_argument("--input_video", type=str, required=True, help="Input video path")
args = parser.parse_args()

input_video = args.input_video
base_name, ext = os.path.splitext(input_video)
output_video_left_masked = f"{base_name}_left_black{ext}"
output_video_right_masked = f"{base_name}_right_black{ext}"

# Open the video
cap = cv2.VideoCapture(input_video)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video file, please check the path.")
    exit()

# Get the original width, height, and FPS of the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Assuming the video is 1280x720, calculate the width boundary for the left half (1280 / 2 = 640)
half_width = width // 2

# Set the codec and parameters for the output video (mp4v is suitable for .mp4 files)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_left = cv2.VideoWriter(output_video_left_masked, fourcc, fps, (width, height))
out_right = cv2.VideoWriter(output_video_right_masked, fourcc, fps, (width, height))

print("Starting to process the video, please wait...")

while True:
    ret, frame = cap.read()

    # If no frame is read, it means the video has ended
    if not ret:
        break

    # Create copies to avoid overwriting the original frame for both modifications
    frame_left_masked = frame.copy()
    frame_right_masked = frame.copy()

    # Mask left hand (covers left half)
    frame_left_masked[:, :half_width] = (0, 0, 0)
    
    # Mask right hand (covers right half)
    frame_right_masked[:, half_width:] = (0, 0, 0)

    # Write the modified frames back
    out_left.write(frame_left_masked)
    out_right.write(frame_right_masked)

# Release all resources
cap.release()
out_left.release()
out_right.release()
print(f"Processing complete! Videos saved as:\n- {output_video_left_masked}\n- {output_video_right_masked}")
