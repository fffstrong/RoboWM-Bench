import cv2

# Set input and output filenames (please modify according to your actual filenames)
input_video = 'data/raw/hand_dataset/box_bi/2/2_human_rgb.mp4'  # Replace with your input video path
output_video = 'data/raw/hand_dataset/box_bi/2/2_human_right_black_rgb.mp4'  # Replace with your output video path

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
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print("Starting to process the video, please wait...")

while True:
    ret, frame = cap.read()

    # If no frame is read, it means the video has ended
    if not ret:
        break

    # Core code: Set the entire height [:] and half the width [half_width:] to pure black (B=0, G=0, R=0)
    frame[:, half_width:] = (0, 0, 0)  # half_width: is for the left hand, if you want to cover the right hand, change it to frame[:, half_width:] = (0, 0, 0)

    # Write the modified frame to the output video
    out.write(frame)

# Release all resources
cap.release()
out.release()
print(f"Processing complete! Video saved as: {output_video}")
