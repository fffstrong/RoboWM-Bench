import cv2
import numpy as np
import os
from pyorbbecsdk import *
from utils import frame_to_bgr_image


def save_to_mp4():
    # ================= Configuration Area =================
    output_folder = "data/raw/hand_dataset/box_bi"
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720
    FPS = 15                 # Frame rate
    TRIM_SECONDS = 3         # Time to trim (seconds)

    # Calculate the number of frames to trim
    TRIM_FRAMES = int(FPS * TRIM_SECONDS)
    # ===========================================

    os.makedirs(output_folder, exist_ok=True)

    def get_next_folder_index(base_folder):
        # Get all child items
        items = os.listdir(base_folder)
        indices = []
        for item in items:
            full_path = os.path.join(base_folder, item)
            # Check if it is a directory and the name is purely numeric
            if os.path.isdir(full_path) and item.isdigit():
                indices.append(int(item))

        # If no numeric directory exists, start from 0; otherwise take the max value + 1
        return max(indices, default=-1) + 1

    current_index = get_next_folder_index(output_folder)

    # 2. Create a new numeric folder (e.g., .../pick/3)
    save_directory = os.path.join(output_folder, str(current_index))
    os.makedirs(save_directory, exist_ok=True)

    print(f"Data will be saved to directory: {save_directory}")

    # 3. Set file paths (all under the newly created save_directory)
    color_output_file = os.path.join(save_directory, f"{current_index}_human_rgb.mp4")
    depth_output_file = os.path.join(save_directory, f"{current_index}_human_depth.mp4")
    depth_matrix_file = os.path.join(save_directory, f"{current_index}_human_depth.npy")
    first_frame_file = os.path.join(save_directory, f"{current_index}_first_frame.jpg")

    color_writer = None
    depth_writer = None

    pipeline = Pipeline()
    config = Config()

    # === New: Buffer lists for "tail trim" ===
    # We do not write directly to the file, but first store it in this Buffer
    buffer_rgb = [] 
    buffer_depth_vis = []
    # depth_matrices is used to store raw data, the logic is slightly different, just slice it at the end
    depth_matrices = [] 

    # Counter: used for "head trim"
    valid_frame_count = 0

    has_saved_first_frame = False

    try:
        device = pipeline.get_device()
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(DISPLAY_WIDTH, DISPLAY_HEIGHT, OBFormat.RGB, FPS)
        config.enable_stream(color_profile)

        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_video_stream_profile(1024, 1024, OBFormat.Y16, FPS)
        config.enable_stream(depth_profile)

        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        print("Starting recording...")
        pipeline.start(config)
        pipeline.enable_frame_sync()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        color_writer = cv2.VideoWriter(color_output_file, fourcc, FPS, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        depth_writer = cv2.VideoWriter(depth_output_file, fourcc, FPS, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        print(f"Recording started. (First {TRIM_SECONDS}s and last {TRIM_SECONDS}s will be trimmed)")

        while True:
            frames = pipeline.wait_for_frames(100)
            if not frames: continue

            frames = align_filter.process(frames)
            if not frames: continue

            frames = frames.as_frame_set()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame: continue

            # 1. Process image data
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None: continue

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                (depth_frame.get_height(), depth_frame.get_width())
            )

            # Depth map visualization processing
            MIN_DEPTH = 300
            MAX_DEPTH = 1000
            clipped_depth = np.clip(depth_data, MIN_DEPTH, MAX_DEPTH)
            depth_image = ((clipped_depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH) * 255).astype(np.uint8)
            depth_image = 255 - depth_image 
            depth_image_vis = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

            # === Real-time display (Preview) ===
            # Note: Here we display the "current frame", even if it hasn't been written yet (or is a trimmed frame at the beginning)
            # This ensures a smooth user experience
            combined_image = cv2.addWeighted(color_image, 0.5, depth_image_vis, 0.5, 0)
            cv2.imshow("Recording Viewer", combined_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # === Core logic modification: Trimming process ===

            # A. Head Trim
            # If the current frame count hasn't reached TRIM_FRAMES, skip the save step directly
            if valid_frame_count < TRIM_FRAMES:
                valid_frame_count += 1
                # We can print to the console so the user knows it hasn't started saving yet
                if valid_frame_count % 10 == 0:
                    print(f"Trimming start... {valid_frame_count}/{TRIM_FRAMES}")
                continue

            if not has_saved_first_frame:
                cv2.imwrite(first_frame_file, color_image)
                print(f"First frame saved: {first_frame_file}")
                has_saved_first_frame = True

            # B. Tail Trim - Delayed write strategy
            # 1. Add the current frame to the buffer first
            buffer_rgb.append(color_image)
            buffer_depth_vis.append(depth_image_vis)

            # 2. Only when the number of frames in the buffer exceeds the length we want to trim (TRIM_FRAMES)
            #    do we write the oldest frame.
            #    This ensures that what remains in the buffer is always the "last few seconds"
            if len(buffer_rgb) > TRIM_FRAMES:
                # Pop the oldest frame and write it
                img_to_write_rgb = buffer_rgb.pop(0)
                img_to_write_depth = buffer_depth_vis.pop(0)

                try:
                    color_writer.write(img_to_write_rgb)
                    depth_writer.write(img_to_write_depth)
                except Exception as e:
                    print("Error writing frame:", e)

            # C. Process raw depth data (.npy)
            # We first save all data that passed the "head trim", and then slice off the tail when finally saving
            depth_matrices.append(depth_data)


        # === Processing after the loop ends ===

        # At this point, the last TRIM_FRAMES frames (i.e., the last few seconds) are still in buffer_rgb
        # Because we want to trim the end, this part of the data is directly discarded and not written to VideoWriter.

        # Process .npy data
        # depth_matrices contains [valid start ---> end]
        # We need to manually slice off the last TRIM_FRAMES data points
        if len(depth_matrices) > TRIM_FRAMES:
            final_depth_data = np.array(depth_matrices[:-TRIM_FRAMES])  # Slice off the last N frames
            # 2. Convert data type to float32 (because uint16 does not support NaN and decimals)
            final_depth_data = final_depth_data.astype(np.float32)

            # 3. Convert millimeters to meters (divide by 1000)
            final_depth_data = final_depth_data / 1000.0

            # 4. Replace depth values of 0 with np.nan
            final_depth_data[final_depth_data == 0.0] = np.nan
            np.save(depth_matrix_file, final_depth_data)
            print(f"Depth matrices saved to {depth_matrix_file} (Trimmed start & end)")
        else:
            print("Warning: Recording was too short to trim end.")

    except Exception as e:
        print(f"Error during recording: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if color_writer: color_writer.release()
        if depth_writer: depth_writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Recording stopped.")


if __name__ == "__main__":
    save_to_mp4()
