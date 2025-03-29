import cv2
import os
import random

# --- Configuration ---
video_path = 'Videos/high_slot.mov'  # <<<--- CHANGE THIS to your video file path
output_dir = 'data'
BLOCK_SIZE = 100
# --- End Configuration ---

def extract_random_frames(video_path, output_dir, block_size=100):
    """
    Extracts one random frame from every block_size frames of a video.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        block_size (int): The size of the frame interval (e.g., 100).
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    frame_count = 0
    saved_count = 0
    block_frame_to_save = -1 # Frame number to save within the current block

    print(f"Processing video: {video_path}")
    print(f"Extracting one random frame every {block_size} frames...")

    while True:
        # Read the next frame
        ret, frame = cap.read()

        # If frame reading was not successful (end of video), break the loop
        if not ret:
            break

        frame_count += 1

        # Check if we are at the beginning of a new block
        # (frame_count - 1) makes frame 1 the start of the first block (index 0)
        if (frame_count - 1) % block_size == 0:
            # Choose a random frame *offset* within this block (0 to block_size - 1)
            random_offset = random.randint(0, block_size - 1)
            # Calculate the absolute frame number to save for this block
            # current_block_start_frame = frame_count
            block_frame_to_save = frame_count + random_offset
            # print(f"Block starting at {frame_count}. Will save frame {block_frame_to_save} from this block.") # Debugging line

        # Check if the current frame is the randomly selected one for this block
        if frame_count == block_frame_to_save:
            # Construct the output filename (padded with zeros for sorting)
            output_filename = f"frame_{frame_count:06d}.png" # Using PNG for lossless, change to .jpg if needed
            output_path = os.path.join(output_dir, output_filename)

            # Save the frame
            cv2.imwrite(output_path, frame)
            saved_count += 1
            print(f"Saved frame: {frame_count} -> {output_path}")
            # Reset block_frame_to_save to ensure we don't save again until the next block
            block_frame_to_save = -1

    # Release the video capture object
    cap.release()
    # cv2.destroyAllWindows() # Not strictly necessary as no windows are shown

    print("\n--------------------")
    print("Processing finished.")
    print(f"Total frames processed: {frame_count}")
    print(f"Total frames saved: {saved_count}")
    print(f"Saved frames are in directory: {output_dir}")
    print("--------------------")

# --- Run the extraction ---
if __name__ == "__main__":
    # Make sure to change 'your_video.mp4' above before running
    if video_path == 'your_video.mp4':
         print("ERROR: Please change the 'video_path' variable in the script to your actual video file.")
    elif not os.path.exists(video_path):
        print(f"ERROR: The video file specified does not exist: {video_path}")
    else:
        extract_random_frames(video_path, output_dir, BLOCK_SIZE)