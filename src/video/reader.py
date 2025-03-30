# src/video/reader.py
import threading
import cv2
import queue
import time
from collections import deque # Import deque
import config # Import config

class VideoReader(threading.Thread):
    """Reads frames from a video source, puts them into a queue, and keeps a buffer of recent frames."""
    def __init__(self, video_path, frame_queue, stop_event, max_queue_size):
        threading.Thread.__init__(self, name="VideoReaderThread")
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.max_queue_size = max_queue_size
        self.cap = None
        self.daemon = True # Allows main program to exit even if this thread is blocking

        # --- Frame Buffer for Retry --- (NEW)
        self.frame_buffer = None
        if config.ENABLE_GRID_RETRY:
            buffer_size = config.GRID_RETRY_FRAME_BUFFER_SIZE
            if buffer_size > 0:
                self.frame_buffer = deque(maxlen=buffer_size)
                print(f"INFO: VideoReader initialized frame buffer with size {buffer_size}")
            else:
                print("Warning: ENABLE_GRID_RETRY is True, but GRID_RETRY_FRAME_BUFFER_SIZE is <= 0. Buffer disabled.")
        # ------------------------------

    def run(self):
        print("INFO: VideoReader thread started.")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"ERROR: VideoReader could not open video source: {self.video_path}")
            self.stop_event.set() # Signal other threads to stop
            # Put sentinel value to unblock processor if it's waiting
            try: self.frame_queue.put_nowait(None)
            except queue.Full: pass
            return

        frame_number = 0
        while not self.stop_event.is_set():
            # Prevent queue from growing indefinitely if processing is slow
            if self.frame_queue.qsize() >= self.max_queue_size:
                time.sleep(0.01) # Wait briefly if queue is full
                continue

            ret, frame = self.cap.read()
            if not ret:
                print("INFO: VideoReader reached end of video or read error.")
                break # End of video

            frame_number += 1

            # --- Add frame to buffer if enabled --- (NEW)
            if self.frame_buffer is not None:
                # Store a copy to avoid issues if frame is modified downstream before retry
                self.frame_buffer.append((frame_number, frame.copy()))
            # --------------------------------------

            try:
                # Put frame and frame number into the queue, non-blocking
                self.frame_queue.put_nowait((frame_number, frame))
            except queue.Full:
                # This case is handled by the qsize check above, but as a fallback:
                # print("Warning: Frame queue full on put attempt. Skipping frame.")
                continue
            except Exception as e:
                 print(f"ERROR in VideoReader putting frame {frame_number}: {e}")
                 break

        # Signal end of stream by putting None
        try:
            self.frame_queue.put(None, block=True, timeout=1) # Signal end, wait briefly if needed
        except queue.Full:
            print("Warning: Frame queue full when trying to signal end.")
            # Main thread will eventually time out or see stop_event

        if self.cap:
            self.cap.release()
        print("INFO: VideoReader thread finished.")

    # --- Method to get frames from buffer --- (NEW)
    def get_frames_from_buffer(self, start_frame_num, end_frame_num):
        """Retrieves frames within the specified range from the internal buffer.

        Args:
            start_frame_num (int): The starting frame number (inclusive).
            end_frame_num (int): The ending frame number (inclusive).

        Returns:
            list: A list of tuples (frame_number, frame) found in the buffer within the range.
                  Returns an empty list if buffering is disabled or no frames are found.
        """
        if self.frame_buffer is None:
            print("Warning: Attempted to get frames from buffer, but buffering is disabled.")
            return []

        # Create a temporary list from the deque for easier searching
        # This avoids issues if the deque rotates while iterating
        # Consider a lock if this becomes problematic, but simple reads should be fine
        buffered_frames = list(self.frame_buffer)

        # Filter frames within the desired range
        result_frames = [(num, frame) for num, frame in buffered_frames if start_frame_num <= num <= end_frame_num]

        if not result_frames:
            print(f"Warning: Could not find frames {start_frame_num}-{end_frame_num} in the buffer (buffer might be too small or frames haven't been read yet).")
        # else:
        #     print(f"DEBUG: Retrieved {len(result_frames)} frames ({result_frames[0][0]} to {result_frames[-1][0]}) from buffer for range {start_frame_num}-{end_frame_num}")

        return result_frames
    # ----------------------------------------