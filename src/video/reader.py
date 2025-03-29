# src/video/reader.py
import threading
import cv2
import queue
import time

class VideoReader(threading.Thread):
    """Reads frames from a video source and puts them into a queue."""
    def __init__(self, video_path, frame_queue, stop_event, max_queue_size):
        threading.Thread.__init__(self, name="VideoReaderThread")
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.max_queue_size = max_queue_size
        self.cap = None
        self.daemon = True # Allows main program to exit even if this thread is blocking

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