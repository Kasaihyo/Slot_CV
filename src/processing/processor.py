# src/processing/processor.py
import threading
import queue
import cv2
import numpy as np
import tensorflow as tf
import time

# Import utility functions using relative paths
from ..utils.image_utils import preprocess_ocr, clean_ocr_text, preprocess_image_for_model

# Import config but avoid circular dependency if utils need config too
import config # Need TESSERACT_CONFIG here

class FrameProcessor(threading.Thread):
    """Processes frames from a queue: performs OCR and ML prediction."""
    def __init__(self, frame_queue, results_queue, stop_event, processor_config, model, trained_class_names):
        threading.Thread.__init__(self, name="FrameProcessorThread")
        self.frame_queue = frame_queue
        self.results_queue = results_queue
        self.stop_event = stop_event
        self.config = processor_config # Dictionary of relevant config values
        self.model = model
        self.trained_class_names = trained_class_names
        self.daemon = True

        # Ensure Tesseract config is set (if provided in main config)
        if config.TESSERACT_CMD:
             import pytesseract
             try:
                 pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
             except Exception as e:
                  print(f"Warning: Could not set Tesseract command path in Processor: {e}")


    def _extract_roi(self, frame, roi_coords):
        """Safely extracts an ROI from the frame."""
        if frame is None or roi_coords is None:
            return None
        x, y, w, h = roi_coords
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2]
        return None

    def run(self):
        print("INFO: FrameProcessor thread started.")

        while not self.stop_event.is_set():
            try:
                # Wait for an item from the frame queue
                queue_item = self.frame_queue.get(block=True, timeout=0.5) # Timeout allows checking stop_event
            except queue.Empty:
                continue # No frame available yet, loop back

            if queue_item is None: # End signal from reader
                print("INFO: FrameProcessor received None, signaling stop.")
                self.stop_event.set() # Ensure stop event is set if not already
                break

            frame_num, frame = queue_item

            # --- Skip frames based on configuration ---
            if frame_num % self.config['process_every_n_frames'] != 0:
                # Send a minimal result for display update smoothness
                minimal_result = {'frame_num': frame_num, 'skipped': True, 'display_frame': frame}
                try: self.results_queue.put_nowait(minimal_result)
                except queue.Full: pass # Drop if results queue is full
                continue

            # --- Perform Full Processing ---
            start_time = time.time()
            result = {'frame_num': frame_num, 'skipped': False, 'display_frame': frame.copy()}

            # === 1. OCR Processing ===
            # Balance (still using OCR)
            balance_roi_frame = self._extract_roi(frame, self.config['balance_roi'])
            bal_processed_img, bal_raw_text = preprocess_ocr(balance_roi_frame)
            cleaned_balance = clean_ocr_text(bal_raw_text, '0123456789-') # Use standard cleaning
            result['cleaned_balance'] = cleaned_balance
            result['balance_processed_img'] = bal_processed_img

            # Round (Reverting to OCR)
            # Extract ROI
            round_roi_frame = self._extract_roi(frame, self.config['round_roi'])
            # Preprocess and perform OCR
            rnd_processed_img, rnd_raw_text = preprocess_ocr(round_roi_frame)
            # Clean the OCR result
            cleaned_round = clean_ocr_text(rnd_raw_text, '0123456789-')

            result['cleaned_round'] = cleaned_round
            result['round_processed_img'] = rnd_processed_img # Store the preprocessed image for display

            # === 2. Stage Detection (ML) ===
            detected_stage = None
            stage_prob = 0.0
            stage_known_cls = None # The class name the model predicted (from its known classes)
            stages_area_frame = self._extract_roi(frame, self.config['stages_area_roi'])
            result['stages_area_frame'] = stages_area_frame # Store for display

            if stages_area_frame is not None and self.model is not None:
                input_batch = preprocess_image_for_model(
                    stages_area_frame,
                    self.config['model_img_height'],
                    self.config['model_img_width']
                )
                if input_batch is not None:
                    try:
                        predictions = self.model.predict(input_batch, verbose=0) # verbose=0 speeds up prediction
                        probs = predictions[0] # Get probabilities for the first (only) image in the batch

                        if len(probs) == len(self.trained_class_names):
                            pred_idx = np.argmax(probs)
                            stage_prob = float(probs[pred_idx]) # Ensure it's a standard float
                            stage_known_cls = self.trained_class_names[pred_idx]

                            # Decide final stage based on confidence
                            if stage_prob >= self.config['prediction_confidence_threshold']:
                                detected_stage = stage_known_cls
                            else:
                                detected_stage = self.config['default_stage'] # Assign default if uncertain
                        else:
                             print(f"F{frame_num} Proc: ML output size ({len(probs)}) / class name count ({len(self.trained_class_names)}) mismatch.")

                    except Exception as e:
                        print(f"F{frame_num} Proc: ML Prediction Error: {e}")
                        # Optionally set detected_stage to default on error? Or leave as None?
                        # detected_stage = self.config['default_stage']
                else:
                     print(f"F{frame_num} Proc: Stage ROI preprocessing failed.")
            elif self.model is None:
                 print(f"F{frame_num} Proc: Model not loaded, skipping stage detection.")

            result['detected_stage'] = detected_stage
            result['stage_prob'] = stage_prob
            result['stage_known_cls'] = stage_known_cls # Raw prediction before thresholding

            # --- Put results into the results queue ---
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time # Add timing info if needed

            try:
                self.results_queue.put_nowait(result)
            except queue.Full:
                # This shouldn't happen often if main thread consumes quickly
                print(f"Warning: Results queue full at frame {frame_num}. Dropping result.")
                pass # Drop result if queue is full to avoid blocking processor

        # Signal end for main thread by putting None in results queue
        try:
             self.results_queue.put(None, block=True, timeout=1)
        except queue.Full:
             print("Warning: Results queue full when trying to signal processor end.")
        print("INFO: FrameProcessor thread finished.")
