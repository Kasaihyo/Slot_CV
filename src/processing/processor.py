# src/processing/processor.py
import threading
import queue
import cv2
import numpy as np
import tensorflow as tf
import time
from collections import Counter # Added for counting symbols
import os # Added for path checks
from enum import Enum # For states

# Import YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics library not found. Please install it: pip install ultralytics")
    YOLO = None # Set YOLO to None if import fails

# Import utility functions using relative paths
from ..utils.image_utils import preprocess_ocr, clean_ocr_text, preprocess_image_for_model

# Import config values needed here
import config as cfg # Renamed import for clarity

# Define Processing States
class ProcessingState(Enum):
    NORMAL = 1        # Processing at regular interval, not seeking stable grid
    SEARCHING = 2     # Actively looking for full grid (e.g., after stage change)
    CONFIRMING = 3    # Found potential grid, confirming stability

# Define the internal order of classes AS PER THE YOLO MODEL'S TRAINING
# This order is provided by the user and might differ from config.YOLO_CLASS_NAMES
YOLO_MODEL_INTERNAL_CLASS_ORDER = [
    "Blue Skull", "Cyan Skull", "Explosive Wild Symbols", "Golden Hat Skull",
    "Green Skull", "Orange Skull", "Pink Skull", "Scatter Symbols", "Wild Symbols"
]

class FrameProcessor(threading.Thread):
    """Processes frames from a queue: performs OCR, ML prediction, and YOLO detection with grid stability logic."""
    def __init__(self, frame_queue, results_queue, stop_event, processor_config, model, trained_class_names):
        threading.Thread.__init__(self, name="FrameProcessorThread")
        self.frame_queue = frame_queue
        self.results_queue = results_queue
        self.stop_event = stop_event
        self.config = processor_config # Dictionary of relevant config values
        self.model = model # Stage classification model
        self.trained_class_names = trained_class_names # Stage class names
        self.daemon = True

        # --- Load YOLO Model ---
        self.yolo_model = None
        self.yolo_class_names = cfg.YOLO_CLASS_NAMES
        self.yolo_counting_threshold = cfg.YOLO_CONFIDENCE_THRESHOLD # Threshold for counting symbols
        self.yolo_predict_conf = cfg.YOLO_PREDICT_CONF             # Threshold for detecting boxes (finding grid)
        yolo_model_path = cfg.YOLO_MODEL_PATH
        print(f"INFO: Attempting to load YOLO model from {yolo_model_path}...")
        if YOLO is None:
             print("ERROR: Cannot load YOLO model because ultralytics library failed to import.")
        elif not os.path.exists(yolo_model_path):
             print(f"ERROR: YOLO model file not found at {yolo_model_path}. Symbol detection will be disabled.")
        else:
            try:
                self.yolo_model = YOLO(yolo_model_path)
                print(f"INFO: YOLO model loaded successfully from {yolo_model_path}.")
            except Exception as e:
                print(f"ERROR: Failed to load YOLO model '{yolo_model_path}'. Symbol detection disabled. Error: {e}")
                self.yolo_model = None # Ensure it's None on failure

        # --- State Variables for Grid Stability ---
        self.processing_state = ProcessingState.NORMAL
        self.confirmation_count = 0
        self.last_detected_stage = None # Track stage to detect changes

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

        reader = None
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            print("INFO: EasyOCR Reader initialized.")
        except ImportError:
            print("ERROR: easyocr library not found. Please install it: pip install easyocr")
        except Exception as e:
            print(f"ERROR: Failed to initialize EasyOCR Reader: {e}")

        while not self.stop_event.is_set():
            try:
                queue_item = self.frame_queue.get(block=True, timeout=0.5)
            except queue.Empty:
                continue

            if queue_item is None:
                print("INFO: FrameProcessor received None, signaling stop.")
                self.stop_event.set()
                break

            frame_num, frame = queue_item

            # --- Determine if Frame Should Be Processed Based on State ---
            should_process_fully = False
            if self.processing_state == ProcessingState.NORMAL:
                if frame_num % cfg.PROCESS_EVERY_N_FRAMES == 0:
                    should_process_fully = True
            elif self.processing_state in [ProcessingState.SEARCHING, ProcessingState.CONFIRMING]:
                if frame_num % cfg.GRID_SEARCH_FRAME_INTERVAL == 0:
                    should_process_fully = True

            if not should_process_fully:
                minimal_result = {'frame_num': frame_num, 'skipped': True, 'display_frame': frame}
                try: self.results_queue.put_nowait(minimal_result)
                except queue.Full: pass
                continue

            # --- Perform Full Processing ---
            start_time = time.time()
            result = {'frame_num': frame_num, 'skipped': False, 'display_frame': frame.copy()}
            is_grid_stable_this_frame = False # Flag to indicate if counts are from stable grid

            # === 1. OCR Processing === (Same as before)
            cleaned_balance = ""
            cleaned_round = ""
            bal_processed_img = None
            rnd_processed_img = None
            if reader:
                balance_roi_frame = self._extract_roi(frame, self.config['balance_roi'])
                bal_processed_img, bal_raw_text = preprocess_ocr(balance_roi_frame, reader)
                cleaned_balance = clean_ocr_text(bal_raw_text, '0123456789.-')

                round_roi_frame = self._extract_roi(frame, self.config['round_roi'])
                rnd_processed_img, rnd_raw_text = preprocess_ocr(round_roi_frame, reader)
                cleaned_round = clean_ocr_text(rnd_raw_text, '0123456789')
            else:
                balance_roi_frame = self._extract_roi(frame, self.config['balance_roi'])
                round_roi_frame = self._extract_roi(frame, self.config['round_roi'])
                if balance_roi_frame is not None: bal_processed_img = balance_roi_frame
                if round_roi_frame is not None: rnd_processed_img = round_roi_frame

            result['cleaned_balance'] = cleaned_balance
            result['balance_processed_img'] = bal_processed_img
            result['cleaned_round'] = cleaned_round
            result['round_processed_img'] = rnd_processed_img

            # === 2. Stage Detection (ML) === (Mostly same as before)
            detected_stage = None
            stage_prob = 0.0
            stage_known_cls = None
            stages_area_frame = self._extract_roi(frame, self.config['stages_area_roi'])
            result['stages_area_frame'] = stages_area_frame

            if stages_area_frame is not None and self.model is not None:
                input_batch = preprocess_image_for_model(
                    stages_area_frame,
                    self.config['model_img_height'],
                    self.config['model_img_width']
                )
                if input_batch is not None:
                    try:
                        predictions = self.model.predict(input_batch, verbose=0)
                        probs = predictions[0]
                        if len(probs) == len(self.trained_class_names):
                            pred_idx = np.argmax(probs)
                            stage_prob = float(probs[pred_idx])
                            stage_known_cls = self.trained_class_names[pred_idx]
                            if stage_prob >= self.config['prediction_confidence_threshold']:
                                detected_stage = stage_known_cls
                            else:
                                detected_stage = self.config['default_stage']
                        else: print(f"F{frame_num} Proc: ML output/class name mismatch.")
                    except Exception as e: print(f"F{frame_num} Proc: ML Prediction Error: {e}")
                else: print(f"F{frame_num} Proc: Stage ROI preprocessing failed.")

            result['detected_stage'] = detected_stage
            result['stage_prob'] = stage_prob
            result['stage_known_cls'] = stage_known_cls

            # --- State Transition: Check for Stage Change ---
            if detected_stage is not None and detected_stage != self.last_detected_stage:
                print(f"INFO F{frame_num}: Stage changed from '{self.last_detected_stage}' to '{detected_stage}'. Entering SEARCHING state.")
                self.processing_state = ProcessingState.SEARCHING
                self.confirmation_count = 0 # Reset confirmation on stage change
                self.last_detected_stage = detected_stage # Update last seen stage
            elif detected_stage is not None:
                self.last_detected_stage = detected_stage # Update even if no change

            # === 3. Symbol Detection (YOLO) with Stability Logic ===
            symbol_counts = {name: 0 for name in self.yolo_class_names}
            total_symbols = 0
            yolo_input_source = frame # Use the whole frame

            if self.yolo_model is not None and yolo_input_source is not None and yolo_input_source.size > 0:
                try:
                    # Perform YOLO inference using lower confidence for box finding
                    yolo_results = self.yolo_model.predict(
                        source=yolo_input_source,
                        conf=self.yolo_predict_conf, # Use lower confidence for finding grid
                        verbose=False
                    )

                    num_detected_boxes = 0
                    detected_boxes_obj = None # Store the boxes object for potential later counting
                    if yolo_results and len(yolo_results) > 0:
                        detected_boxes_obj = yolo_results[0].boxes
                        if detected_boxes_obj is not None:
                            num_detected_boxes = len(detected_boxes_obj)

                    # --- Grid Stability State Machine ---
                    if self.processing_state == ProcessingState.SEARCHING:
                        if num_detected_boxes == cfg.STABLE_GRID_SIZE:
                            print(f"INFO F{frame_num}: Found potential full grid ({num_detected_boxes} symbols). Entering CONFIRMING state (1/{cfg.GRID_STABILITY_CHECKS}).")
                            self.processing_state = ProcessingState.CONFIRMING
                            self.confirmation_count = 1
                        # Else: Stay in SEARCHING state

                    elif self.processing_state == ProcessingState.CONFIRMING:
                        if num_detected_boxes == cfg.STABLE_GRID_SIZE:
                            self.confirmation_count += 1
                            print(f"INFO F{frame_num}: Confirmed potential grid ({num_detected_boxes} symbols). Check ({self.confirmation_count}/{cfg.GRID_STABILITY_CHECKS}).")
                            if self.confirmation_count >= cfg.GRID_STABILITY_CHECKS:
                                print(f"INFO F{frame_num}: Grid confirmed stable! Performing final count. Entering NORMAL state.")
                                self.processing_state = ProcessingState.NORMAL
                                self.confirmation_count = 0
                                is_grid_stable_this_frame = True # Mark this frame's counts as stable

                                # --- Perform Final Symbol Count with Higher Threshold ---
                                if detected_boxes_obj is not None and len(detected_boxes_obj) > 0:
                                    indices = detected_boxes_obj.cls.int().tolist()
                                    confidences = detected_boxes_obj.conf.tolist()

                                    temp_counts_by_internal_name = Counter() # Temporary counter using internal names
                                    current_total = 0
                                    # --- Debug: Print confidences being checked --- 
                                    print(f"DEBUG F{frame_num} Confidences for counting: {confidences}")
                                    # -------------------------------------------
                                    for i, conf in zip(indices, confidences):
                                        # Apply the COUNTING threshold here
                                        if conf >= self.yolo_counting_threshold:
                                            # Map index to internal name for temp counting
                                            if 0 <= i < len(YOLO_MODEL_INTERNAL_CLASS_ORDER):
                                                internal_name = YOLO_MODEL_INTERNAL_CLASS_ORDER[i]
                                                temp_counts_by_internal_name[internal_name] += 1
                                                current_total += 1
                                            else:
                                                 print(f"Warning F{frame_num}: Detected index {i} out of bounds for internal class order.")
                                        current_total += 1

                                    # --- Create final symbol_counts using CONFIG order --- 
                                    symbol_counts = {name: 0 for name in self.yolo_class_names} # Use config names as keys
                                    for internal_name, count in temp_counts_by_internal_name.items():
                                         # Check if the internally found name exists in the config list
                                         if internal_name in symbol_counts:
                                             symbol_counts[internal_name] = count
                                         else:
                                             print(f"Warning F{frame_num}: Internal name '{internal_name}' not found in config.YOLO_CLASS_NAMES. Check consistency.")
                                    total_symbols = current_total # Assign final total

                                    # --- Draw Boxes on Frame for Preview ---
                                    annotated_frame = yolo_input_source.copy()
                                    if detected_boxes_obj is not None:
                                        boxes = detected_boxes_obj.xyxy.cpu().numpy() # Get coordinates
                                        # indices and confidences already calculated above
                                        for i, conf, box in zip(indices, confidences, boxes):
                                            if conf >= self.yolo_counting_threshold:
                                                x1, y1, x2, y2 = map(int, box)
                                                # Use internal name for label
                                                if 0 <= i < len(YOLO_MODEL_INTERNAL_CLASS_ORDER):
                                                     label_name = YOLO_MODEL_INTERNAL_CLASS_ORDER[i]
                                                else:
                                                     label_name = f"IDX_ERR_{i}"
                                                label = f"{label_name} {conf:.2f}"
                                                color = (0, 255, 0) # Green for counted boxes
                                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                    result['annotated_stable_frame'] = annotated_frame
                                    # -----------------------------------------
                                # --------------------------------------------------------
                        else:
                            # Grid became unstable during confirmation
                            print(f"INFO F{frame_num}: Grid unstable during confirmation (found {num_detected_boxes} symbols, expected {cfg.STABLE_GRID_SIZE}). Re-entering SEARCHING state.")
                            self.processing_state = ProcessingState.SEARCHING
                            self.confirmation_count = 0

                    # --- Debug Prints (Optional) ---
                    # print(f"F{frame_num} State: {self.processing_state.name}, Conf Cnt: {self.confirmation_count}, Boxes: {num_detected_boxes}")
                    # if detected_boxes_obj:
                    #    print(f"F{frame_num} Dets (cls, conf): {list(zip(detected_boxes_obj.cls.int().tolist(), detected_boxes_obj.conf.tolist()))}")

                except Exception as e:
                    print(f"F{frame_num} Proc: YOLO/Stability Error: {e}")
                    # Reset state on error? Maybe back to NORMAL or SEARCHING? Let's reset to SEARCHING.
                    self.processing_state = ProcessingState.SEARCHING
                    self.confirmation_count = 0
                    is_grid_stable_this_frame = False
                    symbol_counts = {name: 0 for name in self.yolo_class_names}
                    total_symbols = 0

            # === Add results (including stability flag and counts) ===
            result['is_grid_stable'] = is_grid_stable_this_frame
            result['symbol_counts'] = symbol_counts
            result['total_symbols'] = total_symbols
            result['processing_state'] = self.processing_state.name # Add state for debugging
            # Add grid event marker if stability was just confirmed
            if is_grid_stable_this_frame:
                result['grid_event'] = 'stable'

            # --- Put results into the results queue ---
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time

            while not self.stop_event.is_set():
                try:
                    self.results_queue.put(result, block=True, timeout=0.5)
                    break
                except queue.Full:
                    continue

        # Signal end
        if not self.stop_event.is_set():
            try:
                 self.results_queue.put(None, block=True, timeout=1)
            except queue.Full:
                 print("Warning: Results queue full when trying to signal processor end.")
        print("INFO: FrameProcessor thread finished.")
