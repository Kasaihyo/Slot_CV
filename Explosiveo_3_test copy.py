import cv2
import pytesseract
import pandas as pd
import datetime
import re
import numpy as np
import sys # For exiting cleanly
import tensorflow as tf # Import TensorFlow
import threading # Import threading
import queue     # Import queue
import time      # For sleep and timing

# --- Configuration ---

# 1. Tesseract Path (Uncomment and modify IF TESSERACT IS NOT IN YOUR SYSTEM PATH)
# TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Windows example
# try:
#     pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
# except NameError: # TESSERACT_CMD wasn't defined
#     pass
# except Exception as e:
#      print(f"Warning: Could not set Tesseract command path: {e}")

# 2. Video Source
video_path = 'Videos/Sequence 01.mp4' # Path to your video file
# video_path = 0 # Use 0 for the default webcam

# 3. ROIs - Region of Interest Coordinates (x, y, width, height)
#    Set specific ROIs to None to select them interactively on the first run.
balance_roi = (98, 1753, 130, 52)   # Example: ROI for balance number OCR
round_roi = (1764, 1660, 78, 46)      # Example: ROI for round number OCR
stages_area_roi = (921, 1473, 1175, 190) # ROI covering ALL multipliers (x1 to x32) - Adjust if needed
# stages_area_roi = None # Set to None to select interactively

# 4. Tesseract Configuration for OCR
tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-'

# 5. Output File for Logging
output_csv_file = 'game_log_data_ml.csv' # Changed filename for ML version

# 6. Stability Control (Number of consecutive matching frames required)
CONFIRMATION_FRAMES_OCR = 3   # For balance and round number stability
CONFIRMATION_FRAMES_STAGE = 3 # For ML stage detection stability

# 7. Stage Detection Configuration (USING ML MODEL)
MODEL_PATH = 'stage_classifier_model_5class.keras' # Path to the saved Keras model (5 classes)
CLASS_NAMES_PATH = 'stage_class_names_5class.txt'# Path to the 5 class names text file
# --- Define ALL possible stages, including the default one ---
ALL_POSSIBLE_STAGES = ["x1", "x2", "x4", "x8", "x16", "x32"]
DEFAULT_STAGE = "x32" # The stage to assign if confidence is low for known classes

# --- Model Input Shape (MUST match training script) ---
MODEL_IMG_HEIGHT = 86
MODEL_IMG_WIDTH = 493
# --- Prediction Confidence Threshold ---
# Adjust this based on model performance and desired behavior (0.0 to 1.0)
# Higher value means model needs to be more certain to assign x1-x16, otherwise defaults to x32
PREDICTION_CONFIDENCE_THRESHOLD = 0.80 # e.g., 80% confidence required

# 8. Performance (Process every Nth frame)
process_every_n_frames = 5 # Process slightly more often for ML responsiveness

# 9. Balance Correction (If OCR reads balance without decimal)
DECIMAL_CORRECTION = 100 # Divide OCR balance by this value (e.g., 100)
# Set to 1 or None if no correction is needed

# --- Threading Configuration ---
FRAME_QUEUE_MAXSIZE = 60 # Max frames to buffer from video reader (prevents memory explosion)
USE_PROCESSING_THREAD = True # Set to False to revert to single-threaded for comparison

# --- Load Trained Model ---
print(f"INFO: Loading stage classification model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("INFO: Model loaded successfully.")
    # Optional: Print model summary
    # model.summary()
except Exception as e:
    print(f"ERROR: Failed to load model '{MODEL_PATH}'. "
          f"Make sure the file exists and TensorFlow is installed correctly. Exiting. Error: {e}")
    sys.exit()

# --- Load Trained Class Names (5 classes) ---
print(f"INFO: Loading TRAINED class names from {CLASS_NAMES_PATH}...")
try:
    with open(CLASS_NAMES_PATH, "r") as f:
        # These are the classes the model actually knows (x1 to x16)
        trained_class_names = [line.strip() for line in f.readlines()]
    print(f"INFO: Loaded TRAINED class names: {trained_class_names}")
    # Check if the number of loaded names matches expectation (usually 5)
    if len(trained_class_names) != model.output_shape[-1]:
         print(f"WARNING: Number of loaded class names ({len(trained_class_names)}) "
               f"does not match model output units ({model.output_shape[-1]}). "
               "Ensure class names file corresponds to the loaded model.")
except Exception as e:
    print(f"ERROR: Failed to load class names '{CLASS_NAMES_PATH}'. Exiting. Error: {e}")
    sys.exit()

# --- Initialization ---
data_log = []
# State variables for tracking changes and stability
last_logged_balance_text = None
potential_new_balance = None
balance_confirmation_counter = 0
last_confirmed_balance_value = None

last_logged_round = None
potential_new_round = None
round_confirmation_counter = 0

last_logged_stage = None
potential_new_stage = None
stage_confirmation_counter = 0
current_confirmed_stage = None
previous_confirmed_stage = None  # Add this to track stage changes

frame_count = 0
last_round_balance_value = None
accumulated_win = 0.00  # Add this to track accumulated win within a round

# --- Queues for Thread Communication ---
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
results_queue = queue.Queue() # Queue to hold processing results
stop_event = threading.Event() # Signal for threads to stop

# --- Helper Function for ROI Selection ---
def select_roi_interactively(window_title, frame_to_select_on):
    """Handles interactive ROI selection."""
    print(f"\nINFO: Select '{window_title}' in the window.")
    print("Draw a rectangle and press ENTER or SPACE.")
    print("Press 'c' to cancel selection (will exit script).")
    # Use a unique window name to avoid conflicts if called multiple times quickly
    temp_window_name = f"{window_title}_{datetime.datetime.now().timestamp()}"
    cv2.namedWindow(temp_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(temp_window_name, 800, 600) # Adjust size if needed
    roi_coords = cv2.selectROI(temp_window_name, frame_to_select_on, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(temp_window_name)
    if roi_coords == (0, 0, 0, 0):
        print(f"ERROR: ROI selection cancelled for '{window_title}'. Exiting.")
        sys.exit() # Exit script if user cancels
    print(f"INFO: ROI for '{window_title}' selected: {roi_coords}")
    return roi_coords

# --- Preprocessing Function for Model ---
def preprocess_image_for_model(img_roi, target_height, target_width):
    """Resizes an image ROI for model prediction, keeping color."""
    if img_roi is None or img_roi.size == 0:
        return None
    # Resize - Use INTER_AREA for shrinking, INTER_LINEAR for enlarging generally good
    if img_roi.shape[0] > target_height or img_roi.shape[1] > target_width:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    img_resized = cv2.resize(img_roi, (target_width, target_height), interpolation=interpolation)
    # Add batch dimension (Model expects batch_size, height, width, channels)
    img_batch = np.expand_dims(img_resized, axis=0)
    # Normalization is typically done *inside* the model with a Rescaling layer now
    # If your model DOES NOT have a Rescaling layer, uncomment the next line:
    # img_batch = img_batch.astype(np.float32) / 255.0
    return img_batch

# --- Video Reading Thread ---
class VideoReader(threading.Thread):
    def __init__(self, video_path, frame_queue, stop_event):
        threading.Thread.__init__(self)
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.cap = None
        self.daemon = True # Allows main program to exit even if this thread is running

    def run(self):
        print("INFO: VideoReader thread started.")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"ERROR: VideoReader could not open video source: {self.video_path}")
            self.stop_event.set() # Signal other threads to stop
            return

        f_count = 0
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("INFO: VideoReader reached end of video or read error.")
                break # End of video

            f_count += 1
            try:
                # Put frame and frame number into the queue, block if full
                self.frame_queue.put((f_count, frame), block=True, timeout=1)
            except queue.Full:
                # Log if queue is consistently full (processing is too slow)
                # print("Warning: Frame queue is full. Processing might be lagging.")
                time.sleep(0.01) # Avoid busy-waiting if queue is full
                continue # Try putting again
            except Exception as e:
                 print(f"ERROR in VideoReader putting frame: {e}")
                 break

        # Signal end of stream by putting None
        try:
            self.frame_queue.put(None, block=False) # Signal end
        except queue.Full:
            pass # If queue is full at the end, processor will eventually hit None

        if self.cap:
            self.cap.release()
        print("INFO: VideoReader thread finished.")

# --- Frame Processing Thread ---
class FrameProcessor(threading.Thread):
    def __init__(self, frame_queue, results_queue, stop_event, config):
        threading.Thread.__init__(self)
        self.frame_queue = frame_queue
        self.results_queue = results_queue
        self.stop_event = stop_event
        self.config = config # Pass necessary configs (ROIs, model details, etc.)
        self.daemon = True

    def run(self):
        print("INFO: FrameProcessor thread started.")
        global model, trained_class_names # Access model loaded in main thread

        while not self.stop_event.is_set():
            try:
                queue_item = self.frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue # No frame available yet

            if queue_item is None: # End signal
                break

            frame_num, frame = queue_item

            # --- Only process every N frames ---
            if frame_num % self.config['process_every_n_frames'] != 0:
                # Optionally put a minimal result for display update?
                # Or just skip putting anything in results_queue for skipped frames
                # Putting minimal result allows smoother display updates
                try:
                     self.results_queue.put({'frame_num': frame_num, 'skipped': True, 'display_frame': frame}, block=False)
                except queue.Full:
                     pass # Skip putting result if full
                continue

            # --- Perform Full Processing ---
            frame_h, frame_w = frame.shape[:2]
            result = {'frame_num': frame_num, 'skipped': False, 'display_frame': frame.copy()} # Start result dict

            # === 1. OCR Processing ===
            cleaned_balance = ""
            cleaned_round = ""
            balance_img = None
            round_img = None

            # Balance ROI
            bx, by, bw, bh = self.config['balance_roi']
            bx1, by1, bx2, by2 = max(0, bx), max(0, by), min(frame_w, bx + bw), min(frame_h, by + bh)
            if bx2 > bx1 and by2 > by1:
                bal_roi_f = frame[by1:by2, bx1:bx2]
                bal_gray = cv2.cvtColor(bal_roi_f, cv2.COLOR_BGR2GRAY)
                bal_thresh = cv2.threshold(bal_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                balance_img = bal_thresh # For potential display later
                try:
                    bal_raw = pytesseract.image_to_string(bal_thresh, config=self.config['tesseract_config'])
                    cleaned_balance = re.sub(r'[^\d-]+', '', bal_raw).strip()
                except Exception as e: print(f"F{frame_num} Proc: Bal OCR Err: {e}")
            result['cleaned_balance'] = cleaned_balance
            result['balance_processed_img'] = balance_img

            # Round ROI
            rx, ry, rw, rh = self.config['round_roi']
            rx1, ry1, rx2, ry2 = max(0, rx), max(0, ry), min(frame_w, rx + rw), min(frame_h, ry + rh)
            if rx2 > rx1 and ry2 > ry1:
                rnd_roi_f = frame[ry1:ry2, rx1:rx2]
                rnd_gray = cv2.cvtColor(rnd_roi_f, cv2.COLOR_BGR2GRAY)
                rnd_thresh = cv2.threshold(rnd_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                round_img = rnd_thresh
                try:
                    rnd_raw = pytesseract.image_to_string(rnd_thresh, config=self.config['tesseract_config'])
                    cleaned_round = re.sub(r'[^\d-]+', '', rnd_raw).strip()
                except Exception as e: print(f"F{frame_num} Proc: Rnd OCR Err: {e}")
            result['cleaned_round'] = cleaned_round
            result['round_processed_img'] = round_img

            # === 2. Stage Detection (ML) ===
            detected_stage = None
            stage_prob = 0.0
            stage_known_cls = None
            stage_img = None

            sx, sy, sw, sh = self.config['stages_area_roi']
            sx1, sy1, sx2, sy2 = max(0, sx), max(0, sy), min(frame_w, sx + sw), min(frame_h, sy + sh)
            if sx2 > sx1 and sy2 > sy1:
                stage_roi_f = frame[sy1:sy2, sx1:sx2]
                stage_img = stage_roi_f # For display
                input_batch = preprocess_image_for_model(stage_roi_f, self.config['MODEL_IMG_HEIGHT'], self.config['MODEL_IMG_WIDTH'])
                if input_batch is not None:
                    try:
                        predictions = model.predict(input_batch, verbose=0)
                        probs = predictions[0]
                        if len(probs) == len(trained_class_names):
                            pred_idx = np.argmax(probs)
                            stage_prob = probs[pred_idx]
                            stage_known_cls = trained_class_names[pred_idx]
                            if stage_prob >= self.config['PREDICTION_CONFIDENCE_THRESHOLD']:
                                detected_stage = stage_known_cls
                            else:
                                detected_stage = self.config['DEFAULT_STAGE']
                        else: print(f"F{frame_num} Proc: ML output/class mismatch.")
                    except Exception as e: print(f"F{frame_num} Proc: ML Predict Err: {e}")
            result['detected_stage'] = detected_stage
            result['stage_prob'] = stage_prob
            result['stage_known_cls'] = stage_known_cls
            result['stages_area_frame'] = stage_img # For display

            # --- Put results into the results queue ---
            try:
                self.results_queue.put(result, block=False)
            except queue.Full:
                # This shouldn't happen often if main thread consumes quickly
                print(f"Warning: Results queue full at frame {frame_num}.")
                pass # Drop result if queue is full? Or block? Blocking might stall processing.

        # Signal end for main thread
        try:
             self.results_queue.put(None, block=False)
        except queue.Full:
             pass
        print("INFO: FrameProcessor thread finished.")

# --- Video Processing ---
print(f"INFO: Opening video source to get details: {video_path}")
temp_cap = cv2.VideoCapture(video_path)
if not temp_cap.isOpened():
    print(f"ERROR: Could not open video source: {video_path}")
    sys.exit()
ret, first_frame = temp_cap.read()
if not ret:
    print("ERROR: Could not read the first frame.")
    temp_cap.release()
    sys.exit()
frame_h_orig, frame_w_orig = first_frame.shape[:2]
print(f"INFO: Video frame dimensions: {frame_w_orig}x{frame_h_orig}")
temp_cap.release() # Close temporary capture

# --- Interactive ROI Selection (if needed) ---
needs_roi_selection = any(roi is None for roi in [stages_area_roi, round_roi, balance_roi])
if stages_area_roi is None:
    stages_area_roi = select_roi_interactively("Stages Area ROI (Cover ALL x1-x32)", first_frame)
if round_roi is None:
    round_roi = select_roi_interactively("Round Number ROI", first_frame)
if balance_roi is None:
    balance_roi = select_roi_interactively("Balance Number ROI", first_frame)

# Unpack ROIs for use in loop
bx, by, bw, bh = balance_roi
rx, ry, rw, rh = round_roi
stages_x, stages_y, stages_w, stages_h = stages_area_roi

# --- Prepare Config Dict for Threads ---
processing_config = {
    'balance_roi': balance_roi,
    'round_roi': round_roi,
    'stages_area_roi': stages_area_roi,
    'tesseract_config': tesseract_config,
    'MODEL_IMG_HEIGHT': MODEL_IMG_HEIGHT,
    'MODEL_IMG_WIDTH': MODEL_IMG_WIDTH,
    'PREDICTION_CONFIDENCE_THRESHOLD': PREDICTION_CONFIDENCE_THRESHOLD,
    'DEFAULT_STAGE': DEFAULT_STAGE,
    'process_every_n_frames': process_every_n_frames,
}

# --- Start Threads ---
if USE_PROCESSING_THREAD:
    reader = VideoReader(video_path, frame_queue, stop_event)
    processor = FrameProcessor(frame_queue, results_queue, stop_event, processing_config)
    reader.start()
    processor.start()
    print("INFO: Reader and Processor threads started.")
else: # Fallback to single-threaded for testing/comparison
    print("INFO: Running in single-threaded mode.")
    cap = cv2.VideoCapture(video_path) # Open capture directly
    if not cap.isOpened(): sys.exit("Error opening video in single-thread mode.")

# --- Main Loop (Handles Results, State, Display) ---
print("\n" + "="*40)
print("INFO: Starting main processing loop using ML Stage Classification...")
print(f" - Model Path: {MODEL_PATH}")
print(f" - Trained Classes: {trained_class_names}")
print(f" - Default Stage (Low Confidence): {DEFAULT_STAGE}")
print(f" - Confidence Threshold: {PREDICTION_CONFIDENCE_THRESHOLD:.2f}")
print(f" - OCR Stability: {CONFIRMATION_FRAMES_OCR} frames")
print(f" - Stage Stability: {CONFIRMATION_FRAMES_STAGE} frames")
print(f" - Processing every {process_every_n_frames} frames")
print("Press 'q' in the 'Video Feed' window to quit.")
print("="*40 + "\n")

latest_display_frame = first_frame # Initialize display frame
latest_balance_img = None
latest_round_img = None
latest_stage_img = None
last_processed_frame_num = 0

try:
    while True:
        processed_a_result = False
        result = None
        # Initialize prediction variables for display logic
        current_detected_stage = None
        predicted_known_class = None
        predicted_prob = 0.0

        if USE_PROCESSING_THREAD:
            try:
                # Get results from the processor thread
                result = results_queue.get(block=True, timeout=0.01) # Block shortly
                if result is None: # End signal from processor
                    print("INFO: Main loop received None, breaking.")
                    break
                processed_a_result = True
            except queue.Empty:
                # No new results, main loop can continue other tasks or brief sleep
                 pass # Keep looping to check for 'q' press
        else: # Single-threaded mode: Read and process directly
             if cap.isOpened():
                 ret, frame = cap.read()
                 if not ret: break # End of video
                 frame_count += 1
                 
                 # Initialize result to ensure it's not None
                 result = {'frame_num': frame_count, 'skipped': True, 'display_frame': frame}

                 if frame_count % process_every_n_frames == 0:
                     # Direct processing code would go here
                     # For brevity, we're omitting the single-threaded processing
                     # This should set more values in result
                     pass
                 
                 processed_a_result = True
             else: break # Capture closed

        # --- Process the received result (if any) ---
        if processed_a_result and result:
            frame_num = result['frame_num']
            latest_display_frame = result['display_frame'] # Update frame for display
            last_processed_frame_num = frame_num

            if not result.get('skipped', True):
                # Update images for display windows if available
                latest_balance_img = result.get('balance_processed_img')
                latest_round_img = result.get('round_processed_img')
                latest_stage_img = result.get('stages_area_frame')

                # Extract processed data
                cleaned_balance_text = result.get('cleaned_balance', "")
                cleaned_round_text = result.get('cleaned_round', "")
                current_detected_stage = result.get('detected_stage')
                predicted_prob = result.get('stage_prob', 0.0)
                predicted_known_class = result.get('stage_known_cls')

                # --- Apply Stability Checks ---
                # Balance Stability
                if not cleaned_balance_text: potential_new_balance = None; balance_confirmation_counter = 0
                elif cleaned_balance_text == last_logged_balance_text: potential_new_balance = None; balance_confirmation_counter = 0
                elif cleaned_balance_text == potential_new_balance: balance_confirmation_counter += 1
                else: potential_new_balance = cleaned_balance_text; balance_confirmation_counter = 1
                # Round Stability
                if not cleaned_round_text: potential_new_round = None; round_confirmation_counter = 0
                elif cleaned_round_text == last_logged_round: potential_new_round = None; round_confirmation_counter = 0
                elif cleaned_round_text == potential_new_round: round_confirmation_counter += 1
                else: potential_new_round = cleaned_round_text; round_confirmation_counter = 1
                # Stage Stability
                if not current_detected_stage: potential_new_stage = None; stage_confirmation_counter = 0
                elif current_detected_stage == last_logged_stage: potential_new_stage = None; stage_confirmation_counter = 0
                elif current_detected_stage == potential_new_stage: stage_confirmation_counter += 1
                else: potential_new_stage = current_detected_stage; stage_confirmation_counter = 1

                # --- Update Confirmed Balance ---
                if balance_confirmation_counter >= CONFIRMATION_FRAMES_OCR:
                    try:
                        balance_value = int(potential_new_balance)
                        if DECIMAL_CORRECTION and DECIMAL_CORRECTION != 0:
                            last_confirmed_balance_value = balance_value / DECIMAL_CORRECTION
                        else:
                            last_confirmed_balance_value = float(balance_value)
                    except (ValueError, TypeError):
                        print(f"Frame {frame_num}: Warning - Could not convert confirmed balance '{potential_new_balance}' to number.")
                    last_logged_balance_text = potential_new_balance
                    potential_new_balance = None; balance_confirmation_counter = 0

                # --- Update Confirmed Stage & Log Stage Change ---
                if stage_confirmation_counter >= CONFIRMATION_FRAMES_STAGE:
                    # Check if stage actually changed before logging mid-round change
                    previous_confirmed_stage = current_confirmed_stage  # Store previous stage before updating
                    last_logged_stage = potential_new_stage
                    current_confirmed_stage = potential_new_stage # Update current state
                    
                    # Log stage change if it happens *during* a round
                    if (current_confirmed_stage != previous_confirmed_stage and 
                        previous_confirmed_stage is not None and 
                        last_logged_round is not None):
                        timestamp = datetime.datetime.now()
                        
                        # Create log entry for stage change within round
                        balance_display_log = f"{last_confirmed_balance_value:.2f}" if last_confirmed_balance_value is not None else None
                        balance_print = f"{last_confirmed_balance_value:.2f}" if last_confirmed_balance_value is not None else f"(RAW:{last_logged_balance_text})"
                        
                        log_entry = {
                            'Timestamp': timestamp,
                            'Frame': frame_num,
                            'RoundValue': last_logged_round,
                            'Stage': current_confirmed_stage,
                            'CorrectedBalance': balance_display_log,
                            'BalanceChange': "0.00",  # No balance change during stage change
                            'Outcome': "STAGE_CHANGE",
                            'RawBalanceValue': last_logged_balance_text,
                            'AccumulatedWin': f"{accumulated_win:.2f}"  # Use current accumulated win
                        }
                        data_log.append(log_entry)
                        
                        # Print confirmation to console
                        print(f"STAGE CHANGE >> Frame:{frame_num} Round:'{last_logged_round}' Stage:'{current_confirmed_stage}' "
                              f"Balance:{balance_print} Change:0.00")
                    
                    potential_new_stage = None; stage_confirmation_counter = 0 # Reset potential

                # --- Log on Confirmed Round Change ---
                if round_confirmation_counter >= CONFIRMATION_FRAMES_OCR:
                    timestamp = datetime.datetime.now()
                    current_round_text = potential_new_round # The round number just confirmed

                    # Calculate Balance Change
                    balance_change = 0.00
                    outcome = "N/A" # Default outcome
                    if last_confirmed_balance_value is not None:
                        if last_round_balance_value is not None:
                            balance_change = last_confirmed_balance_value - last_round_balance_value
                            outcome = f"{balance_change:.2f}"
                            # Reset accumulated win at the start of a new round
                            accumulated_win = balance_change
                        else:
                            outcome = "START"
                            accumulated_win = 0.00 # Reset at start
                        last_round_balance_value = last_confirmed_balance_value
                    else:
                         outcome = "BAL_ERR"
                         last_round_balance_value = None
                         accumulated_win = 0.00 # Reset on error

                    balance_display_log = f"{last_confirmed_balance_value:.2f}" if last_confirmed_balance_value is not None else None
                    balance_print = f"{last_confirmed_balance_value:.2f}" if last_confirmed_balance_value is not None else f"(RAW:{last_logged_balance_text})"

                    # Create Log Entry
                    log_entry = {
                        'Timestamp': timestamp,
                        'Frame': frame_num,
                        'RoundValue': current_round_text,
                        'Stage': current_confirmed_stage, # Log the stage confirmed at this time
                        'CorrectedBalance': balance_display_log,
                        'BalanceChange': f"{balance_change:.2f}",
                        'Outcome': outcome,
                        'RawBalanceValue': last_logged_balance_text,
                        'AccumulatedWin': f"{accumulated_win:.2f}"  # Add accumulated win to log
                    }
                    data_log.append(log_entry)
                    
                    # Print Confirmation to Console
                    print(f"ROUND CHANGE >> Frame:{frame_num} Round:'{current_round_text}' Stage:'{current_confirmed_stage}' "
                          f"Balance:{balance_print} Change:{outcome} Accumulated:{accumulated_win:.2f}")

                    # Update and Reset Round State
                    last_logged_round = current_round_text
                    potential_new_round = None
                    round_confirmation_counter = 0
                    previous_confirmed_stage = current_confirmed_stage  # Update previous stage at round change

        # === Display Frame and Information ===
        if latest_display_frame is not None:
            display_copy = latest_display_frame.copy() # Work on copy for drawing
            frame_h, frame_w = display_copy.shape[:2]

            # Draw ROIs
            bx1, by1, bx2, by2 = max(0, bx), max(0, by), min(frame_w, bx + bw), min(frame_h, by + bh)
            cv2.rectangle(display_copy, (bx1, by1), (bx2, by2), (0, 255, 0), 2)  # Balance (Green)
            rx1, ry1, rx2, ry2 = max(0, rx), max(0, ry), min(frame_w, rx + rw), min(frame_h, ry + rh)
            cv2.rectangle(display_copy, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)  # Round (Blue)
            sx1, sy1, sx2, sy2 = max(0, stages_x), max(0, stages_y), min(frame_w, stages_x + stages_w), min(frame_h, stages_y + stages_h)
            cv2.rectangle(display_copy, (sx1, sy1), (sx2, sy2), (0, 200, 200), 2) # Stages Area (Teal)

            # Prepare Status Text
            balance_str_disp = f"{last_confirmed_balance_value:.2f}" if last_confirmed_balance_value is not None else (last_logged_balance_text if last_logged_balance_text else "N/A")

            # Display stage status clearly
            stage_display = 'None'
            if current_confirmed_stage:
                 stage_display = current_confirmed_stage
            elif potential_new_stage:
                 stage_display = f"{potential_new_stage}?({stage_confirmation_counter})"

            status_text = f"R: {last_logged_round}  S: {stage_display}  B: {balance_str_disp}"

            # Show OCR confirmation status
            potentials_ocr = []
            if potential_new_round: potentials_ocr.append(f"R?{potential_new_round}({round_confirmation_counter})")
            if potential_new_balance: potentials_ocr.append(f"B?{potential_new_balance}({balance_confirmation_counter})")
            if potentials_ocr: status_text += " | OCR?: " + " ".join(potentials_ocr)

            # Show raw ML prediction detail if stage is uncertain or being confirmed
            if result is not None and not result.get('skipped', True) and (not current_confirmed_stage or potential_new_stage) and predicted_known_class:
                 raw_pred_text = f"(ML: {predicted_known_class} {predicted_prob:.2f})"
                 # Show if it defaulted
                 if current_detected_stage == DEFAULT_STAGE and predicted_prob < PREDICTION_CONFIDENCE_THRESHOLD:
                     raw_pred_text += f" -> {DEFAULT_STAGE}"
                 status_text += f" {raw_pred_text}"

            # Put Text on Frame
            cv2.putText(display_copy, f"Frame: {last_processed_frame_num}", (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display_copy, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 50), 2) # Bright green text

            # Show Windows
            cv2.imshow("Video Feed", display_copy)
            if result is not None and not result.get('skipped', True):
                if latest_balance_img is not None: cv2.imshow("Balance OCR Input", latest_balance_img)
                if latest_round_img is not None: cv2.imshow("Round OCR Input", latest_round_img)
                if latest_stage_img is not None: cv2.imshow("Stages Area Input (ML)", latest_stage_img) # Show what model sees

            # Exit Condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("INFO: 'q' pressed, exiting.")
                raise KeyboardInterrupt
            
            # Check if threads are still running (in threaded mode)
            if USE_PROCESSING_THREAD and (not reader.is_alive() or not processor.is_alive()):
                if not stop_event.is_set(): # If they stopped unexpectedly
                     print("ERROR: A processing thread has stopped unexpectedly. Exiting.")
                     raise SystemExit # Force exit if a thread died

except KeyboardInterrupt: # Handle Ctrl+C or 'q'
    print("INFO: Process interrupted by user.")

finally:
    # --- Cleanup ---
    print("INFO: Stopping threads and cleaning up...")
    stop_event.set() # Signal threads to stop

    if USE_PROCESSING_THREAD:
        if 'reader' in locals() and reader.is_alive():
            reader.join(timeout=2) # Wait for reader thread 
        if 'processor' in locals() and processor.is_alive():
            processor.join(timeout=2) # Wait for processor thread
    elif 'cap' in locals() and cap.isOpened():
        cap.release()
        
    cv2.destroyAllWindows()
    # Add a small delay to ensure windows are closed before final print
    time.sleep(0.5)
    print("INFO: Cleanup complete.")

    # --- Save Data Log ---
    if data_log:
        print(f"INFO: Saving {len(data_log)} logged entries to '{output_csv_file}'...")
        df = pd.DataFrame(data_log)
        # Define desired column order
        cols_order = ['Timestamp', 'Frame', 'RoundValue', 'Stage', 'CorrectedBalance',
                      'BalanceChange', 'Outcome', 'RawBalanceValue', 'AccumulatedWin']
        # Reindex DataFrame, keeping only existing columns from the desired list
        df = df.reindex(columns=[col for col in cols_order if col in df.columns])

        try:
            df.to_csv(output_csv_file, index=False, float_format='%.2f')
            print(f"INFO: Data log successfully saved.")
            print("\n--- Final Log Summary ---")
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(df)
            print("-" * 60)
        except Exception as e:
            print(f"\nERROR: Failed to save data log to CSV: {e}")
            print("\n--- Log Data (NOT SAVED) ---")
            for entry in data_log:
                print(entry)
            print("-" * 60)
    else:
        print("\nINFO: No data entries were logged.")

print("INFO: Script finished.")