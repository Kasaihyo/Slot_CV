import cv2
import pytesseract
import pandas as pd
import datetime
import re
import numpy as np
import sys # For exiting cleanly
import tensorflow as tf # Import TensorFlow

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
video_path = 'Videos/high_slot.mov' # Path to your video file
# video_path = 0 # Use 0 for the default webcam

# 3. ROIs - Region of Interest Coordinates (x, y, width, height)
#    Set specific ROIs to None to select them interactively on the first run.
balance_roi = (141, 1293, 106, 41)    # Example: ROI for balance number OCR
round_roi = (1571, 1211, 57, 38)      # Example: ROI for round number OCR
stages_area_roi = (845, 1051, 933, 162) # ROI covering ALL multipliers (x1 to x32) - Adjust if needed
# stages_area_roi = None # Set to None to select interactively

# 4. Tesseract Configuration for OCR
tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-'

# 5. Output File for Logging
output_csv_file = 'game_log_data_ml.csv' # Changed filename for ML version

# 6. Stability Control (Number of consecutive matching frames required)
CONFIRMATION_FRAMES_OCR = 2   # For balance and round number stability
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
process_every_n_frames = 6 # Process slightly more often for ML responsiveness

# 9. Balance Correction (If OCR reads balance without decimal)
DECIMAL_CORRECTION = 100 # Divide OCR balance by this value (e.g., 100)
# Set to 1 or None if no correction is needed


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


# --- Video Processing ---
print(f"INFO: Opening video source: {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"ERROR: Could not open video source: {video_path}")
    sys.exit()

# --- Read First Frame for ROI Selection (if needed) ---
ret, first_frame = cap.read()
if not ret:
    print("ERROR: Could not read the first frame from the video source.")
    cap.release()
    sys.exit()
frame_h_orig, frame_w_orig = first_frame.shape[:2]
print(f"INFO: Video frame dimensions: {frame_w_orig}x{frame_h_orig}")

# --- Interactive ROI Selection (if any ROI is None) ---
needs_roi_selection = any(roi is None for roi in [stages_area_roi, round_roi, balance_roi])

# Select ROIs in a specific order if needed
if stages_area_roi is None:
    stages_area_roi = select_roi_interactively("Stages Area ROI (Cover ALL x1-x32)", first_frame)
if round_roi is None:
    round_roi = select_roi_interactively("Round Number ROI", first_frame)
if balance_roi is None:
    balance_roi = select_roi_interactively("Balance Number ROI", first_frame)

# Reopen video source if reading from file AND interactive selection occurred
if needs_roi_selection and isinstance(video_path, str):
    print("INFO: Re-opening video file from the beginning after ROI selection.")
    cap.release()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not re-open video source after ROI selection: {video_path}")
        sys.exit()

# Unpack ROIs for use in loop
bx, by, bw, bh = balance_roi
rx, ry, rw, rh = round_roi
stages_x, stages_y, stages_w, stages_h = stages_area_roi

# --- Start Processing ---
print("\n" + "="*40)
print("INFO: Starting video processing using ML Stage Classification...")
print(f" - Model Path: {MODEL_PATH}")
print(f" - Trained Classes: {trained_class_names}")
print(f" - Default Stage (Low Confidence): {DEFAULT_STAGE}")
print(f" - Confidence Threshold: {PREDICTION_CONFIDENCE_THRESHOLD:.2f}")
print(f" - OCR Stability: {CONFIRMATION_FRAMES_OCR} frames")
print(f" - Stage Stability: {CONFIRMATION_FRAMES_STAGE} frames")
print(f" - Processing every {process_every_n_frames} frames")
print("Press 'q' in the 'Video Feed' window to quit.")
print("="*40 + "\n")

try: # Main processing loop within a try block for clean exit
    while True:
        ret, frame = cap.read()
        if not ret:
            print("INFO: End of video source reached.")
            break # Exit loop if video ends or fails to read

        frame_count += 1
        frame_h, frame_w = frame.shape[:2] # Get current frame dimensions

        # --- Frame Skipping Logic ---
        if frame_count % process_every_n_frames != 0:
            # Minimal display on skipped frames
            display_frame_skipped = frame.copy()
            try:
                bx1_d, by1_d, bx2_d, by2_d = max(0, bx), max(0, by), min(frame_w, bx + bw), min(frame_h, by + bh)
                cv2.rectangle(display_frame_skipped, (bx1_d, by1_d), (bx2_d, by2_d), (0, 100, 0), 1)
                rx1_d, ry1_d, rx2_d, ry2_d = max(0, rx), max(0, ry), min(frame_w, rx + rw), min(frame_h, ry + rh)
                cv2.rectangle(display_frame_skipped, (rx1_d, ry1_d), (rx2_d, ry2_d), (100, 0, 0), 1)
                sx1_d, sy1_d, sx2_d, sy2_d = max(0, stages_x), max(0, stages_y), min(frame_w, stages_x + stages_w), min(frame_h, stages_y + stages_h)
                cv2.rectangle(display_frame_skipped, (sx1_d, sy1_d), (sx2_d, sy2_d), (0, 100, 100), 1)
                balance_str_disp = f"{last_confirmed_balance_value:.2f}" if last_confirmed_balance_value is not None else "N/A"
                status_text = f"Frame: {frame_count} (Skipped) | R:{last_logged_round} S:{current_confirmed_stage} B:{balance_str_disp}"
                cv2.putText(display_frame_skipped, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                cv2.imshow("Video Feed", display_frame_skipped)
            except Exception as e:
                 print(f"Warning: Error displaying skipped frame {frame_count}: {e}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                 raise KeyboardInterrupt
            continue # Skip processing

        # --- Process This Frame ---
        display_frame = frame.copy() # For visualization

        # === 1. OCR Processing (Balance and Round) ===
        cleaned_balance_text = ""
        cleaned_round_text = ""
        balance_processed_img = None
        round_processed_img = None

        # --- Extract and Process Balance ROI ---
        bx1, by1, bx2, by2 = max(0, bx), max(0, by), min(frame_w, bx + bw), min(frame_h, by + bh)
        if bx2 > bx1 and by2 > by1:
            balance_roi_frame = frame[by1:by2, bx1:bx2]
            balance_gray_roi = cv2.cvtColor(balance_roi_frame, cv2.COLOR_BGR2GRAY)
            balance_thresh_roi = cv2.threshold(balance_gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            balance_processed_img = balance_thresh_roi # For display
            try:
                balance_raw_text = pytesseract.image_to_string(balance_processed_img, config=tesseract_config)
                cleaned_balance_text = re.sub(r'[^\d-]+', '', balance_raw_text).strip()
            except Exception as e:
                print(f"Frame {frame_count}: Warning - Balance OCR Error: {e}")

        # --- Extract and Process Round ROI ---
        rx1, ry1, rx2, ry2 = max(0, rx), max(0, ry), min(frame_w, rx + rw), min(frame_h, ry + rh)
        if rx2 > rx1 and ry2 > ry1:
            round_roi_frame = frame[ry1:ry2, rx1:rx2]
            round_gray_roi = cv2.cvtColor(round_roi_frame, cv2.COLOR_BGR2GRAY)
            round_thresh_roi = cv2.threshold(round_gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            round_processed_img = round_thresh_roi # For display
            try:
                round_raw_text = pytesseract.image_to_string(round_processed_img, config=tesseract_config)
                cleaned_round_text = re.sub(r'[^\d-]+', '', round_raw_text).strip()
            except Exception as e:
                print(f"Frame {frame_count}: Warning - Round OCR Error: {e}")

        # --- OCR Stability Checks ---
        # Balance
        if not cleaned_balance_text: potential_new_balance = None; balance_confirmation_counter = 0
        elif cleaned_balance_text == last_logged_balance_text: potential_new_balance = None; balance_confirmation_counter = 0
        elif cleaned_balance_text == potential_new_balance: balance_confirmation_counter += 1
        else: potential_new_balance = cleaned_balance_text; balance_confirmation_counter = 1
        # Round
        if not cleaned_round_text: potential_new_round = None; round_confirmation_counter = 0
        elif cleaned_round_text == last_logged_round: potential_new_round = None; round_confirmation_counter = 0
        elif cleaned_round_text == potential_new_round: round_confirmation_counter += 1
        else: potential_new_round = cleaned_round_text; round_confirmation_counter = 1

        # --- Update Confirmed Balance Value ---
        if balance_confirmation_counter >= CONFIRMATION_FRAMES_OCR:
            try:
                balance_value = int(potential_new_balance)
                if DECIMAL_CORRECTION and DECIMAL_CORRECTION != 0:
                    last_confirmed_balance_value = balance_value / DECIMAL_CORRECTION
                else:
                    last_confirmed_balance_value = float(balance_value)
            except (ValueError, TypeError):
                print(f"Frame {frame_count}: Warning - Could not convert confirmed balance '{potential_new_balance}' to number.")
            last_logged_balance_text = potential_new_balance
            potential_new_balance = None; balance_confirmation_counter = 0

        # === 2. Stage Detection (Using 5-Class ML Model + Default Logic) ===
        current_detected_stage = None # Reset prediction for this frame
        stages_area_frame = None
        predicted_prob = 0.0
        predicted_known_class = None # The class name among x1-x16 with highest prob

        # --- Extract Stages Area ROI ---
        sx1, sy1, sx2, sy2 = max(0, stages_x), max(0, stages_y), min(frame_w, stages_x + stages_w), min(frame_h, stages_y + stages_h)
        if sx2 > sx1 and sy2 > sy1:
            stages_area_frame = frame[sy1:sy2, sx1:sx2]

            # --- Preprocess the ROI for the model ---
            input_batch = preprocess_image_for_model(stages_area_frame, MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH)

            if input_batch is not None:
                # --- Make Prediction (Model outputs probabilities for the 5 known classes) ---
                try:
                    predictions = model.predict(input_batch, verbose=0) # verbose=0 for less console spam
                    prediction_probabilities = predictions[0] # Probabilities for the known classes

                    # --- Interpret Prediction ---
                    if len(prediction_probabilities) == len(trained_class_names):
                        predicted_index = np.argmax(prediction_probabilities) # Index of the highest prob among the KNOWN classes
                        predicted_prob = prediction_probabilities[predicted_index] # The highest probability value
                        predicted_known_class = trained_class_names[predicted_index] # The name of that class (x1-x16)

                        # --- Apply Confidence Threshold & Default Logic ---
                        if predicted_prob >= PREDICTION_CONFIDENCE_THRESHOLD:
                            # High confidence in one of the trained classes (x1-x16)
                            current_detected_stage = predicted_known_class
                        else:
                            # Low confidence in any of the known classes -> Default to DEFAULT_STAGE
                            current_detected_stage = DEFAULT_STAGE
                    else:
                        print(f"Frame {frame_count}: Warning - Mismatch between model output units ({len(prediction_probabilities)}) "
                              f"and loaded class names ({len(trained_class_names)}). Skipping stage prediction.")

                except Exception as e:
                    print(f"Frame {frame_count}: ERROR during model prediction: {e}")
                    # Decide how to handle prediction errors (e.g., default stage, skip?)
                    current_detected_stage = None # Or DEFAULT_STAGE?


        # --- Stage Stability Check (Works on the result of the ML logic) ---
        if not current_detected_stage: # Includes None and prediction errors
            potential_new_stage = None; stage_confirmation_counter = 0
        elif current_detected_stage == last_logged_stage:
            potential_new_stage = None; stage_confirmation_counter = 0
        elif current_detected_stage == potential_new_stage:
            stage_confirmation_counter += 1
        else:
            potential_new_stage = current_detected_stage; stage_confirmation_counter = 1

        # --- Update Confirmed Stage ---
        if stage_confirmation_counter >= CONFIRMATION_FRAMES_STAGE:
             previous_confirmed_stage = current_confirmed_stage  # Store previous stage before updating
             last_logged_stage = potential_new_stage
             current_confirmed_stage = potential_new_stage
             potential_new_stage = None; stage_confirmation_counter = 0
             
             # Log when stage changes during a round
             if (current_confirmed_stage != previous_confirmed_stage and 
                 previous_confirmed_stage is not None and 
                 last_logged_round is not None):
                timestamp = datetime.datetime.now()
                
                # Create log entry for stage change within round
                log_entry = {
                    'Timestamp': timestamp,
                    'Frame': frame_count,
                    'RoundValue': last_logged_round,
                    'Stage': current_confirmed_stage,
                    'CorrectedBalance': f"{last_confirmed_balance_value:.2f}" if last_confirmed_balance_value is not None else None,
                    'BalanceChange': "0.00",  # No balance change during round
                    'Outcome': "STAGE_CHANGE",
                    'RawBalanceValue': last_logged_balance_text,
                }
                data_log.append(log_entry)
                
                # Print confirmation to console
                balance_display_print = f"{last_confirmed_balance_value:.2f}" if last_confirmed_balance_value is not None else f"(RAW:{last_logged_balance_text})"
                print(f"STAGE CHANGE >> Frame:{frame_count} Round:'{last_logged_round}' Stage:'{current_confirmed_stage}' "
                      f"Balance:{balance_display_print} Change:0.00")
        # === END Stage Detection Section ===


        # === 3. Logging on Confirmed Round Change ===
        if round_confirmation_counter >= CONFIRMATION_FRAMES_OCR:
            timestamp = datetime.datetime.now()
            current_round_text = potential_new_round # The round number just confirmed

            # --- Calculate Balance Change ---
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
                last_round_balance_value = last_confirmed_balance_value
            else:
                 outcome = "BAL_ERR"
                 last_round_balance_value = None

            balance_display_log = f"{last_confirmed_balance_value:.2f}" if last_confirmed_balance_value is not None else None
            balance_display_print = f"{last_confirmed_balance_value:.2f}" if last_confirmed_balance_value is not None else f"(RAW:{last_logged_balance_text})"

            # --- Create Log Entry ---
            log_entry = {
                'Timestamp': timestamp,
                'Frame': frame_count,
                'RoundValue': current_round_text,
                'Stage': current_confirmed_stage, # Log the stage confirmed *at this time*
                'CorrectedBalance': balance_display_log,
                'BalanceChange': f"{balance_change:.2f}",
                'Outcome': outcome,
                'RawBalanceValue': last_logged_balance_text,
                'AccumulatedWin': f"{accumulated_win:.2f}"  # Add accumulated win to log entry
            }
            data_log.append(log_entry)
            
            # --- Print Confirmation to Console ---
            print(f"ROUND CHANGE >> Frame:{frame_count} Round:'{current_round_text}' Stage:'{current_confirmed_stage}' "
                  f"Balance:{balance_display_print} Change:{outcome} Accumulated:{accumulated_win:.2f}")

            # --- Update and Reset Round State ---
            last_logged_round = current_round_text
            potential_new_round = None
            round_confirmation_counter = 0
            previous_confirmed_stage = current_confirmed_stage  # Update previous stage at round change

        # === 4. Display Frame and Information ===
        # --- Draw ROIs ---
        cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)  # Balance (Green)
        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)  # Round (Blue)
        cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), (0, 200, 200), 2) # Stages Area (Teal)

        # --- Prepare Status Text ---
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
        if (not current_confirmed_stage or potential_new_stage) and predicted_known_class:
             raw_pred_text = f"(ML: {predicted_known_class} {predicted_prob:.2f})"
             # Show if it defaulted
             if current_detected_stage == DEFAULT_STAGE and predicted_prob < PREDICTION_CONFIDENCE_THRESHOLD:
                 raw_pred_text += f" -> {DEFAULT_STAGE}"
             status_text += f" {raw_pred_text}"

        # --- Put Text on Frame ---
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 50), 2) # Bright green text

        # --- Show Windows ---
        cv2.imshow("Video Feed", display_frame)
        if balance_processed_img is not None: cv2.imshow("Balance OCR Input", balance_processed_img)
        if round_processed_img is not None: cv2.imshow("Round OCR Input", round_processed_img)
        if stages_area_frame is not None: cv2.imshow("Stages Area Input (ML)", stages_area_frame) # Show what model sees

        # --- Exit Condition ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("INFO: 'q' pressed, exiting.")
            raise KeyboardInterrupt
        # Add other key bindings if needed

except KeyboardInterrupt: # Handle Ctrl+C or 'q'
    print("INFO: Process interrupted by user.")

finally:
    # --- Cleanup ---
    print("INFO: Releasing video capture and destroying windows...")
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    # Add a small delay to ensure windows are closed before final print
    import time
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