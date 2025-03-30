import cv2
import sys
import datetime
import time
import queue
import threading
import numpy as np
import tensorflow as tf
import pandas as pd # Used only for final log display settings
import os

# Import configurations and modules
import config
from src.utils.roi_selector import select_roi_interactively
from src.video.reader import VideoReader
from src.processing.processor import FrameProcessor
from src.logging import manager as log_manager
from src.state.tracker import GameState

def load_model_and_classes(model_path, class_names_path):
    """Loads the Keras model and class names."""
    model = None
    trained_class_names = []

    # Load Model
    print(f"INFO: Loading stage classification model from {model_path}...")
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("INFO: Model loaded successfully.")
        # model.summary() # Optional: Print model summary
    except Exception as e:
        print(f"ERROR: Failed to load model '{model_path}'. "
              f"Ensure the file exists and TensorFlow is installed correctly. Exiting. Error: {e}")
        sys.exit(1)

    # Load Class Names (corresponding to model output indices)
    print(f"INFO: Loading TRAINED class names from {class_names_path}...")
    try:
        if not os.path.exists(class_names_path):
             raise FileNotFoundError(f"Class names file not found at {class_names_path}")
        with open(class_names_path, "r") as f:
            trained_class_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"INFO: Loaded TRAINED class names: {trained_class_names}")

        # Validation
        if not trained_class_names:
             print("ERROR: Class names file is empty. Exiting.")
             sys.exit(1)
        if len(trained_class_names) != model.output_shape[-1]:
            print(f"WARNING: Number of loaded class names ({len(trained_class_names)}) "
                  f"does not match model output units ({model.output_shape[-1]}). "
                  "Ensure class names file corresponds to the loaded model.")
            # Decide whether to exit or continue with caution
            # sys.exit(1)

    except Exception as e:
        print(f"ERROR: Failed to load class names '{class_names_path}'. Exiting. Error: {e}")
        sys.exit(1)

    return model, trained_class_names

def main():
    """Main function to run the game analysis application."""

    # --- Initial Setup ---
    print("--- Game Analyzer Startup ---")

    # Load ML Model and Class Names
    model, trained_class_names = load_model_and_classes(config.MODEL_PATH, config.MODEL_LABELS_PATH)

    # --- Video Source and ROI Setup ---
    print(f"INFO: Opening video source for details: {config.VIDEO_PATH}")
    temp_cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not temp_cap.isOpened():
        print(f"ERROR: Could not open video source: {config.VIDEO_PATH}. Exiting.")
        sys.exit(1)
    ret, first_frame = temp_cap.read()
    if not ret or first_frame is None:
        print("ERROR: Could not read the first frame from video source. Exiting.")
        temp_cap.release()
        sys.exit(1)
    frame_h_orig, frame_w_orig = first_frame.shape[:2]
    print(f"INFO: Video frame dimensions: {frame_w_orig}x{frame_h_orig}")
    temp_cap.release()

    # Make local copies of ROIs from config to allow interactive modification
    balance_roi = config.BALANCE_ROI
    round_roi = config.ROUND_ROI
    stages_area_roi = config.STAGES_AREA_ROI

    # Interactive ROI Selection (if any ROI is None)
    if stages_area_roi is None:
        stages_area_roi = select_roi_interactively("Stages Area ROI (Cover ALL x1-x32)", first_frame)
    if round_roi is None:
        round_roi = select_roi_interactively("Round Number ROI", first_frame)
    if balance_roi is None:
        balance_roi = select_roi_interactively("Balance Number ROI", first_frame)

    # Prepare configuration dictionary for the processor thread
    processor_config = {
        'balance_roi': balance_roi,
        'round_roi': round_roi,
        'stages_area_roi': stages_area_roi,
        'model_img_height': config.MODEL_IMG_HEIGHT,
        'model_img_width': config.MODEL_IMG_WIDTH,
        'prediction_confidence_threshold': config.PREDICTION_CONFIDENCE_THRESHOLD,
        'default_stage': config.DEFAULT_STAGE,
        'process_every_n_frames': config.PROCESS_EVERY_N_FRAMES,
    }

    # --- Initialization ---
    data_log = log_manager.initialize_log()
    game_state = GameState()

    frame_queue = queue.Queue(maxsize=config.FRAME_QUEUE_MAXSIZE)
    # Limit the results queue size to prevent unbounded memory growth
    results_queue_maxsize = config.FRAME_QUEUE_MAXSIZE * 2
    results_queue = queue.Queue(maxsize=results_queue_maxsize)
    print(f"INFO: Initialized results queue with maxsize={results_queue_maxsize}") # Added info log
    stop_event = threading.Event()

    reader_thread = None
    processor_thread = None
    video_capture = None # For single-threaded mode

    # --- Start Threads or Prepare Single-Threaded Mode ---
    if config.USE_PROCESSING_THREAD:
        print("INFO: Starting in multi-threaded mode.")
        reader_thread = VideoReader(config.VIDEO_PATH, frame_queue, stop_event, config.FRAME_QUEUE_MAXSIZE)
        processor_thread = FrameProcessor(frame_queue, results_queue, stop_event, processor_config, model, trained_class_names)
        reader_thread.start()
        processor_thread.start()
        print("INFO: Reader and Processor threads started.")
    else:
        print("INFO: Starting in single-threaded mode.")
        video_capture = cv2.VideoCapture(config.VIDEO_PATH)
        if not video_capture.isOpened():
            print(f"ERROR: Could not open video source for single-threaded mode: {config.VIDEO_PATH}. Exiting.")
            sys.exit(1)
        # Create a dummy processor instance just to use its processing logic
        # Note: This isn't ideal, refactoring processor logic into functions would be better for single-thread
        dummy_processor = FrameProcessor(None, None, stop_event, processor_config, model, trained_class_names)
        current_frame_num = 0

    # --- Main Loop (Handles Results, State, Display) ---
    print("\n" + "="*40)
    print("INFO: Starting main processing loop...")
    # Print key config settings
    print(f" - Mode: {'Multi-Threaded' if config.USE_PROCESSING_THREAD else 'Single-Threaded'}")
    print(f" - Model: {os.path.basename(config.MODEL_PATH)}")
    print(f" - Trained Classes: {trained_class_names}")
    print(f" - Default Stage (Low Confidence): {config.DEFAULT_STAGE}")
    print(f" - Confidence Threshold: {config.PREDICTION_CONFIDENCE_THRESHOLD:.2f}")
    print(f" - OCR Stability: {config.CONFIRMATION_FRAMES_OCR} frames")
    print(f" - Stage Stability: {config.CONFIRMATION_FRAMES_STAGE} frames")
    print(f" - Processing every {config.PROCESS_EVERY_N_FRAMES} frames")
    print(f"Press 'q' in the '{config.DISPLAY_WINDOW_NAME}' window to quit.")
    print("="*40 + "\n")

    latest_display_frame = first_frame.copy()
    latest_balance_img = None
    latest_round_img = None
    latest_stage_img = None
    last_processed_frame_num = 0
    last_ml_pred_info = "" # Store last raw ML prediction details
    latest_stable_grid_frame = None # To store the annotated frame

    try:
        while not stop_event.is_set():
            result_data = None

            # --- Get Processed Frame Data ---
            if config.USE_PROCESSING_THREAD:
                try:
                    # Get results from the processor thread, with timeout
                    result_data = results_queue.get(block=True, timeout=0.1) # Short block
                    if result_data is None: # End signal from processor
                        print("INFO: Main loop received None signal from processor.")
                        stop_event.set()
                        break
                except queue.Empty:
                    # No new results, main loop can check stop event or briefly sleep
                    if not reader_thread.is_alive() or not processor_thread.is_alive():
                         if not stop_event.is_set(): # If they stopped unexpectedly
                              print("ERROR: A processing thread has stopped unexpectedly. Setting stop event.")
                              stop_event.set()
                         break # Exit main loop if threads are dead
                    time.sleep(0.005) # Small sleep to prevent busy-waiting
                    continue # Go to next loop iteration
            else: # Single-threaded mode
                if video_capture and video_capture.isOpened():
                    ret, frame = video_capture.read()
                    if not ret or frame is None:
                        print("INFO: End of video or read error in single-threaded mode.")
                        stop_event.set()
                        break
                    current_frame_num += 1

                    # Process frame directly if it's the Nth frame
                    if current_frame_num % config.PROCESS_EVERY_N_FRAMES == 0:
                        # Simulate processing (needs refactor of FrameProcessor.run)
                        # This is a simplified placeholder - ideally, processor logic is callable functions
                        # For now, we'll assume the logic is complex and skip full single-thread impl.
                        # result_data = dummy_processor.process_single_frame(current_frame_num, frame) # Imaginary method
                         print(f"Warning: Single-threaded processing logic not fully implemented in refactor. Frame: {current_frame_num}")
                         result_data = {'frame_num': current_frame_num, 'skipped': False, 'display_frame': frame.copy()}
                         # TODO: Add actual calls to OCR/ML functions here for single-thread
                    else:
                         result_data = {'frame_num': current_frame_num, 'skipped': True, 'display_frame': frame}
                else:
                     stop_event.set()
                     break

            # --- Process the Received Result ---
            if result_data:
                frame_num = result_data['frame_num']
                last_processed_frame_num = frame_num
                # Always update the display frame
                latest_display_frame = result_data.get('display_frame', latest_display_frame) # Use latest frame

                if not result_data.get('skipped', True):
                    # Update debug images if available and enabled
                    if config.SHOW_DEBUG_WINDOWS:
                        latest_balance_img = result_data.get('balance_processed_img')
                        latest_round_img = result_data.get('round_processed_img')
                        latest_stage_img = result_data.get('stages_area_frame')

                    # --- Check for Stable Grid Frame ---
                    if result_data.get('grid_event') == 'stable':
                        latest_stable_grid_frame = result_data.get('annotated_stable_frame')
                    # -----------------------------------

                    # Store raw ML prediction details for display
                    pred_cls = result_data.get('stage_known_cls')
                    pred_prob = result_data.get('stage_prob', 0.0)
                    detected_st = result_data.get('detected_stage')
                    if pred_cls:
                        last_ml_pred_info = f"(ML: {pred_cls} {pred_prob:.2f})"
                        if detected_st == config.DEFAULT_STAGE and pred_prob < config.PREDICTION_CONFIDENCE_THRESHOLD:
                           last_ml_pred_info += f" -> {config.DEFAULT_STAGE}"
                    else:
                        last_ml_pred_info = "" # Reset if no prediction this frame

                    # Update Game State using the State Tracker
                    round_changed, stage_changed_mid_round = game_state.update_from_result(result_data)

                    # --- Logging ---
                    # Log round change IF it occurred
                    if round_changed:
                        log_data = game_state.get_log_data(frame_num)
                        trigger_retry = False # Flag to check if retry is needed
                        initial_grid_found = log_data.get('stable_grid_found', False)
                        initial_symbols_zero = log_data.get('total_symbols', 0) == 0

                        if config.ENABLE_GRID_RETRY and (not initial_grid_found or (initial_grid_found and initial_symbols_zero)):
                            trigger_retry = True
                            if not initial_grid_found:
                                print(f"INFO F{frame_num}: Round change detected, stable grid missed. Initiating retry...")
                            elif initial_symbols_zero:
                                print(f"INFO F{frame_num}: Round change detected, stable grid found BUT 0 symbols counted (F:{log_data.get('stable_grid_frame_num')}). Initiating retry...")

                        # --- Grid Retry Logic --- #
                        if trigger_retry:
                            start_retry_frame, end_retry_frame = game_state.get_previous_state_frame_range('ROUND_CHANGE')
                            if start_retry_frame is not None and end_retry_frame is not None:
                                grid_found_on_retry = False
                                for attempt in range(config.GRID_RETRY_ATTEMPTS):
                                    print(f"  Retry Attempt {attempt + 1}/{config.GRID_RETRY_ATTEMPTS} for frames {start_retry_frame}-{end_retry_frame}")
                                    frames_to_retry = reader_thread.get_frames_from_buffer(start_retry_frame, end_retry_frame)
                                    if frames_to_retry:
                                        retry_result = processor_thread.find_stable_grid_in_frames(frames_to_retry)
                                        # Check if retry found grid AND counted symbols
                                        if retry_result.get('stable_grid_found') and retry_result.get('total_symbols', 0) > 0:
                                            print(f"  Retry SUCCESS: Found stable grid with {retry_result.get('total_symbols')} symbols at frame {retry_result.get('stable_grid_frame_num')}")
                                            log_data.update(retry_result)
                                            grid_found_on_retry = True
                                            break
                                        elif retry_result.get('stable_grid_found'):
                                            print(f"  Retry Note: Found stable grid at {retry_result.get('stable_grid_frame_num')} but 0 symbols counted. Continuing attempts...")
                                        # Else: find_stable_grid_in_frames already prints failure message
                                    else:
                                        print(f"  Retry Warning: Could not retrieve frames {start_retry_frame}-{end_retry_frame} from buffer for attempt {attempt + 1}.")
                                        break
                                if not grid_found_on_retry:
                                    print(f"  Retry FAILED to find a valid grid after {config.GRID_RETRY_ATTEMPTS} attempts.")
                                    # Log failure explicitly ONLY if retry was attempted and failed
                                    log_data['stable_grid_found'] = False # Add the key to indicate failed retry
                            else:
                                # ... (print frame range warning) ...
                                # If range failed, still mark grid as not found if it wasn't initially
                                if not initial_grid_found:
                                    log_data['stable_grid_found'] = False
                        # --- End Grid Retry Logic --- #

                        # Log the final data (may include grid_found=False only if retry failed)
                        entry = log_manager.add_log_entry(data_log, log_data)
                        # Updated print statement (already done in previous step)
                        balance_print = f"{log_data['confirmed_balance_val']:.2f}" if log_data['confirmed_balance_val'] is not None else f"(RAW:{log_data['raw_balance_text']})"
                        grid_info_print = f" Grid Symbols: {log_data.get('total_symbols')} (F:{log_data.get('stable_grid_frame_num')})" if log_data.get('stable_grid_found') else " Grid Missed"
                        print(f"ROUND CHANGE >> F:{frame_num} R:'{log_data['current_round']}' S:'{log_data['current_stage']}' "
                              f"B:{balance_print} Change:{log_data['outcome_str']} Accum:{log_data['accumulated_win_val']:.2f}{grid_info_print}")

                    # Log stage change IF it occurred
                    elif stage_changed_mid_round:
                         log_data = game_state.get_log_data_stage_change(frame_num)
                         trigger_retry = False # Flag to check if retry is needed
                         initial_grid_found = log_data.get('stable_grid_found', False)
                         initial_symbols_zero = log_data.get('total_symbols', 0) == 0

                         if config.ENABLE_GRID_RETRY and (not initial_grid_found or (initial_grid_found and initial_symbols_zero)):
                             trigger_retry = True
                             if not initial_grid_found:
                                 print(f"INFO F{frame_num}: Stage change detected mid-round, stable grid missed. Initiating retry...")
                             elif initial_symbols_zero:
                                 print(f"INFO F{frame_num}: Stage change detected mid-round, stable grid found BUT 0 symbols counted (F:{log_data.get('stable_grid_frame_num')}). Initiating retry...")

                         # --- Grid Retry Logic --- #
                         if trigger_retry:
                            start_retry_frame, end_retry_frame = game_state.get_previous_state_frame_range('STAGE_CHANGE')
                            if start_retry_frame is not None and end_retry_frame is not None:
                                grid_found_on_retry = False
                                for attempt in range(config.GRID_RETRY_ATTEMPTS):
                                    print(f"  Retry Attempt {attempt + 1}/{config.GRID_RETRY_ATTEMPTS} for frames {start_retry_frame}-{end_retry_frame}")
                                    frames_to_retry = reader_thread.get_frames_from_buffer(start_retry_frame, end_retry_frame)
                                    if frames_to_retry:
                                        retry_result = processor_thread.find_stable_grid_in_frames(frames_to_retry)
                                        # Check if retry found grid AND counted symbols
                                        if retry_result.get('stable_grid_found') and retry_result.get('total_symbols', 0) > 0:
                                            print(f"  Retry SUCCESS: Found stable grid with {retry_result.get('total_symbols')} symbols at frame {retry_result.get('stable_grid_frame_num')}")
                                            log_data.update(retry_result)
                                            grid_found_on_retry = True
                                            break
                                        elif retry_result.get('stable_grid_found'):
                                            print(f"  Retry Note: Found stable grid at {retry_result.get('stable_grid_frame_num')} but 0 symbols counted. Continuing attempts...")
                                        # Else: find_stable_grid_in_frames already prints failure message
                                    else:
                                        print(f"  Retry Warning: Could not retrieve frames {start_retry_frame}-{end_retry_frame} from buffer for attempt {attempt + 1}.")
                                        break
                                if not grid_found_on_retry:
                                    print(f"  Retry FAILED to find a valid grid after {config.GRID_RETRY_ATTEMPTS} attempts.")
                                    # Log failure explicitly ONLY if retry was attempted and failed
                                    log_data['stable_grid_found'] = False # Add the key to indicate failed retry
                            else:
                                # ... (print frame range warning) ...
                                # If range failed, still mark grid as not found if it wasn't initially
                                if not initial_grid_found:
                                    log_data['stable_grid_found'] = False
                         # --- End Grid Retry Logic --- #

                         # Log the final data (may include grid_found=False only if retry failed)
                         entry = log_manager.add_log_entry(data_log, log_data)
                         # Updated print statement (already done)
                         balance_print = f"{log_data['confirmed_balance_val']:.2f}" if log_data['confirmed_balance_val'] is not None else f"(RAW:{log_data['raw_balance_text']})"
                         grid_info_print = f" Grid Symbols: {log_data.get('total_symbols')} (F:{log_data.get('stable_grid_frame_num')})" if log_data.get('stable_grid_found') else " Grid Missed"
                         print(f"STAGE CHANGE >> F:{frame_num} R:'{log_data['current_round']}' S:'{log_data['current_stage']}' "
                               f"B:{balance_print} Change:0.00 Accum:{log_data['accumulated_win_val']:.2f}{grid_info_print}")

            # --- Display Frame and Information ---
            if latest_display_frame is not None:
                display_copy = latest_display_frame.copy()
                frame_h, frame_w = display_copy.shape[:2]

                # Draw ROIs (use the potentially updated ROI variables)
                # Function to draw ROI safely
                def draw_roi(img, roi, color, label):
                     if roi:
                         x, y, w, h = roi
                         x1, y1 = max(0, x), max(0, y)
                         x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
                         if x2 > x1 and y2 > y1:
                             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                             cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                draw_roi(display_copy, balance_roi, (0, 255, 0), "Balance")  # Green
                draw_roi(display_copy, round_roi, (255, 0, 0), "Round")    # Blue
                draw_roi(display_copy, stages_area_roi, (0, 200, 200), "Stages") # Teal

                # Get status text from GameState
                status_line = game_state.get_display_status()

                # Add raw ML prediction if relevant
                if last_ml_pred_info and (not game_state.last_confirmed_stage or game_state.stage_tracker.get_potential()):
                    status_line += f" {last_ml_pred_info}"


                # Put Text on Frame
                cv2.putText(display_copy, f"Frame: {last_processed_frame_num}", (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(display_copy, status_line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 50), 2) # Bright green text

                # Show Windows
                cv2.imshow(config.DISPLAY_WINDOW_NAME, display_copy)
                if config.SHOW_DEBUG_WINDOWS and not result_data.get('skipped', True):
                    if latest_balance_img is not None: cv2.imshow(config.BALANCE_WINDOW_NAME, latest_balance_img)
                    if latest_round_img is not None: cv2.imshow(config.ROUND_WINDOW_NAME, latest_round_img)
                    if latest_stage_img is not None: cv2.imshow(config.STAGE_WINDOW_NAME, latest_stage_img)
                # Show the stable grid window if we have a frame
                if latest_stable_grid_frame is not None:
                    cv2.imshow(config.STABLE_GRID_WINDOW_NAME, latest_stable_grid_frame)

            # --- Exit Condition ---
            key = cv2.waitKey(1) & 0xFF # Wait minimally (1ms)
            if key == ord('q'):
                print("INFO: 'q' pressed, initiating shutdown.")
                stop_event.set() # Signal threads to stop
                break

    except KeyboardInterrupt: # Handle Ctrl+C
        print("INFO: Process interrupted by user (Ctrl+C). Initiating shutdown.")
        stop_event.set()
    except Exception as e:
         print(f"\nFATAL ERROR in main loop: {e}")
         import traceback
         traceback.print_exc()
         stop_event.set() # Try to shutdown gracefully

    finally:
        # --- Cleanup ---
        print("\nINFO: Shutting down...")
        stop_event.set() # Ensure stop event is set

        if config.USE_PROCESSING_THREAD:
            print("INFO: Waiting for threads to finish...")
            if reader_thread and reader_thread.is_alive():
                reader_thread.join(timeout=2.0)
                if reader_thread.is_alive(): print("Warning: VideoReader thread did not stop gracefully.")
            if processor_thread and processor_thread.is_alive():
                # Ensure processor gets the None signal if reader finished abruptly
                try: results_queue.put_nowait(None)
                except queue.Full: pass
                processor_thread.join(timeout=2.0)
                if processor_thread.is_alive(): print("Warning: FrameProcessor thread did not stop gracefully.")
        elif video_capture:
            video_capture.release()
            print("INFO: Video capture released (single-threaded mode).")

        cv2.destroyAllWindows()
        # Add a small delay to ensure windows are closed before final print
        time.sleep(0.2)
        print("INFO: Display windows closed.")

        # --- Save Data Log ---
        if data_log:
            try:
                # Ensure output directory exists
                os.makedirs(config.OUTPUT_DIR, exist_ok=True)
                # Generate timestamped filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"game_log_{timestamp}.csv"
                full_output_path = os.path.join(config.OUTPUT_DIR, output_filename)
                
                log_manager.save_log_to_csv(data_log, full_output_path)
            except Exception as e:
                print(f"ERROR: Failed to save log file: {e}")
        else:
            print("INFO: No log data generated to save.")

        print("\nINFO: Script finished.")


# --- Entry Point ---
if __name__ == "__main__":
    main()