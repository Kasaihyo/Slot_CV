import cv2
import pytesseract
import pandas as pd
import datetime
import re
import collections # For keeping track of recent readings

# --- Configuration ---

# 1. Set path to Tesseract executable (Uncomment and modify if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows

# 2. Video Source
# video_path = 'your_video.mp4' # Path to your video file
video_path = 'Videos/high_slot.mov' # Use 0 for the default webcam

# 3. Define the Region of Interest (ROI) coordinates (x, y, width, height)
#    Example: roi = (100, 150, 200, 50)
#    Set roi = None to select interactively on the first frame
balance_roi = (141, 1293, 106, 41)  # Current ROI for balance
round_roi = (1571, 1211, 57, 38)  # Set to None to select interactively on first frame, or provide coordinates

# 4. Tesseract Configuration (adjust for better accuracy)
#    --psm 7: Treat the image as a single text line (often good for single numbers)
#    --psm 8: Treat the image as a single word
#    --psm 6: Assume a single uniform block of text
#    Whitelist: ONLY digits and maybe a minus sign (remove comma/period if they are errors)
tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-' # Using PSM 6 for uniform block of text

# 5. Output file for the data log
output_csv_file = 'ocr_log_stable.csv'

# 6. --- Stability Control ---
# How many consecutive frames must show the *same* new value before logging it?
CONFIRMATION_FRAMES = 2 # Increase this value for more stability (e.g., 5), decrease for faster response

# --- Initialization ---
data_log = []
# Balance tracking
last_logged_balance = None
potential_new_balance = None
balance_confirmation_counter = 0
# Round tracking
last_logged_round = None
potential_new_round = None
round_confirmation_counter = 0

frame_count = 0
previous_balance = None
last_round_balance = None  # Track the balance at the last round change
process_every_n_frames = 15

# --- Win/Loss Analysis Configuration ---
EXPECTED_BET_AMOUNT = 1  # The standard bet amount per round (-1 expected for a loss)
DECIMAL_CORRECTION = 100  # Divide the OCR-read balance by this to fix missing decimal points

# --- Video Processing ---
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video source: {video_path}")
    exit()

# --- ROI Selection (if not predefined) ---
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame to select ROI.")
    cap.release()
    exit()

# First select the round ROI if not defined
if round_roi is None:
    print("\n" + "="*30)
    print(" Select the ROUND Region of Interest (ROI) in the window.")
    print(" Draw a rectangle and press ENTER or SPACE.")
    print(" Press 'c' to cancel selection.")
    print("="*30 + "\n")
    cv2.namedWindow("Select Round ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Round ROI", 800, 600)
    round_roi_coords = cv2.selectROI("Select Round ROI", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Round ROI")
    if round_roi_coords == (0, 0, 0, 0):
        print("Round ROI selection cancelled or invalid. Exiting.")
        cap.release()
        exit()
    round_roi = round_roi_coords
    print(f"Round ROI selected: {round_roi}")

# Then confirm or select the balance ROI if needed
if balance_roi is None:
    print("\n" + "="*30)
    print(" Select the BALANCE Region of Interest (ROI) in the window.")
    print(" Draw a rectangle and press ENTER or SPACE.")
    print(" Press 'c' to cancel selection.")
    print("="*30 + "\n")
    cv2.namedWindow("Select Balance ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Balance ROI", 800, 600)
    balance_roi_coords = cv2.selectROI("Select Balance ROI", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Balance ROI")
    if balance_roi_coords == (0, 0, 0, 0):
        print("Balance ROI selection cancelled or invalid. Exiting.")
        cap.release()
        exit()
    balance_roi = balance_roi_coords
    print(f"Balance ROI selected: {balance_roi}")

# Reopen video source if needed
if isinstance(video_path, str):
    cap.release()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not re-open video source after ROI selection: {video_path}")
        exit()

# Unpack ROI coordinates
bx, by, bw, bh = balance_roi
rx, ry, rw, rh = round_roi

print(f"\nStarting video processing. Press 'q' to quit.")
print(f"Using confirmation threshold: {CONFIRMATION_FRAMES} frames.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        frame_count += 1
        
        # Skip frames to improve performance
        if frame_count % process_every_n_frames != 0:
            continue
            
        frame_h, frame_w = frame.shape[:2]
        
        # --- Process Balance ROI ---
        bx1, by1 = max(0, bx), max(0, by)
        bx2, by2 = min(frame_w, bx + bw), min(frame_h, by + bh)
        if bx2 <= bx1 or by2 <= by1:
            print(f"Warning: Invalid Balance ROI {balance_roi} for frame dimensions {frame_w}x{frame_h}. Skipping.")
            continue
        balance_roi_frame = frame[by1:by2, bx1:bx2]
        
        # --- Process Round ROI ---
        rx1, ry1 = max(0, rx), max(0, ry)
        rx2, ry2 = min(frame_w, rx + rw), min(frame_h, ry + rh)
        if rx2 <= rx1 or ry2 <= ry1:
            print(f"Warning: Invalid Round ROI {round_roi} for frame dimensions {frame_w}x{frame_h}. Skipping.")
            continue
        round_roi_frame = frame[ry1:ry2, rx1:rx2]
        
        # --- Pre-processing Balance ROI for OCR ---
        balance_gray_roi = cv2.cvtColor(balance_roi_frame, cv2.COLOR_BGR2GRAY)
        balance_thresh_roi = cv2.threshold(balance_gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        balance_processed_ocr_image = balance_thresh_roi
        
        # --- Pre-processing Round ROI for OCR ---
        round_gray_roi = cv2.cvtColor(round_roi_frame, cv2.COLOR_BGR2GRAY)
        round_thresh_roi = cv2.threshold(round_gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        round_processed_ocr_image = round_thresh_roi
        
        # --- Perform OCR on Balance ROI ---
        balance_raw_text = ""
        try:
            balance_raw_text = pytesseract.image_to_string(balance_processed_ocr_image, config=tesseract_config)
        except Exception as e:
            print(f"Frame {frame_count}: Error during Balance OCR: {e}")
            
        # --- Perform OCR on Round ROI ---
        round_raw_text = ""
        try:
            round_raw_text = pytesseract.image_to_string(round_processed_ocr_image, config=tesseract_config)
        except Exception as e:
            print(f"Frame {frame_count}: Error during Round OCR: {e}")
        
        # --- Clean OCR Text ---
        cleaned_balance_text = re.sub(r'[^\d-]+', '', balance_raw_text).strip()
        cleaned_round_text = re.sub(r'[^\d-]+', '', round_raw_text).strip()
        
        # --- Stability Check for Balance ---
        if not cleaned_balance_text:
            potential_new_balance = None
            balance_confirmation_counter = 0
        elif cleaned_balance_text == last_logged_balance:
            potential_new_balance = None
            balance_confirmation_counter = 0
        elif cleaned_balance_text == potential_new_balance:
            balance_confirmation_counter += 1
        else:
            potential_new_balance = cleaned_balance_text
            balance_confirmation_counter = 1
            
        # --- Stability Check for Round ---
        if not cleaned_round_text:
            potential_new_round = None
            round_confirmation_counter = 0
        elif cleaned_round_text == last_logged_round:
            potential_new_round = None
            round_confirmation_counter = 0
        elif cleaned_round_text == potential_new_round:
            round_confirmation_counter += 1
        else:
            potential_new_round = cleaned_round_text
            round_confirmation_counter = 1

        # --- Log Confirmed Balance Change ---
        # We'll still track balance changes, but not log them independently
        if balance_confirmation_counter >= CONFIRMATION_FRAMES:
            # Record the confirmed balance, but don't log it yet
            try:
                # Parse the integer value from OCR
                raw_balance = int(potential_new_balance)
                
                # Apply decimal correction (divide by 100)
                current_balance = raw_balance / DECIMAL_CORRECTION
                
                previous_balance = current_balance
            except ValueError:
                pass
            
            last_logged_balance = potential_new_balance
            potential_new_balance = None
            balance_confirmation_counter = 0

        # --- Log Confirmed Round Change ---
        if round_confirmation_counter >= CONFIRMATION_FRAMES:
            timestamp = datetime.datetime.now()
            
            # Calculate balance change only at round changes
            balance_change = 0
            outcome = "0.00"
            corrected_balance = None
            
            if last_logged_balance and last_logged_balance.isdigit():
                try:
                    # Get current balance
                    current_raw_balance = int(last_logged_balance)
                    current_balance = current_raw_balance / DECIMAL_CORRECTION
                    corrected_balance = current_balance
                    
                    # If we have a previous round's balance, calculate the change
                    if last_round_balance is not None:
                        balance_change = current_balance - last_round_balance
                        outcome = f"{balance_change:.2f}"
                    else:
                        # First round, no previous balance to compare
                        balance_change = 0
                        outcome = "START"
                        
                    # Update the balance for this round
                    last_round_balance = current_balance
                except ValueError:
                    print(f"Error processing balance at round change: {last_logged_balance}")
            
            # Fixed f-string formatting
            balance_display = f"{corrected_balance:.2f}" if corrected_balance is not None else "N/A"
            print(f"CONFIRMED Round Change! Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}, "
                  f"Frame: {frame_count}, Round: '{potential_new_round}', "
                  f"Balance: {balance_display}, Change: {outcome}")
            
            # Log the round and balance change together
            round_log_entry = {
                'Timestamp': timestamp,
                'Frame': frame_count,
                'RoundValue': potential_new_round,
                'RawBalanceValue': last_logged_balance,
                'CorrectedBalance': f"{corrected_balance:.2f}" if corrected_balance is not None else None,
                'BalanceChange': balance_change,
                'Outcome': outcome,
            }
            data_log.append(round_log_entry)
            
            # Update the last logged round
            last_logged_round = potential_new_round
            potential_new_round = None
            round_confirmation_counter = 0

        # --- Display Output ---
        # Draw ROIs on the frame
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)  # Balance ROI in green
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)  # Round ROI in blue
        
        # Prepare display text
        display_text = f"Round: {last_logged_round}"
        
        # Show corrected balance if available
        if last_logged_balance and last_logged_balance.isdigit():
            try:
                corrected_balance = int(last_logged_balance) / DECIMAL_CORRECTION
                display_text += f" | Balance: {corrected_balance:.2f}"
            except ValueError:
                display_text += f" | Balance: {last_logged_balance}"
        else:
            display_text += f" | Balance: {last_logged_balance}"
        
        # Show potential values if in confirmation process
        if potential_new_balance:
            display_text += f" | Potential Balance: {potential_new_balance} ({balance_confirmation_counter}/{CONFIRMATION_FRAMES})"
        if potential_new_round:
            display_text += f" | Potential Round: {potential_new_round} ({round_confirmation_counter}/{CONFIRMATION_FRAMES})"
            
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display frames
        cv2.imshow("Video Feed", frame)
        cv2.imshow("Balance ROI", balance_processed_ocr_image)
        cv2.imshow("Round ROI", round_processed_ocr_image)

        # --- Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

finally:
    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()

    # --- Save Data Log ---
    if data_log:
        df = pd.DataFrame(data_log)
        try:
            df.to_csv(output_csv_file, index=False)
            print(f"\nData log saved to '{output_csv_file}'")
            print("\n--- Log Summary ---")
            print(df)
            print("------------------")
        except Exception as e:
            print(f"\nError saving data log to CSV: {e}")
            print("\n--- Log Data (not saved) ---")
            for entry in data_log:
                print(entry)
            print("-------------------------")
    else:
        print("\nNo confirmed changes were detected or logged.")