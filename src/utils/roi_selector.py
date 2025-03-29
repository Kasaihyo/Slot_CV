# src/utils/roi_selector.py
import cv2
import datetime
import sys

def select_roi_interactively(window_title, frame_to_select_on):
    """Handles interactive ROI selection."""
    print(f"\nINFO: Select '{window_title}' in the window.")
    print("Draw a rectangle and press ENTER or SPACE.")
    print("Press 'c' to cancel selection (will exit script).")

    # Use a unique window name to avoid conflicts
    temp_window_name = f"{window_title}_{datetime.datetime.now().timestamp()}"
    cv2.namedWindow(temp_window_name, cv2.WINDOW_NORMAL)
    # Try to make window reasonably sized
    try:
        screen_height = 1080 # Assume a common screen height or get dynamically if needed
        scale_factor = min(1.0, 800 / frame_to_select_on.shape[1], 600 / frame_to_select_on.shape[0])
        display_width = int(frame_to_select_on.shape[1] * scale_factor)
        display_height = int(frame_to_select_on.shape[0] * scale_factor)
        cv2.resizeWindow(temp_window_name, display_width, display_height)
    except Exception:
         cv2.resizeWindow(temp_window_name, 800, 600) # Fallback size

    roi_coords = cv2.selectROI(temp_window_name, frame_to_select_on, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(temp_window_name) # Destroy only the temp window

    if roi_coords == (0, 0, 0, 0):
        print(f"ERROR: ROI selection cancelled for '{window_title}'. Exiting.")
        sys.exit(1) # Exit script if user cancels

    print(f"INFO: ROI for '{window_title}' selected: {roi_coords}")
    return roi_coords