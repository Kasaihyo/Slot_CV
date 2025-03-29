import cv2
import sys
import numpy as np

# --- Configuration ---
# Option 1: Path to your video file
# Make sure this path is correct!
# video_source = 'your_video.mp4'

# Option 2: Use the default webcam (usually 0)
# If you want to select ROI from a live feed
video_source = 'Videos/Sequence 01.mp4'

# --- Main Logic ---

print(f"Attempting to open video source: {video_source}")
cap = cv2.VideoCapture(video_source)

# 1. Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video source '{video_source}'")
    print("Please check the path or camera index.")
    sys.exit() # Exit the script if the source can't be opened

# 2. Read the first frame
ret, frame = cap.read()

# 3. Check if frame was read successfully
if not ret:
    print("Error: Could not read the first frame from the video source.")
    cap.release() # Release the source before exiting
    sys.exit()

# Create a copy of the original frame for display
original_frame = frame.copy()
display_frame = frame.copy()
selection_complete = False
roi_coords = (0, 0, 0, 0)

# Variables for zoom and pan functionality
zoom_scale = 1.0
offset_x, offset_y = 0, 0
dragging = False
drag_start_x, drag_start_y = 0, 0
previous_offset_x, previous_offset_y = 0, 0

# Add instructions on the frame
def add_instructions(img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    instructions = [
        "ROI Selection Instructions:",
        "1. Click and drag to select region",
        "2. Press ENTER/SPACE to confirm",
        "3. Press 'r' to reset selection",
        "4. Press 'c' to cancel and exit",
        "5. Scroll mouse wheel to zoom in/out",
        "6. Right-click and drag to move around"
    ]
    
    # Add semi-transparent overlay for text background
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    y_pos = 40
    for line in instructions:
        cv2.putText(img, line, (20, y_pos), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y_pos += 25
    
    # Display zoom level
    cv2.putText(img, f"Zoom: {zoom_scale:.1f}x", (20, y_pos + 10), font, 0.6, (100, 255, 100), 1, cv2.LINE_AA)
    
    return img

# Function to show preview of selected ROI
def show_roi_preview(img, roi):
    x, y, w, h = roi
    if w > 0 and h > 0:
        # Draw rectangle on the main image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Create a preview window in the corner
        preview = img[y:y+h, x:x+w].copy()
        
        # Resize preview if it's too large
        max_preview_size = 200
        preview_h, preview_w = preview.shape[:2]
        if preview_h > max_preview_size or preview_w > max_preview_size:
            scale = max_preview_size / max(preview_h, preview_w)
            preview = cv2.resize(preview, (int(preview_w * scale), int(preview_h * scale)))
        
        # Create preview background
        h_prev, w_prev = preview.shape[:2]
        cv2.rectangle(img, (img.shape[1]-w_prev-20, 20), (img.shape[1]-10, 30+h_prev), (0, 0, 0), -1)
        
        # Add preview to corner
        img[30:30+h_prev, img.shape[1]-10-w_prev:img.shape[1]-10] = preview
        
        # Add preview label
        cv2.putText(img, "ROI Preview", (img.shape[1]-w_prev-15, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img

# Function to apply zoom and pan to the frame
def apply_zoom_and_pan(frame, scale, offset_x, offset_y):
    height, width = frame.shape[:2]
    
    # Calculate new dimensions and center point
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Calculate visible region
    x_start = max(0, int(offset_x))
    y_start = max(0, int(offset_y))
    x_end = min(width, int(offset_x + width/scale))
    y_end = min(height, int(offset_y + height/scale))
    
    # Extract visible portion
    visible = frame[y_start:y_end, x_start:x_end].copy()
    
    # Resize to fit display
    return cv2.resize(visible, (width, height))

# Mouse callback function for zoom and pan
def mouse_callback(event, x, y, flags, param):
    global zoom_scale, offset_x, offset_y, dragging, drag_start_x, drag_start_y, previous_offset_x, previous_offset_y, display_frame
    
    if event == cv2.EVENT_MOUSEWHEEL:
        # Get wheel direction (positive or negative)
        wheel_direction = flags > 0
        
        # Scale factor for zoom
        scale_factor = 1.1
        
        # Apply zoom
        if wheel_direction:  # Zoom in
            zoom_scale *= scale_factor
        else:  # Zoom out
            zoom_scale /= scale_factor
            
        # Limit zoom scale
        zoom_scale = max(1.0, min(zoom_scale, 10.0))
        
        # Update display with new zoom
        display_frame = original_frame.copy()
        display_frame = apply_zoom_and_pan(display_frame, zoom_scale, offset_x, offset_y)
        display_frame = add_instructions(display_frame)
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Start dragging (right mouse button)
        dragging = True
        drag_start_x, drag_start_y = x, y
        previous_offset_x, previous_offset_y = offset_x, offset_y
    
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        # Calculate new offset while dragging
        dx = (x - drag_start_x) / zoom_scale
        dy = (y - drag_start_y) / zoom_scale
        offset_x = previous_offset_x - dx
        offset_y = previous_offset_y - dy
        
        # Constrain offset to prevent going out of bounds
        h, w = original_frame.shape[:2]
        max_offset_x = max(0, w - w/zoom_scale)
        max_offset_y = max(0, h - h/zoom_scale)
        offset_x = max(0, min(offset_x, max_offset_x))
        offset_y = max(0, min(offset_y, max_offset_y))
        
        # Update display with new pan position
        display_frame = original_frame.copy()
        display_frame = apply_zoom_and_pan(display_frame, zoom_scale, offset_x, offset_y)
        display_frame = add_instructions(display_frame)
    
    elif event == cv2.EVENT_RBUTTONUP:
        # Stop dragging
        dragging = False

# Add instructions
display_frame = add_instructions(display_frame)

# Set up mouse callback
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", mouse_callback)

# Main selection loop
while not selection_complete:
    # Show the frame for selection
    cv2.imshow("Select ROI", display_frame)
    
    # Create a working copy for ROI selection
    working_frame = display_frame.copy()
    
    # Let user select the ROI
    roi_coords = cv2.selectROI("Select ROI", working_frame, fromCenter=False, showCrosshair=True)
    
    # Convert ROI coordinates back to original frame coordinates when zoomed
    if zoom_scale > 1.0:
        x, y, w, h = roi_coords
        orig_x = int(x / zoom_scale + offset_x)
        orig_y = int(y / zoom_scale + offset_y)
        orig_w = int(w / zoom_scale)
        orig_h = int(h / zoom_scale)
        orig_roi = (orig_x, orig_y, orig_w, orig_h)
    else:
        orig_roi = roi_coords
    
    # Create a new display frame with the ROI highlighted
    display_frame = apply_zoom_and_pan(original_frame, zoom_scale, offset_x, offset_y)
    display_frame = add_instructions(display_frame)
    display_frame = show_roi_preview(display_frame, roi_coords)
    
    # Show result and ask for confirmation
    cv2.imshow("Confirm Selection", display_frame)
    
    # Wait for key to confirm, reset, or cancel
    key = cv2.waitKey(0) & 0xFF
    
    if key == 13 or key == 32:  # ENTER or SPACE key
        selection_complete = True
        roi_coords = orig_roi  # Use the coordinates mapped to original frame
    elif key == ord('r'):  # 'r' key to reset
        display_frame = apply_zoom_and_pan(original_frame, zoom_scale, offset_x, offset_y)
        display_frame = add_instructions(display_frame)
        cv2.destroyWindow("Confirm Selection")
    elif key == ord('c'):  # 'c' key to cancel
        roi_coords = (0, 0, 0, 0)
        selection_complete = True

# Destroy all windows
cv2.destroyAllWindows()

# Release the video capture object
cap.release()

# Process and display the results
if roi_coords == (0, 0, 0, 0):
    print("ROI selection cancelled by user.")
else:
    print("-" * 40)
    print("ROI Selection Complete!")
    print(f"Selected Coordinates (x, y, width, height): {roi_coords}")
    print("-" * 40)
    print("\nCopy the tuple above (including the parentheses) and paste it")
    print("into the main OCR script like this:")
    print(f"\nroi = {roi_coords}\n")

print("ROI selection script finished.")