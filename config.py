# config.py
import os

# --- Project Root (calculated) ---
# Assuming config.py is in the root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Paths ---
VIDEO_PATH = os.path.join(PROJECT_ROOT, '/Users/temogiorgadze/Documents/FluxGaming/CV_slots/Data/Videos/Sequence 01.mp4') # Path to your video file
# video_path = 0 # Use 0 for default webcam
# === Logging and Output ===
# OUTPUT_CSV_FILE = "game_log.csv"      # Path to save the game log CSV file
OUTPUT_DIR = "Output/"              # Directory to save the timestamped game log CSV files
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/stage_classifier_model_5class.keras')
MODEL_LABELS_PATH = os.path.join(PROJECT_ROOT, 'models/stage_class_names_5class.txt')
# Tesseract executable path (Uncomment and modify if needed, otherwise it relies on PATH)
# TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows
TESSERACT_CMD = None # Set to None if Tesseract is in your system PATH

# Remove path for round digit templates
# ROUND_TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'assets/digit_templates/round/')
# TODO: Add BALANCE_TEMPLATE_DIR when ready

# --- Region of Interest (ROI) coordinates (x, y, width, height) ---
# Define ROIs relative to the video frame dimensions
# Format: (x, y, width, height)
BALANCE_ROI = (98, 1753, 130, 52)
ROUND_ROI   = (1764, 1660, 78, 46) 
STAGES_AREA_ROI = (921, 1473, 1175, 190)
# --- OCR Configuration ---
# Tesseract config string (adjust for better accuracy)
# --psm 7: Treat the image as a single text line
# --psm 6: Assume a single uniform block of text
TESSERACT_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-'

# --- Stability Control (Number of consecutive frames for confirmation) ---
CONFIRMATION_FRAMES_OCR = 2 # For Round/Balance values
CONFIRMATION_FRAMES_STAGE = 3 # For Stage changes

# --- Stage Detection (ML Model) ---
# Default stage if ML prediction confidence is below threshold
DEFAULT_STAGE = "x32" # The stage to assign if confidence is low for known classes

# --- Model Input Shape (MUST match training script) ---
MODEL_IMG_HEIGHT = 86
MODEL_IMG_WIDTH = 493
PREDICTION_CONFIDENCE_THRESHOLD = 0.80 # 80% confidence required

# --- Performance ---
PROCESS_EVERY_N_FRAMES = 5

# --- Balance Correction ---
DECIMAL_CORRECTION = 100 # Divide OCR balance by this value (e.g., 100)
# Set to 1 or None if no correction is needed

# --- Threading Configuration ---
FRAME_QUEUE_MAXSIZE = 60
USE_PROCESSING_THREAD = True

# --- Display Configuration ---
SHOW_DEBUG_WINDOWS = True # Show individual ROI processing steps
DISPLAY_WINDOW_NAME = "Video Feed"
BALANCE_WINDOW_NAME = "Balance OCR Input"
ROUND_WINDOW_NAME = "Round Input"
STAGE_WINDOW_NAME = "Stages Area Input (ML)"