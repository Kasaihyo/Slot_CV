# config.py
import os

# --- Core Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, 'videos')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Tesseract Configuration ---
# Uncomment and modify IF TESSERACT IS NOT IN YOUR SYSTEM PATH
# TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Windows example
TESSERACT_CMD = None # Set to None if it's in PATH
TESSERACT_CONFIG = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-'

# --- Video Source ---
VIDEO_FILENAME = '/Users/temogiorgadze/Documents/FluxGaming/CV_slots/Data/Videos/Sequence 01.mp4'
VIDEO_PATH = os.path.join(VIDEO_DIR, VIDEO_FILENAME)
# VIDEO_PATH = 0 # Use 0 for the default webcam

# --- ROIs - Region of Interest Coordinates (x, y, width, height) ---
# Set specific ROIs to None to select them interactively on the first run.
BALANCE_ROI = (98, 1753, 130, 52)   # Example: ROI for balance number OCR
ROUND_ROI = (1764, 1660, 78, 46)    # Example: ROI for round number OCR
STAGES_AREA_ROI = (921, 1473, 1175, 190) # ROI covering ALL multipliers (x1 to x32)
# STAGES_AREA_ROI = None # Set to None to select interactively

# --- Output File ---
OUTPUT_CSV_FILENAME = 'game_log_data_ml.csv'
OUTPUT_CSV_FILE = os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILENAME)

# --- Stability Control ---
CONFIRMATION_FRAMES_OCR = 3   # For balance and round number stability
CONFIRMATION_FRAMES_STAGE = 3 # For ML stage detection stability

# --- Stage Detection Configuration (ML) ---
MODEL_FILENAME = 'stage_classifier_model_5class.keras'
CLASS_NAMES_FILENAME = 'stage_class_names_5class.txt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, CLASS_NAMES_FILENAME)

# --- Define ALL possible stages, including the default one ---
# These are used for logic/display, not directly by the model prediction index
ALL_POSSIBLE_STAGES = ["x1", "x2", "x4", "x8", "x16", "x32"]
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
ROUND_WINDOW_NAME = "Round OCR Input"
STAGE_WINDOW_NAME = "Stages Area Input (ML)"