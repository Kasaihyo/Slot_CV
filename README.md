# Slot symbol counter with Computer Vision

## Overview

This project uses Computer Vision (OpenCV, YOLO) and Machine Learning (TensorFlow/Keras) to automatically analyze video footage of a slot-style game. It tracks key game events like rounds, player balance, and game stages (multipliers), and counts symbols on the grid when stable, logging everything for analysis.

## Key Features

*   **Automatic Game State Tracking:** Identifies current round, player balance (using EasyOCR), and game stage/multiplier (using a CNN model).
*   **Symbol Detection & Counting:** Uses a YOLO model to detect and count symbols on the game grid.
*   **Stable Grid Detection:** Ensures symbols are counted only when the grid is visually full and stable, improving accuracy.
*   **Event-Based Logging:** Records round changes, stage changes, and stable grid symbol counts to a timestamped CSV file.
*   **Configurable:** Easily adjust paths, ROIs, detection thresholds, and other parameters in `config.py`.
*   **Optional Multi-threading:** Can speed up processing on multi-core systems.
*   **Debug Visualizations:** Optional windows show ROIs, model inputs, and YOLO detections on stable grids.

## Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Kasaihyo/Slot_CV/
    cd Slot_CV
    ```
2.  **Create Virtual Environment & Install Dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Configure:**
    *   Edit `config.py`. **Crucially, set the `VIDEO_PATH`** to your game recording.
    *   Ensure `MODEL_PATH`, `MODEL_LABELS_PATH`, and `YOLO_MODEL_PATH` point to your model files.
    *   Define the ROIs (`BALANCE_ROI`, `ROUND_ROI`, `STAGES_AREA_ROI`) or set them to `None` to draw them interactively on startup.
4.  **Run:**
    ```bash
    python main.py
    ```
5.  Press 'q' in the display window to quit. Logs are saved in the `Output/` directory.

---

## Detailed Configuration (`config.py`)

Review and adjust these parameters in `config.py` as needed:

*   **Paths:**
    *   `VIDEO_PATH`: Path to the input game video file. *Required.*
    *   `MODEL_PATH`: Path to the trained TensorFlow/Keras stage classifier model.
    *   `MODEL_LABELS_PATH`: Path to the text file with stage class names (matching model output order).
    *   `YOLO_MODEL_PATH`: Path to the trained YOLO symbol detection model.
    *   `OUTPUT_DIR`: Directory for saving log CSV files.
*   **Regions of Interest (ROIs):** `(x, y, width, height)` tuples.
    *   `BALANCE_ROI`, `ROUND_ROI`, `STAGES_AREA_ROI`.
    *   Set to `None` to enable interactive drawing on the first frame.
*   **Stage Classification:**
    *   `MODEL_IMG_WIDTH`, `MODEL_IMG_HEIGHT`: Input dimensions for the stage classifier model. *Must match the trained model.*
    *   `PREDICTION_CONFIDENCE_THRESHOLD`: Minimum confidence (0.0-1.0) to accept a stage prediction.
    *   `DEFAULT_STAGE`: Stage assigned if confidence is below the threshold.
*   **Symbol Detection (YOLO):**
    *   `YOLO_CLASS_NAMES`: List of symbol names. **Must match the order expected for logging/output CSV.**
    *   `YOLO_CONFIDENCE_THRESHOLD`: Confidence needed for a symbol to be *counted* on a stable grid.
    *   `YOLO_PREDICT_CONF`: Lower confidence used only for *detecting* if the grid *might* be full (part of stability check).
    *   `STABLE_GRID_SIZE`: Expected number of symbols in a full, stable grid (e.g., 25).
    *   `GRID_STABILITY_CHECKS`: Number of consecutive frames the grid must meet `STABLE_GRID_SIZE` criteria.
    *   `GRID_SEARCH_FRAME_INTERVAL`: Frame processing interval when actively searching for a stable grid.
*   **Stability & Performance:**
    *   `CONFIRMATION_FRAMES_OCR`, `CONFIRMATION_FRAMES_STAGE`: Frames needed for OCR/Stage values to be considered stable.
    *   `PROCESS_EVERY_N_FRAMES`: Skips processing frames during normal operation (set > 1).
    *   `USE_PROCESSING_THREAD`: `True` for multi-threading, `False` for single-threaded.
*   **Display:**
    *   `SHOW_DEBUG_WINDOWS`: `True` to show ROI processing and stable grid detection windows.
    *   Window names (`DISPLAY_WINDOW_NAME`, etc.).

## Dependencies

### Python Packages (`requirements.txt`):

*   `opencv-python`: Video/image processing.
*   `pandas`: Log file handling.
*   `tensorflow`: Stage classification model execution.
*   `numpy`: Numerical operations.
*   `easyocr`: OCR for balance/round.
*   `ultralytics`: YOLO model execution.

### External Software:

*   **EasyOCR Models:** Automatically downloaded on first run (requires internet).
*   **YOLO Models:** Ultralytics library may download components if needed.

## Detailed Usage

1.  Ensure `config.py` is configured correctly.
2.  Activate your virtual environment.
3.  Run `python main.py` from the project root.
4.  If ROIs were `None`, draw them interactively (press Enter/Space when done).
5.  The main display window shows the video feed with detected state.
6.  Debug windows (if `SHOW_DEBUG_WINDOWS = True`) show intermediate steps and stable grid detections.
7.  Console output shows detected events (round/stage changes, grid confirmations).
8.  Press 'q' to quit.
9.  A timestamped CSV log file is saved in `config.OUTPUT_DIR`. Event types (`EventType` column) distinguish `ROUND_CHANGE`, `STAGE_CHANGE`, and `STABLE_GRID` entries. Symbol counts are only present for `STABLE_GRID` events.

## Project Structure

```
├── main.py                 # Main execution script
├── config.py               # Configuration file
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/                 # ML models, label files
│   └── ...
├── src/                    # Source code modules
│   ├── video/              # Video reading
│   ├── processing/         # Frame processing (OCR, ML, YOLO)
│   ├── utils/              # Utility functions (ROI selection)
│   ├── logging/            # Logging setup
│   └── state/              # Game state tracking
├── Output/                 # Default directory for log files
│   └── game_log_... .csv   # Example output log
├── assets/                 # Optional: Sample videos, etc.
└── .gitignore              # Git ignored files
```

## TODO

*   Consider adding configuration for YOLO drawing options.
*   Investigate performance impact of full-frame YOLO detection.
*   Refine single-threaded processing logic in `main.py` if needed.

## Contributing

(Optional: Add contribution guidelines.)

## License

(Optional: Specify project license.) 
