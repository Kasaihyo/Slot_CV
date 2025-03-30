# Slot symbol counter with Computer Vision

## Description

This project analyzes video footage from a game (presumably a slot or casino-style game) using computer vision (OpenCV) and machine learning (TensorFlow/Keras) techniques. It aims to automatically track key game state information, including:

*   **Game Round:** Identifying the start and end of game rounds using EasyOCR.
*   **Player Balance:** Tracking the player's balance using EasyOCR.
*   **Game Stage/Multiplier:** Classifying the current game stage or multiplier using a trained CNN model.
*   **Logging:** Recording detected changes in round, stage, and balance to a CSV file for later analysis.
*   **Symbol Detection (YOLOv11):** Detects symbols using a YOLOv11 model (`config.YOLO_MODEL_PATH`) across the entire frame.
*   **Stable Grid Detection:** Implements logic to ensure symbol counts are logged only when the game grid is full (`config.STABLE_GRID_SIZE`) and visually stable over several checks (`config.GRID_STABILITY_CHECKS`). Uses a lower confidence (`config.YOLO_PREDICT_CONF`) for finding the grid and a higher confidence (`config.YOLO_CONFIDENCE_THRESHOLD`) for the final counting.
*   **Adaptive Frame Processing:** Reduces the frame processing interval (`config.GRID_SEARCH_FRAME_INTERVAL`) when actively searching for a stable grid (e.g., after a stage change).
*   **Event-Based Logging:** Logs round changes, stage changes, and stable grid detections as separate events in the output CSV, ensuring symbol counts are associated with the correct frame.

## Features

*   **Video Input:** Processes video files (`config.VIDEO_PATH`).
*   **Region of Interest (ROI) Selection:** Allows defining specific areas in the video frame for analyzing balance, round number, and game stage. Includes an interactive ROI selection tool if ROIs are not pre-defined in `config.py`.
*   **OCR Integration:** Uses EasyOCR to extract text for balance and round number from defined ROIs. Includes stability checks (`config.CONFIRMATION_FRAMES_OCR`).
*   **Machine Learning Stage Classification:** Employs a pre-trained TensorFlow/Keras model (`config.MODEL_PATH`) to classify the game stage based on the visual information within the `STAGES_AREA_ROI`. Uses class names from `config.MODEL_LABELS_PATH`. Includes confidence thresholding (`config.PREDICTION_CONFIDENCE_THRESHOLD`) and a default stage assignment (`config.DEFAULT_STAGE`) for low-confidence predictions.
*   **State Tracking:** Maintains the current game state (round, balance, stage) and identifies significant changes, applying stability checks (`config.CONFIRMATION_FRAMES_STAGE`).
*   **Multi-threaded Processing (Optional):** Can utilize separate threads (`config.USE_PROCESSING_THREAD`) for reading video frames and processing them (ML, OCR) to potentially improve performance on multi-core systems.
*   **Configurable Parameters:** Most operational parameters (file paths, ROI coordinates, thresholds, etc.) are managed through the `config.py` file.
*   **Output:** Displays the processed video feed with overlays showing detected information and ROIs. Saves a detailed log of events (round changes, stage changes, stable grids) to a timestamped CSV file in the `config.OUTPUT_DIR` directory. Optional debug windows show intermediate processing steps, including YOLO detections on stable grids.

## Dependencies

### Python Packages:

Ensure you have Python 3 installed. The required packages are listed in `requirements.txt`:

*   `opencv-python`: For video reading and image processing.
*   `pandas`: Used for structuring and saving the final log file.
*   `tensorflow`: For loading and running the ML stage classification model.
*   `numpy`: For numerical operations, especially with image data.
*   `easyocr`: For Optical Character Recognition (OCR) to read balance and round numbers.
*   `ultralytics`: For loading and running the YOLOv11 symbol detection model.

### External Software:

*   **EasyOCR Models:** EasyOCR automatically downloads required language models (e.g., for English) on first use. Ensure you have an internet connection when running the script for the first time after installing `easyocr`.
*   **YOLO Models:** The Ultralytics library might download specific components if needed, depending on the model format and environment.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Kasaihyo/Slot_CV/
    cd Slot_CV
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration (`config.py`)

Before running, carefully review and edit the `config.py` file:

1.  **Paths:**
    *   `VIDEO_PATH`: Path to the input game video file.
    *   `MODEL_PATH`: Path to the trained TensorFlow/Keras model (`.keras` or SavedModel directory).
    *   `MODEL_LABELS_PATH`: Path to the text file containing class names (one per line, matching model output order).
    *   `OUTPUT_DIR`: Directory where the output log CSV files will be saved.
2.  **Regions of Interest (ROIs):**
    *   `BALANCE_ROI`, `ROUND_ROI`, `STAGES_AREA_ROI`: Define the `(x, y, width, height)` tuples for the areas to analyze.
    *   **Important:** If any ROI is set to `None`, the script will pause on startup and prompt you to draw the ROI interactively on the first frame of the video. Close the selection window by pressing ENTER or SPACE.
3.  **Processing Parameters:**
    *   `MODEL_IMG_WIDTH`, `MODEL_IMG_HEIGHT`: Dimensions the stage area image will be resized to before feeding into the ML model. Must match the model's expected input size.
    *   `PREDICTION_CONFIDENCE_THRESHOLD`: Minimum confidence level (0.0 to 1.0) required from the ML model to accept its prediction.
    *   `DEFAULT_STAGE`: The stage name assigned if the ML model's confidence is below the threshold.
    *   `CONFIRMATION_FRAMES_OCR`, `CONFIRMATION_FRAMES_STAGE`: Number of consecutive frames a value (Round/Balance or Stage) must remain stable before being accepted as confirmed.
    *   `PROCESS_EVERY_N_FRAMES`: Set to > 1 to skip processing on some frames (e.g., process every 10th frame) for potential performance gains, especially in single-threaded mode during normal operation.
4.  **YOLO Symbol Detection Parameters:**
    *   `YOLO_MODEL_PATH`: Path to the trained YOLOv11 model (`.pt` file).
    *   `YOLO_CLASS_NAMES`: **Crucially, this list must match the order used for logging and expected in the output CSV.** It might differ from the model's internal training order.
    *   `YOLO_CONFIDENCE_THRESHOLD`: Minimum confidence (0.0 to 1.0) required for a detected symbol to be *counted* during the final stable grid count.
    *   `YOLO_PREDICT_CONF`: Lower confidence threshold (0.0 to 1.0) used *only* for initially detecting boxes to determine if the grid *might* be full. This helps find potential stable frames even if some symbols have low confidence.
    *   `STABLE_GRID_SIZE`: The exact number of symbols expected when the grid is full and stable (e.g., 25).
    *   `GRID_STABILITY_CHECKS`: How many consecutive frames (processed at the search interval) must show the `STABLE_GRID_SIZE` number of boxes before the grid is considered stable.
    *   `GRID_SEARCH_FRAME_INTERVAL`: Process every Nth frame when actively searching for a stable grid (state `SEARCHING` or `CONFIRMING`), typically lower than `PROCESS_EVERY_N_FRAMES`.
5.  **Execution Mode:**
    *   `USE_PROCESSING_THREAD`: Set to `True` to use multi-threading, `False` for single-threaded operation.
6.  **Display:**
    *   `DISPLAY_WINDOW_NAME`: Name for the main output window.
    *   `ROUND_WINDOW_NAME`, `STAGE_WINDOW_NAME`: Names for the OCR/Stage ROI debug windows.
    *   `STABLE_GRID_WINDOW_NAME`: Name for the debug window showing the full frame with YOLO bounding boxes drawn when a stable grid is detected.
    *   `SHOW_DEBUG_WINDOWS`: Set to `True` to show separate windows displaying the processed ROI images (Balance, Round, Stage).

## Usage

1.  Ensure all configurations in `config.py` are correct.
2.  Activate your virtual environment (if used).
3.  Run the main script from the project's root directory:
    ```bash
    python main.py
    ```
4.  The script will start processing the video specified in `config.py`.
5.  If any ROIs were set to `None`, interactive selection windows will appear first. Draw the rectangles and press Enter/Space.
6.  A display window (`config.DISPLAY_WINDOW_NAME`) will show the video feed with detected information (Round, Balance, Stage) and ROI overlays.
7.  If `SHOW_DEBUG_WINDOWS` is `True`, additional windows will show the isolated ROIs being processed, **and a window (`config.STABLE_GRID_WINDOW_NAME`) will show the frame with YOLO detections whenever a stable grid is confirmed.**
8.  Console output will provide information about startup, round/stage changes, stable grid confirmations, and potential errors.
9.  Press 'q' in the display window to gracefully shut down the application.
10. Upon completion or shutdown, the analysis log will be saved to a timestamped file in `config.OUTPUT_DIR`. This CSV file contains separate rows for different event types (`EventType` column: `ROUND_CHANGE`, `STAGE_CHANGE`, `STABLE_GRID`). Symbol counts will only be populated on `STABLE_GRID` rows.

## Project Structure

```
├── main.py                 # Main execution script
├── config.py               # Configuration file for paths, ROIs, parameters
├── requirements.txt        # Python package dependencies
├── TODO.md                 # List of tasks and future work
├── README.md               # This file
├── models/                 # Directory to store ML models and label files
│   └── ...
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── video/              # Video reading components (e.g., reader.py)
│   ├── processing/         # Frame processing logic (e.g., processor.py)
│   ├── utils/              # Utility functions (e.g., roi_selector.py)
│   ├── logging/            # Logging setup and management (e.g., manager.py)
│   └── state/              # Game state tracking (e.g., tracker.py)
├── Output/                 # Directory where output logs are saved
│   └── game_log_... .csv   # Example output log file
├── assets/                 # Optional: Store sample videos or other assets
│   └── ...
└── .gitignore              # Specifies intentionally untracked files for Git
```

*(Note: The exact structure within `src/` might vary slightly.)*

## TODO

*   ~~Fix the balance tracking/extraction.~~ (Seems fixed)
*   ~~Implement symbol counting/detection:~~ (Implemented)
    *   ~~Train classification model (9 symbols).~~ (Assumed done - YOLO model provided)
    *   ~~Map the boxes for detection.~~ (Implemented via YOLO)
    *   ~~Find the still frame within animations for reliable detection.~~ (Implemented via stable grid logic)
*   Consider adding configuration for YOLO drawing (e.g., different colors for boxes below counting threshold).
*   Investigate performance impact of full-frame YOLO detection.
*   Refine single-threaded processing logic in `main.py` if needed.

## Contributing

(Optional: Add guidelines if you want others to contribute.)

## License

(Optional: Specify the project license, e.g., MIT, Apache 2.0.) 
