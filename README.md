# Slot Analyzer

## Description

This project analyzes video footage from a game (presumably a slot or casino-style game) using computer vision (OpenCV) and machine learning (TensorFlow/Keras) techniques. It aims to automatically track key game state information, including:

*   **Game Round:** Identifying the start and end of game rounds using OCR.
*   **Player Balance:** Tracking the player's balance using OCR.
*   **Game Stage/Multiplier:** Classifying the current game stage or multiplier using a trained CNN model.
*   **Logging:** Recording detected changes in round, stage, and balance to a CSV file for later analysis.

## Features

*   **Video Input:** Processes video files (`config.VIDEO_PATH`).
*   **Region of Interest (ROI) Selection:** Allows defining specific areas in the video frame for analyzing balance, round number, and game stage. Includes an interactive ROI selection tool if ROIs are not pre-defined in `config.py`.
*   **OCR Integration:** Uses Pytesseract (a Python wrapper for Tesseract OCR) to extract text for balance and round number from defined ROIs. Includes basic text processing and stability checks (`config.CONFIRMATION_FRAMES_OCR`).
*   **Machine Learning Stage Classification:** Employs a pre-trained TensorFlow/Keras model (`config.MODEL_PATH`) to classify the game stage based on the visual information within the `STAGES_AREA_ROI`. Uses class names from `config.MODEL_LABELS_PATH`. Includes confidence thresholding (`config.PREDICTION_CONFIDENCE_THRESHOLD`) and a default stage assignment (`config.DEFAULT_STAGE`) for low-confidence predictions.
*   **State Tracking:** Maintains the current game state (round, balance, stage) and identifies significant changes, applying stability checks (`config.CONFIRMATION_FRAMES_STAGE`).
*   **Multi-threaded Processing (Optional):** Can utilize separate threads (`config.USE_PROCESSING_THREAD`) for reading video frames and processing them (OCR, ML) to potentially improve performance on multi-core systems.
*   **Configurable Parameters:** Most operational parameters (file paths, ROI coordinates, thresholds, etc.) are managed through the `config.py` file.
*   **Output:** Displays the processed video feed with overlays showing detected information and ROIs. Saves a detailed log of round changes and outcomes to a CSV file (`config.OUTPUT_CSV_FILE`). Optional debug windows show intermediate processing steps.

## Dependencies

### Python Packages:

Ensure you have Python 3 installed. The required packages are listed in `requirements.txt`:

*   `opencv-python`: For video reading and image processing.
*   `pytesseract`: Python wrapper for Tesseract OCR.
*   `pandas`: Used for structuring and saving the final log file.
*   `tensorflow`: For loading and running the ML stage classification model.
*   `numpy`: For numerical operations, especially with image data.

### External Software:

*   **Tesseract OCR Engine:** `pytesseract` requires the Tesseract OCR engine to be installed on your system. Installation instructions vary by OS:
    *   **macOS:** `brew install tesseract`
    *   **Debian/Ubuntu:** `sudo apt-get update && sudo apt-get install tesseract-ocr`
    *   **Windows:** Download the installer from the [official Tesseract repository](https://github.com/UB-Mannheim/tesseract/wiki). Ensure you add Tesseract to your system's PATH or explicitly set the path in `config.py` (`TESSERACT_CMD`).

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
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
4.  **Install Tesseract OCR:** Follow the instructions for your operating system (see Dependencies section).

## Configuration (`config.py`)

Before running, carefully review and edit the `config.py` file:

1.  **Paths:**
    *   `VIDEO_PATH`: Path to the input game video file.
    *   `MODEL_PATH`: Path to the trained TensorFlow/Keras model (`.h5` or SavedModel directory).
    *   `MODEL_LABELS_PATH`: Path to the text file containing class names (one per line, matching model output order).
    *   `OUTPUT_CSV_FILE`: Path where the output log CSV will be saved.
    *   `TESSERACT_CMD`: (Optional) Set the full path to the Tesseract executable if it's not in your system's PATH.
2.  **Regions of Interest (ROIs):**
    *   `BALANCE_ROI`, `ROUND_ROI`, `STAGES_AREA_ROI`: Define the `(x, y, width, height)` tuples for the areas to analyze.
    *   **Important:** If any ROI is set to `None`, the script will pause on startup and prompt you to draw the ROI interactively on the first frame of the video. Close the selection window by pressing ENTER or SPACE.
3.  **Processing Parameters:**
    *   `MODEL_IMG_WIDTH`, `MODEL_IMG_HEIGHT`: Dimensions the stage area image will be resized to before feeding into the ML model. Must match the model's expected input size.
    *   `PREDICTION_CONFIDENCE_THRESHOLD`: Minimum confidence level (0.0 to 1.0) required from the ML model to accept its prediction.
    *   `DEFAULT_STAGE`: The stage name assigned if the ML model's confidence is below the threshold.
    *   `CONFIRMATION_FRAMES_OCR`, `CONFIRMATION_FRAMES_STAGE`: Number of consecutive frames a value (OCR or Stage) must remain stable before being accepted as confirmed.
    *   `PROCESS_EVERY_N_FRAMES`: Set to > 1 to skip processing on some frames (e.g., process every 3rd frame) for potential performance gains, especially in single-threaded mode.
4.  **Execution Mode:**
    *   `USE_PROCESSING_THREAD`: Set to `True` to use multi-threading, `False` for single-threaded operation.
5.  **Display:**
    *   `DISPLAY_WINDOW_NAME`: Name for the main output window.
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
7.  If `SHOW_DEBUG_WINDOWS` is `True`, additional windows will show the isolated ROIs being processed.
8.  Console output will provide information about startup, round/stage changes, and potential errors.
9.  Press 'q' in the display window to gracefully shut down the application.
10. Upon completion or shutdown, the analysis log will be saved to `config.OUTPUT_CSV_FILE`.

## Project Structure

```
├── main.py                 # Main execution script
├── config.py               # Configuration file for paths, ROIs, parameters
├── requirements.txt        # Python package dependencies
├── README.md               # This file
├── models/                 # Directory to store ML models and label files
│   └── ...
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── video/              # Video reading components (e.g., reader.py)
│   ├── processing/         # Frame processing logic (e.g., processor.py)
│   ├── utils/              # Utility functions (e.g., roi_selector.py, text_processing.py)
│   ├── logging/            # Logging setup and management (e.g., manager.py)
│   └── state/              # Game state tracking (e.g., tracker.py)
├── Output/                 # Directory where output logs are saved
│   └── game_log.csv        # Example output log file
├── assets/                 # Optional: Store sample videos or other assets
│   └── ...
└── .gitignore              # Specifies intentionally untracked files for Git
```

*(Note: The exact structure within `src/` might vary slightly based on the actual implementation details not fully visible here.)*

## Contributing

(Optional: Add guidelines if you want others to contribute.)

## License

(Optional: Specify the project license, e.g., MIT, Apache 2.0.) 
