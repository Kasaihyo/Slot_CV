# src/utils/image_utils.py
import cv2
import numpy as np
import re
import config
import easyocr
import os

# Initialize EasyOCR Reader globally (loads the model once)
# You might want to specify GPU usage: easyocr.Reader(['en'], gpu=True)
reader = easyocr.Reader(['en'])

def preprocess_ocr(roi_frame):
    """Basic preprocessing and OCR using EasyOCR."""
    if roi_frame is None or roi_frame.size == 0:
        return None, "" # Return None for image, empty string for text
    try:
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        # Thresholding often not needed or detrimental for EasyOCR
        # Perform OCR using EasyOCR
        # detail=0 returns only the text strings
        # paragraph=False might be better for single-line ROIs
        results = reader.readtext(gray, detail=0, paragraph=False)
        raw_text = " ".join(results) # Join potential multiple detections
        
        # Return the preprocessed (grayscale) image and the raw text
        return gray, raw_text
    except Exception as e:
        print(f"Warning: EasyOCR preprocessing/execution failed: {e}")
        return None, "" # Return gracefully on error

def clean_ocr_text(raw_text, allowed_chars='0123456789-'):
    """Removes unwanted characters from OCR text."""
    if not raw_text:
        return ""
    pattern = f'[^{re.escape(allowed_chars)}]+' # Create regex pattern dynamically
    cleaned = re.sub(pattern, '', raw_text)
    return cleaned.strip()

def preprocess_image_for_model(img_roi, target_height, target_width):
    """Resizes an image ROI for model prediction, keeping color."""
    if img_roi is None or img_roi.size == 0:
        return None

    # Resize - Use INTER_AREA for shrinking, INTER_LINEAR for enlarging generally good
    if img_roi.shape[0] > target_height or img_roi.shape[1] > target_width:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    try:
        img_resized = cv2.resize(img_roi, (target_width, target_height), interpolation=interpolation)
        # Add batch dimension (Model expects batch_size, height, width, channels)
        img_batch = np.expand_dims(img_resized, axis=0)
        # Normalization is often done *inside* the model with a Rescaling layer.
        # If your model DOES NOT have a Rescaling layer, uncomment the next line:
        # img_batch = img_batch.astype(np.float32) / 255.0
        return img_batch
    except Exception as e:
         print(f"Warning: Failed to preprocess image for model: {e}")
         return None