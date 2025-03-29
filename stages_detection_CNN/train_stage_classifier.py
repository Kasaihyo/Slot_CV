import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator # Not needed with layers
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

# --- Configuration ---
DATASET_DIR = '/Users/temogiorgadze/Documents/FluxGaming/CV_slots/stages_detection_CNN/dataset'        # Path to your dataset (MUST contain ONLY x1, x2, x4, x8, x16 folders)
IMG_HEIGHT = 86               # Target image height (resize to this)
IMG_WIDTH = 493               # Target image width (adjust based on ROI aspect ratio)
BATCH_SIZE = 16
EPOCHS = 30                   # Increase epochs slightly due to augmentation (monitor val_loss)
VALIDATION_SPLIT = 1/6        # Use 20% for validation
MODEL_SAVE_PATH = 'stage_classifier_model_5class.keras' # Model trained on 5 classes
CLASS_NAMES_SAVE_PATH = "stage_class_names_5class.txt" # Class names for the 5 trained classes

# --- Load Data ---
data_dir = pathlib.Path(DATASET_DIR)
if not data_dir.exists():
    raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

image_count = len(list(data_dir.glob('*/*.png'))) # Adjust extension if needed
print(f"Found {image_count} images.")

# Use image_dataset_from_directory
print("Loading training data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=VALIDATION_SPLIT,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

print("Loading validation data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=VALIDATION_SPLIT,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes found for training: {class_names}")
if num_classes != 5:
     print(f"WARNING: Expected 5 classes (x1-x16), but found {num_classes}. Check dataset folders!")
     # Consider raising an error if num_classes is drastically wrong

# --- Configure Dataset for Performance ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Data Augmentation Layers ---
# Applied ONLY during training, creates variations of your images
data_augmentation = keras.Sequential(
  [
    # Start with input shape definition here
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    # Apply augmentations
    # layers.RandomFlip("horizontal"), # Probably NOT useful unless your ROI can be mirrored
    layers.RandomRotation(0.08),   # Increase rotation slightly?
    layers.RandomZoom(0.1),      # Increase zoom slightly?
    layers.RandomContrast(0.15),  # Increase contrast variation
    layers.RandomBrightness(0.15) # Increase brightness variation
    # You can add more like layers.RandomTranslation
  ],
  name="data_augmentation"
)
print("INFO: Data augmentation layers configured.")

# --- Build the CNN Model ---
model = keras.Sequential([
  # Add augmentation layer first
  data_augmentation,

  # Rescaling MUST happen AFTER augmentation if augmentation expects 0-255
  layers.Rescaling(1./255), # No input_shape here, taken from augmentation layer

  # Conv Blocks (Keep the same structure or simplify if needed for smaller data)
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  # Consider adding Dropout for regularization, especially with augmentation
  layers.Dropout(0.2),

  # Flatten and Dense Layers
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.3), # Another dropout layer before the final one
  # Final layer MUST have num_classes units (which should be 5)
  layers.Dense(num_classes, activation='softmax')
])

# --- Compile the Model ---
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# --- Print Model Summary ---
# Need to call build or fit first usually, let's rely on fit/summary after
# model.build(input_shape=(None, IMG_HEIGHT, IMG_WIDTH, 3)) # Build model to show summary
# model.summary() # Summary will print when model.fit is called

# --- Train the Model ---
print("\nStarting training (with augmentation)...")
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS
)
print("Training finished.")
model.summary() # Print summary after training

# --- Evaluate ---
loss, acc = model.evaluate(val_ds, verbose=2)
print(f"\nValidation Accuracy: {acc:.4f}, Validation Loss: {loss:.4f}")

# --- Plot Training History ---
# (Plotting code remains the same - good for checking if augmentation helped/overfitting)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc)) # Use len(acc) in case training stopped early

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# --- Save the Trained Model (5 classes) ---
print(f"\nSaving trained 5-class model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved.")

# --- Save the 5 Class Names ---
print(f"Saving 5 class names to {CLASS_NAMES_SAVE_PATH}...")
with open(CLASS_NAMES_SAVE_PATH, "w") as f:
    for item in class_names: # class_names should contain x1, x2, x4, x8, x16
        f.write("%s\n" % item)
print("Class names saved.")