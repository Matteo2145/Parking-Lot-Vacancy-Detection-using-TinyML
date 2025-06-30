import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

# --- 1. Configuration ---
IMG_WIDTH = 48
IMG_HEIGHT = 48
BATCH_SIZE = 16
DATA_DIR = pathlib.Path("./processed_parking_data/") 

# --- 2. Load and Prepare the Dataset ---
print("--- Loading and preparing dataset ---")

# Create training and validation datasets
# Using 80% of images for training, 20% for validation.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale' # Using grayscale to reduce model size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

class_names = train_ds.class_names
print(f"Class names found: {class_names}")

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. Define the Machine Learning Model ---
print("--- Building the model ---")

model = keras.Sequential([
    # Input layer with rescaling. The input shape must match the image dimensions.
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

    # First Convolutional Block
    layers.Conv2D(8, (3, 3), activation='relu'), # 8 filters
    layers.MaxPooling2D(),

    # Second Convolutional Block
    layers.Conv2D(16, (3, 3), activation='relu'), # 16 filters
    layers.MaxPooling2D(),

    # Flatten the results to feed into a DNN
    layers.Flatten(),
    
    # A single, dense hidden layer
    layers.Dense(16, activation='relu'),

    # Output layer with a single neuron (binary classification: empyt or occupied)
    layers.Dense(1)
])

model.summary()

# --- 4. Compile and Train the Model ---
print("--- Compiling and training the model ---")
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25
)

print("--- Training complete ---")

# --- 5. Convert to TensorFlow Lite with Quantization ---
print("--- Converting model to TensorFlow Lite ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Define a representative dataset generator for quantization
def representative_dataset_gen():
    for images, _ in train_ds.take(1): 
        for i in range(images.shape[0]):
            img = images[i]
            img = tf.expand_dims(img, 0)
            yield [img]

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()

# Check the size of the converted model
tflite_model_size = len(tflite_model_quant)
print(f"TFLite model size: {tflite_model_size / 1024:.2f} KB")

# Save the quantized model to a file
tflite_model_file = pathlib.Path("parking_model_quant.tflite")
tflite_model_file.write_bytes(tflite_model_quant)
print(f"Quantized TFLite model saved to: {tflite_model_file}")

# --- 6. Convert to a C array for Arduino ---
print("--- Converting TFLite model to C source file ---")

c_model_name = 'parking_model_data'
os.system(f'xxd -i {tflite_model_file} > {c_model_name}.h')

print(f"\nModel converted to a C header file: '{c_model_name}.h'")


# --- 7. Visualize Training Results ---

print("--- Generating performance plots ---")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Save the figure and show it
plt.savefig('training_performance.png')
print("Saved performance plot to 'training_performance.png'")
plt.show()

