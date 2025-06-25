import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO
import tensorflow as tf
import shutil

# Convert Yolov11 model to TensorFlow format
model = YOLO("best.pt")

converter = tf.lite.TFLiteConverter.from_saved_model("best_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the converted TFLite model
with open("best.tflite", "wb") as f:
    f.write(tflite_model)

# xxd -i model.tflite > model.h

# Clean up the exported TensorFlow model directory
shutil.rmtree("best_tf", ignore_errors=True)

