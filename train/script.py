import os
import cv2
import datetime
import shutil
from glob import glob
from tqdm import tqdm
import yaml 
from ultralytics import YOLO
import tensorflow as tf


# --- Configuration ---
DATASET_BASE_DIR = "./dataset"
TEMP_DIR = "./tmp"
LOG_FILE = "log.txt"
MODEL_NAME = "yolo11n.pt" 
PROJECT_NAME = "pklot_detection"
EPOCHS = 1 
IMG_SIZE = 640 

PRUNING_AMOUNT = 0.5 
QUANTIZATION_TYPE = tf.lite.Optimize.OPTIMIZE_FOR_SIZE 

start_time = datetime.datetime.now()
with open(LOG_FILE, 'a') as f:
    f.write(f"Training started at: {start_time}\n")

# --- 1. Grayscale Conversion and Data Preparation ---

print("--- Starting Grayscale Conversion ---")
for split in ['train', 'valid', 'test']:
    input_dir_images = os.path.join(DATASET_BASE_DIR, split, 'images')
    output_dir_images = os.path.join(TEMP_DIR, split, 'images')
    os.makedirs(output_dir_images, exist_ok=True)

    image_paths = glob(os.path.join(input_dir_images, "*.*"))
    print(f"Processing {split} set with {len(image_paths)} images...")

    for img_path in tqdm(image_paths):
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Warning: Could not read image {img_path}, skipping.")
            continue
        gray_3ch = cv2.merge([gray, gray, gray]) # Convert to 3-channel grayscale
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir_images, filename)
        cv2.imwrite(save_path, gray_3ch)

print("✅ Grayscale conversion complete. Output saved in 'tmp/' folder.")

# Copy annotations
print("--- Copying Annotations ---")
for split in ['train', 'valid', 'test']:
    input_ann_dir = os.path.join(DATASET_BASE_DIR, split, 'labels')
    output_ann_dir = os.path.join(TEMP_DIR, split, 'labels')
    os.makedirs(output_ann_dir, exist_ok=True)

    ann_paths = glob(os.path.join(input_ann_dir, "*.txt"))
    print(f"Copying annotations for {split} set with {len(ann_paths)} annotations...")

    for ann_path in tqdm(ann_paths):
        filename = os.path.basename(ann_path)
        save_path = os.path.join(output_ann_dir, filename)
        shutil.copy(ann_path, save_path)

# Copy data.yaml
data_yaml_path = os.path.join(DATASET_BASE_DIR, 'data.yaml')
output_data_yaml_path = os.path.join(TEMP_DIR, 'data.yaml')
shutil.copy(data_yaml_path, output_data_yaml_path)
print("✅ Annotations and data.yaml copied successfully.")

# --- 2. Model Training with YOLOv11 ---
try:
    print(f"--- Starting YOLOv11 Training with {MODEL_NAME} ---")
    model = YOLO(MODEL_NAME) 

    results = model.train(
        data=os.path.join(TEMP_DIR, "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=PROJECT_NAME,
        name='yolov11n_grayscale_trained'
    )

    # Path to the best trained model
    trained_model_path = os.path.join(PROJECT_NAME, 'yolov11n_grayscale_trained', 'weights', 'best.pt')
    if not os.path.exists(trained_model_path):
        print(f"Warning: Trained model not found at {trained_model_path}. Checking last.pt...")
        trained_model_path = os.path.join(PROJECT_NAME, 'yolov11n_grayscale_trained', 'weights', 'last.pt')
        if not os.path.exists(trained_model_path):
            raise FileNotFoundError(f"Neither best.pt nor last.pt found at {os.path.join(PROJECT_NAME, 'yolov11n_grayscale_trained', 'weights')}")


    print(f"✅ Training complete. Model saved at: {trained_model_path}")

    # --- 3. Pruning the Trained Model (PyTorch-based) ---
    print(f"--- Starting Model Pruning (Amount: {PRUNING_AMOUNT*100:.0f}%) ---")
    model_for_pruning = YOLO(trained_model_path).model


    parameters_to_prune = []
    for module_name, module in model_for_pruning.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            # Prune 'weight' of Linear and Conv2d layers
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=PRUNING_AMOUNT,
    )

    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

    print("✅ Pruning complete. Saving pruned model...")

    pruned_model_path = os.path.join(PROJECT_NAME, 'yolov11n_grayscale_trained', 'weights', 'pruned.pt')
    torch.save(model_for_pruning.state_dict(), pruned_model_path)
    yolo_pruned_model = YOLO(pruned_model_path)


    # --- 4. Export to TFLite with Quantization ---
    print("--- Starting TFLite Export with Quantization ---")
    tflite_model_path = os.path.join(PROJECT_NAME, 'yolov11n_grayscale_trained', 'weights', 'pruned_quantized.tflite')

    def representative_dataset_gen():
        val_images_path = os.path.join(TEMP_DIR, 'valid', 'images')
        val_image_files = glob(os.path.join(val_images_path, "*.jpg"))[:100]

        for image_path in val_image_files:
            img = cv2.imread(image_path)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # TFLite models often expect RGB
            #img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            #img = img.astype(np.float32) / 255.0 # Normalize to 0-1 range
            yield [img]#.transpose(2,0,1)[None,:,:,:]] 


    # First, export to ONNX (intermediate step)
    onnx_path = os.path.join(PROJECT_NAME, 'yolov11n_grayscale_trained', 'weights', 'pruned.onnx')
    yolo_pruned_model.export(format='onnx', opset=12, simplify=True, half=False) 

    import onnx
    from onnx_tf.backend import prepare

    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    # Save as TensorFlow SavedModel
    saved_model_path = os.path.join(PROJECT_NAME, 'yolov11n_grayscale_trained', 'weights', 'pruned_saved_model')
    tf_rep.export_graph(saved_model_path)

    # Convert to TFLite with full integer quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [QUANTIZATION_TYPE]

    # This will map float values to int8 ranges.
    converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)

    # Ensure all operations are supported in integer format.
    # If not, some ops might fall back to float, increasing size.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.uint8 
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f"✅ TFLite model with full integer quantization saved at: {tflite_model_path}")
    print(f"TFLite model size: {os.path.getsize(tflite_model_path) / (1024*1024):.2f} MB")

    # Clean up temporary directory
    shutil.rmtree(TEMP_DIR)

    end_time = datetime.datetime.now()
    with open(LOG_FILE, 'a') as f:
        f.write(f"Training and optimization finished at: {end_time}\n")
        f.write(f"Total duration: {end_time - start_time}\n")
        f.write("-" * 50 + "\n")

except Exception as e:
    end_time = datetime.datetime.now()
    with open(LOG_FILE, 'a') as f:
        f.write(f"Training and optimization failed at: {end_time}\n")
        f.write(f"Error: {e}\n")
        f.write("-" * 50 + "\n")
    print(f"An error occurred: {e}")

