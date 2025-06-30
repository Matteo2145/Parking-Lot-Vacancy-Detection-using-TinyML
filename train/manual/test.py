import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import pathlib

# --- Configuration ---
IMG_WIDTH = 48
IMG_HEIGHT = 48
CLASS_NAMES = ['empty', 'occupied']


def preprocess_image(image_path, target_size):
    """
    Loads an image, and preprocesses it exactly like the training data:
    1. Opens and converts to grayscale.
    2. Resizes with padding to maintain aspect ratio.
    """
    with Image.open(image_path) as img:
        # 1. Convert to grayscale
        img_gray = img.convert('L')
        
        # 2. Resize with padding
        img_gray.thumbnail((target_size, target_size))
        background = Image.new('L', (target_size, target_size), 0)
        paste_x = (target_size - img_gray.width) // 2
        paste_y = (target_size - img_gray.height) // 2
        background.paste(img_gray, (paste_x, paste_y))
        
        return background

def main():
    # --- 1. Set up Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Test the quantized TFLite parking model.")
    parser.add_argument("image_path", type=pathlib.Path, help="Path to the input image.")
    parser.add_argument(
        "--model_path",
        type=pathlib.Path,
        default="parking_model_quant.tflite",
        help="Path to the .tflite model file."
    )
    args = parser.parse_args()

    if not args.image_path.is_file():
        print(f"Error: Image file not found at {args.image_path}")
        return

    print(f"--- Loading model: {args.model_path} ---")
    
    # --- 2. Load the TFLite Model and Allocate Tensors ---
    interpreter = tf.lite.Interpreter(model_path=str(args.model_path))
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # --- 3. Preprocess the Input Image ---
    print(f"--- Preprocessing image: {args.image_path} ---")
    processed_img = preprocess_image(args.image_path, IMG_WIDTH)

    # Convert image to a numpy array, add a batch dimension
    input_data = np.expand_dims(processed_img, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)
    
    input_data = (input_data - 128).astype(np.int8)
    
    # --- 4. Run Inference ---
    print("--- Running inference ---")
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # --- 5. Interpret the Output ---
    # The output is a raw score (logit) from the final dense layer.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    score = output_data[0][0] # Get the single score value

    if score > 0:
        predicted_class = CLASS_NAMES[1]
    else:
        predicted_class = CLASS_NAMES[0]

    print("\n--- Results ---")
    print(f"Raw Output Score: {score}")
    print(f"Prediction: '{predicted_class}'")


if __name__ == '__main__':
    main()
