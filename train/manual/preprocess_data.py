import os
from PIL import Image
import pathlib

# --- Configuration ---
TARGET_SIZE = 48 

# The directory with your original, variable-sized images.
RAW_DATA_DIR = pathlib.Path("../dataset/")

# The directory where the processed, 48x48 images will be saved.
PROCESSED_DATA_DIR = pathlib.Path("./processed_parking_data")

print(f"--- Starting Image Preprocessing ---")
print(f"Source: {RAW_DATA_DIR}")
print(f"Destination: {PROCESSED_DATA_DIR}")
print(f"Target Size: {TARGET_SIZE}x{TARGET_SIZE}")

for class_dir in RAW_DATA_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name
    print(f"\nProcessing class: {class_name}")

    # Create the corresponding output directory
    output_class_dir = PROCESSED_DATA_DIR / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in the class directory
    for image_path in class_dir.glob('*'):
        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        
        try:
            # Open the image
            with Image.open(image_path) as img:
                # Convert to grayscale, as color is not needed and adds size
                img = img.convert('L')

                img.thumbnail((TARGET_SIZE, TARGET_SIZE))

                background = Image.new('L', (TARGET_SIZE, TARGET_SIZE), 0)

                paste_x = (TARGET_SIZE - img.width) // 2
                paste_y = (TARGET_SIZE - img.height) // 2
                
                background.paste(img, (paste_x, paste_y))

                output_path = output_class_dir / image_path.name
                background.save(output_path)

        except Exception as e:
            print(f"  - Could not process {image_path}: {e}")

print("\n--- Preprocessing Complete ---")
print(f"Processed images are saved in: {PROCESSED_DATA_DIR}")
