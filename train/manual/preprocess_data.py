import os
from PIL import Image
import shutil
from datetime import datetime
import pathlib
import cv2

# Adjusted to get only images of a single parking lot
TIMESTAMP_START = "2013-02-22_06_05_00"
TIMESTAMP_END   = "2013-04-16_10_50_05"

def extract_timestamp_from_filename(name):
    try:
        ts = name.split('_jpg')[0]
        return datetime.strptime(ts, "%Y-%m-%d_%H_%M_%S")
    except Exception as e:
        return None

def is_within_timestamp(filename):
    ts = extract_timestamp_from_filename(filename)
    if ts is None:
        return False
    start = datetime.strptime(TIMESTAMP_START, "%Y-%m-%d_%H_%M_%S")
    end = datetime.strptime(TIMESTAMP_END, "%Y-%m-%d_%H_%M_%S")
    return start <= ts <= end

def ensure_dirs(base_path):
    for cls in ['occupied', 'empty']:
        os.makedirs(os.path.join(base_path, cls), exist_ok=True)

def yolo_to_bbox(xc, yc, w, h, img_w, img_h):
    xmin = int((xc - w / 2) * img_w)
    ymin = int((yc - h / 2) * img_h)
    xmax = int((xc + w / 2) * img_w)
    ymax = int((yc + h / 2) * img_h)
    return max(xmin,0), max(ymin,0), min(xmax,img_w-1), min(ymax,img_h-1)

def process_split(output_dir, split):
    print(f"Processing {split}...")
    image_dir = os.path.join(split, "images")
    label_dir = os.path.join(split, "labels")

    ensure_dirs(output_dir)

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        if not is_within_timestamp(label_file):
            continue

        image_name = os.path.splitext(label_file)[0]
        image_path = os.path.join(image_dir, image_name + '.jpg')

        if not os.path.exists(image_path):
            print(f"Image not found for label file {label_file}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image {image_path}")
            continue

        # Convert image to 160x120
        image = cv2.resize(image, (160, 120))

        h, w = image.shape[:2]

        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts)
            xmin, ymin, xmax, ymax = yolo_to_bbox(xc, yc, bw, bh, w, h)

            crop = image[ymin:ymax, xmin:xmax]

            if crop.size == 0:
                continue

            category = 'occupied' if int(cls) == 1 else 'empty'
            filename = f"{image_name}_{i}_{category}.jpg"
            out_path = os.path.join(output_dir, category, filename)
            cv2.imwrite(out_path, crop)
    print(f"Finished {split}.")

for split in ['train', 'test', 'valid']:
    process_split(os.path.join('tmp'), os.path.join('..', '..', 'dataset', split))

# --- Configuration ---
TARGET_SIZE = 48 
RAW_DATA_DIR = pathlib.Path("./tmp/")
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

# Clean up the temporary directory
shutil.rmtree(RAW_DATA_DIR, ignore_errors=True)

print("\n--- Preprocessing Complete ---")
print(f"Processed images are saved in: {PROCESSED_DATA_DIR}")
