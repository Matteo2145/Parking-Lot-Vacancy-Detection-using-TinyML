import os
import cv2
import shutil
from datetime import datetime

TIMESTAMP_START = "2013-02-22_06_05_00"
TIMESTAMP_END   = "2013-04-16_10_50_05"

# Converts '2013-04-16_10_50_05' → datetime object
def extract_timestamp_from_filename(name):
    try:
        ts = name.split('_jpg')[0]  # '2013-04-16_10_50_05'
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

# Create output dirs if they don't exist
def ensure_dirs(base_path):
    for cls in ['occupied', 'empty']:
        os.makedirs(os.path.join(base_path, cls), exist_ok=True)

# Convert YOLO normalized coordinates to pixel coordinates
def yolo_to_bbox(xc, yc, w, h, img_w, img_h):
    xmin = int((xc - w / 2) * img_w)
    ymin = int((yc - h / 2) * img_h)
    xmax = int((xc + w / 2) * img_w)
    ymax = int((yc + h / 2) * img_h)
    return max(xmin,0), max(ymin,0), min(xmax,img_w-1), min(ymax,img_h-1)

def process_split(split):
    print(f"Processing {split}...")
    image_dir = os.path.join(split, "images")
    label_dir = os.path.join(split, "labels")
    output_dir = split

    ensure_dirs(output_dir)

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        if not is_within_timestamp(label_file):
            continue

        image_name = os.path.splitext(label_file)[0]
        # Try common image extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            image_path = os.path.join(image_dir, image_name + ext)
            if os.path.exists(image_path):
                break
            else:
                print(f"Image not found for label file {label_file}")
                continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image {image_path}")
            continue

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

# Run for all dataset splits
for split in ['train', 'test', 'valid']:
    process_split(os.path.join('dataset', split))

print("✅ All done!")

