from PIL import Image
import os

def convert_rgb_to_grayscale_header(
    input_image_path,
    output_header_path="injected_image.h",
    target_width=160,
    target_height=120
):
    """
    Converts a standard RGB image (e.g., JPG, PNG) to Grayscale format
    and generates a C header file with the image data.

    Args:
        input_image_path (str): Path to the input image file.
        output_header_path (str): Path to the output C header file.
        target_width (int): The desired width of the image for the model input.
        target_height (int): The desired height of the image for the model input.
    """
    try:
        img = Image.open(input_image_path)
        print(f"Opened image: {input_image_path} (Original size: {img.size[0]}x{img.size[1]})")

        # Resize the image to the target dimensions
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        print(f"Resized image to: {img.size[0]}x{img.size[1]}")

        # Ensure the image is in RGB format
        img = img.convert("RGB")

        # Convert the image to grayscale
        img = img.convert("L")

        # Store the new image (resized and grayscale) as output.jpg
        img.save("output.jpg", "JPEG")

        # Generate the C header file
        with open(output_header_path, "w") as f:
            f.write(f"#ifndef INJECTED_IMAGE_H\n")
            f.write(f"#define INJECTED_IMAGE_H\n\n")
            f.write(f"const unsigned char injected_image_data[{target_width * target_height}] = {{\n")

            # Write data with formatting (16 bytes per line for readability)
            pixel_data = img.tobytes()
            for i in range(0, len(pixel_data), 16):
                line_data = pixel_data[i:i + 16]
                hex_values = ', '.join(f'0x{byte:02x}' for byte in line_data)
                f.write(f"    {hex_values},\n")
            f.write("\n};\n\n")
            f.write(f"const int kInjectedImageWidth = {target_width};\n")
            f.write(f"const int kInjectedImageHeight = {target_height};\n\n")
            f.write(f"#endif // INJECTED_IMAGE_H\n")

        print(f"Successfully generated {output_header_path}")

    except FileNotFoundError:
        print(f"Error: Input image file not found at {input_image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    input_image = "input.jpg"
    output_file = "injected_image.h"
    desired_width = 160
    desired_height = 120 

    if not os.path.exists(input_image):
        print(f"Input image file '{input_image}' does not exist.")
    else:
        convert_rgb_to_grayscale_header(input_image, output_file, desired_width, desired_height)
