from PIL import Image
import os

def convert_rgb_to_rgb565_header(
    input_image_path,
    output_header_path="injected_image.h",
    target_width=160,
    target_height=120
):
    """
    Converts a standard RGB image (e.g., JPG, PNG) to RGB565 format
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

        rgb565_data = []
        for y in range(target_height):
            for x in range(target_width):
                r, g, b = img.getpixel((x, y))

                # Convert 8-bit RGB to 5-bit R, 6-bit G, 5-bit B
                # R: 8 bits -> 5 bits (discard least significant 3 bits)
                # G: 8 bits -> 6 bits (discard least significant 2 bits)
                # B: 8 bits -> 5 bits (discard least significant 3 bits)
                r5 = (r >> 3) & 0x1F
                g6 = (g >> 2) & 0x3F
                b5 = (b >> 3) & 0x1F

                # Combine into a 16-bit RGB565 value
                # Format: RRRRRGGG GGGBBBBB
                pixel565 = (r5 << 11) | (g6 << 5) | b5

                # Store as two bytes (little-endian: low byte first, then high byte)
                # This matches how Arduino might read uint16_t if it's little-endian
                rgb565_data.append(pixel565 & 0xFF)         # Low byte
                rgb565_data.append((pixel565 >> 8) & 0xFF)  # High byte

        # Store the new image (resized and RGB565) as output.jpg
        img.save("output.jpg", "JPEG")

        # Generate the C header file
        with open(output_header_path, "w") as f:
            f.write(f"#ifndef INJECTED_IMAGE_H\n")
            f.write(f"#define INJECTED_IMAGE_H\n\n")
            f.write(f"const unsigned char injected_image_data[{target_width * target_height * 2}] = {{\n")

            # Write data with formatting (16 bytes per line for readability)
            for i, byte_val in enumerate(rgb565_data):
                f.write(f"0x{byte_val:02X}")
                if i < len(rgb565_data) - 1:
                    f.write(", ")
                if (i + 1) % 16 == 0:
                    f.write("\n")
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
        convert_rgb_to_rgb565_header(input_image, output_file, desired_width, desired_height)
