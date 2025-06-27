#include <Arduino_OV767X.h>
#include <Arduino_TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"
#include "boxes.h"
#include "injected_image.h"


extern const unsigned char g_model_data[];


// --- TensorFlow Lite globals ---
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 90 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

// --- Model input shape ---
constexpr int kInputWidth = 640;
constexpr int kInputHeight = 640;
constexpr int kInputChannels = 3;

// --- Camera resolution ---
constexpr int kFrameWidth = 160;
constexpr int kFrameHeight = 120;
uint8_t frame_buffer[kFrameWidth * kFrameHeight * 2]; // RGB565: 2 bytes per pixel

// Helper to get RGB values from RGB565
void rgb565_to_rgb888(uint16_t pixel, uint8_t& r, uint8_t& g, uint8_t& b) {
  r = ((pixel >> 11) & 0x1F) << 3;
  g = ((pixel >> 5) & 0x3F) << 2;
  b = (pixel & 0x1F) << 3;
}

// Crop + resize a box from camera frame to 640x640
void extract_and_resize_box(const Box& box) {
  int image_offset = 0;

  // Convert box from 640x640 normalized to pixel coords on 640x480
  float scale_y = (float)kFrameHeight / 640.0f; // 480 / 640 = 0.75

  int cx = box.x_center * 640;
  int cy = box.y_center * 640 * scale_y;
  int w = box.width * 640;
  int h = box.height * 640 * scale_y;

  int x0 = max(0, cx - w / 2);
  int y0 = max(0, cy - h / 2);
  int x1 = min(kFrameWidth - 1, cx + w / 2);
  int y1 = min(kFrameHeight - 1, cy + h / 2);

  int crop_w = x1 - x0;
  int crop_h = y1 - y0;

  float scale = min((float)kInputWidth / crop_w, (float)kInputHeight / crop_h);
  int scaled_w = crop_w * scale;
  int scaled_h = crop_h * scale;

  int pad_x = (kInputWidth - scaled_w) / 2;
  int pad_y = (kInputHeight - scaled_h) / 2;

  // For each pixel in the model input image
  for (int y = 0; y < kInputHeight; y++) {
    for (int x = 0; x < kInputWidth; x++) {
      int model_index = (y * kInputWidth + x) * 3;

      // Map to source image
      int src_x = (x - pad_x) / scale + x0;
      int src_y = (y - pad_y) / scale + y0;

      uint8_t r = 0, g = 0, b = 0;

      if (src_x >= x0 && src_x < x1 && src_y >= y0 && src_y < y1) {
        int src_index = (src_y * kFrameWidth + src_x) * 2;
        uint16_t pixel = frame_buffer[src_index] | (frame_buffer[src_index + 1] << 8);
        rgb565_to_rgb888(pixel, r, g, b);
      }

      input->data.int8[model_index + 0] = r - 128;
      input->data.int8[model_index + 1] = g - 128;
      input->data.int8[model_index + 2] = b - 128;
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // --- Camera setup ---
  if (!OV7675.begin(QQVGA, RGB565, 1)) {
    Serial.println("Camera init failed!");
    while (1);
  }

  // --- TensorFlow setup ---
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema mismatch");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup complete");
}

void loop() {
  //if (OV7675.readFrame(frame_buffer) != 0) {
  //  Serial.println("Image capture failed");
  //  return;
  //}

  if (kInjectedImageWidth * kInjectedImageHeight * 2 <= sizeof(frame_buffer)) {
    memcpy(frame_buffer, injected_image_data, kInjectedImageWidth * kInjectedImageHeight * 2);
    Serial.println("Injected image into frame_buffer.");
  } else {
    Serial.println("Error: Injected image size mismatch or buffer too small!");
    return;
  }

  int total_cars = 0;

  for (int i = 0; i < num_boxes; i++) {
    extract_and_resize_box(boxes[i]);

    unsigned long start = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
      error_reporter->Report("Invoke failed");
      continue;
    }
    unsigned long duration = millis() - start;

    int8_t score = output->data.int8[0];
    float confidence = (score + 128.0f) / 255.0f;
    
    total_cars += score;

    Serial.print("Box ");
    Serial.print(i);
    Serial.print(": score=");
    Serial.print(score);
    Serial.print(" confidence=");
    Serial.print(confidence);
    Serial.print(" time=");
    Serial.print(duration);
    Serial.println("ms");
  }
  Serial.println("Total cars: " + total_cars);

  delay(2000);
}
