#include <Arduino_OV767X.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_allocator.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <cmath> 

#include "parking_model_data.h"
#include "boxes.h"             
#include "injected_image.h"    

extern unsigned char parking_model_quant_tflite[];

OV767X camera;

// --- TensorFlow Lite globals ---
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

constexpr int kInputWidth = 48;
constexpr int kInputHeight = 48;
constexpr int kInputChannels = 1; // Grayscale

// --- Camera resolution ---
constexpr int kFrameWidth = 160;
constexpr int kFrameHeight = 120;

uint8_t frame_buffer[kFrameWidth * kFrameHeight]; // Grayscale: 1 byte per pixel

void extract_and_resize_box(const Box& box) {
  // Convert box from 640x640 normalized to pixel coords on 160x120 camera frame
  int cx_frame = box.x_center * kFrameWidth;
  int cy_frame = box.y_center * kFrameHeight;
  int w_frame = box.width * kFrameWidth;
  int h_frame = box.height * kFrameHeight;

  int x0_frame = max(0, cx_frame - w_frame / 2);
  int y0_frame = max(0, cy_frame - h_frame / 2);
  int x1_frame = min(kFrameWidth - 1, cx_frame + w_frame / 2);
  int y1_frame = min(kFrameHeight - 1, cy_frame + h_frame / 2);

  int crop_w = x1_frame - x0_frame;
  int crop_h = y1_frame - y0_frame;

  // Scale the cropped region to fit the model's input size (48x48)
  float scale_x_model = (float)crop_w / kInputWidth;
  float scale_y_model = (float)crop_h / kInputHeight;

  // For each pixel in the model input image (48x48 grayscale)
  for (int y = 0; y < kInputHeight; y++) {
    for (int x = 0; x < kInputWidth; x++) {
      int model_index = (y * kInputWidth + x);

      // Map to source image (cropped camera frame)
      int src_x = (int)(x * scale_x_model) + x0_frame;
      int src_y = (int)(y * scale_y_model) + y0_frame;

      uint8_t grayscale_pixel = 0;

      // Ensure we are within the bounds of the actual cropped region
      if (src_x >= x0_frame && src_x < x1_frame && src_y >= y0_frame && src_y < y1_frame) {
        int src_index = (src_y * kFrameWidth + src_x);
        grayscale_pixel = frame_buffer[src_index];
      }

      // Quantize the grayscale value.
      input->data.int8[model_index] = grayscale_pixel - 128;
    }
  }
}

// Sigmoid function for output interpretation
float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}


void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!camera.begin(QQVGA, GRAYSCALE, 1)) {
     Serial.println("Camera init failed!");
     while (1);
  }
  
  model = tflite::GetModel(parking_model_quant_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddFullyConnected();


  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Verify input and output tensor properties match quantized model
  // Input: 48x48x1, Int8
  if (input->type != kTfLiteInt8 || input->dims->data[1] != kInputHeight ||
      input->dims->data[2] != kInputWidth || input->dims->data[3] != kInputChannels) {
    MicroPrintf("Input tensor shape or type mismatch!");
    return;
  }
  // Output: 1, Int8 (logit)
  if (output->type != kTfLiteInt8 || output->dims->data[1] != 1) {
    MicroPrintf("Output tensor shape or type mismatch!");
    return;
  }

  Serial.println("Setup complete");
}

void loop() {
  //if (camera.readFrame(frame_buffer) != 0) {
  //  Serial.println("Image capture failed");
  //  return;
  //}

  memcpy(frame_buffer, injected_image_data, kFrameWidth * kFrameHeight);
  Serial.println("Injected grayscale image into frame_buffer.");

  int occupied_parking_spots = 0;
  const float kPredictionThreshold = 0.5f;

  for (int i = 0; i < num_boxes; i++) {
    extract_and_resize_box(boxes[i]);

    unsigned long start = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
      MicroPrintf("Invoke failed.");
      continue;
    }
    unsigned long duration = millis() - start;

    float dequantized_output = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
    float confidence = sigmoid(dequantized_output);

    if (confidence > kPredictionThreshold) {
      occupied_parking_spots++;
  }
  Serial.print("Total occupied parking spots: ");
  Serial.println(occupied_parking_spots);

  delay(5000);
}