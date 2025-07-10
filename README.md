# Parking Lot Vacancy Detection using TinyML

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A TinyML project to detect vacant parking spaces using an Arduino Nano 33 BLE Sense Lite and TensorFlow Lite.

This project takes a static image of a parking lot, processes it, and determines whether individual parking spaces are **occupied** or **vacant** directly on a low-power microcontroller.

![Project Demo](https://github.com/user-attachments/assets/45310e9d-bc77-4c76-ab5b-d59522dbb253)

---

## 📋 Table of Contents

* [About The Project](#-about-the-project)
* [Key Features](#-key-features)
* [How It Works](#-how-it-works)
* [Project Structure](#-project-structure)
* [Getting Started](#-getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
* [Usage](#-usage)
* [Project Journey](#-project-journey)
    * [Challenges Faced](#challenges-faced)
    * [Future Enhancements](#future-enhancements)
* [Acknowledgements](#-acknowledgements)

---

## <a name="about-the-project"></a>💡 About The Project

The goal of this project is to implement a real-world computer vision application on a highly constrained device by breaking down the problem into multiple binary classification instances and using Tiny Machine Learning to solve each. We use an **Arduino Nano 33 BLE Sense Lite** to run a TensorFlow Lite model that classifies parking spaces.

The model was trained and validated using the [PKLot Dataset on Roboflow](https://public.roboflow.com/object-detection/pklot/2).

![Dataset Sample](https://github.com/user-attachments/assets/d3386dd5-e264-4dba-a1b0-0fe0281f54b1)

---

## ✨ Key Features

* **On-Device Inference**: All processing happens directly on the Arduino. No internet connection required for analysis.
* **Low Power**: Designed for the energy-efficient Arm® Cortex®-M4F processor.
* **Memory Optimized**: The model and workflow are tailored to the 256KB of RAM and 1MB of flash memory on the Arduino.
* **Customizable**: Can be adapted to monitor any parking lot, provided the coordinates of the spaces are known.

---

## ⚙️ How It Works

Instead of a complex object detection model like YOLO, this project uses a more efficient approach suitable for microcontrollers:

1.  **Known Layout**: The system is designed for a **specific parking lot** where the bounding box (position) of each parking space is known beforehand.
2.  **Image Pre-processing**: A Python script takes a high-resolution image, converts it to grayscale, and downsizes it.
3.  **Individual Slot Classification**: The Arduino code iterates through the pre-defined coordinates of each parking space. It crops the tiny image of each slot and feeds it to a classification model.
4.  **Binary Output**: The model determines if the slot contains a car ("Occupied") or not ("Vacant").
5.  **Final Count**: The results for all slots are aggregated to provide a final count of available spaces.

---

## <a name="project-structure"></a>📂 Project Structure

The repository is organized as follows:

| File / Folder                        | Description                                  |
| ------------------------------------ | -------------------------------------------- |
| `dataset/`                           | Contains the image dataset for training, test and validation.   |
| `inference/`                         | Scripts and Arduino code for running the model. |
| `├─ parking-lot-vacancy-detection/` | Arduino sketch folder.                       |
| `└─ generate_injected_image.py`      | Python script to prepare images for Arduino. |
| `report/`                            | Project report.       |
| `train/`                             | Scripts for training the model (manual) as well as artifacts for experimenting with Yolov11n (yolo).  |
| `README.md`                          | Starting point for exploring the repository.         |

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

You will need the following hardware and software:

* **Hardware**
    * Arduino Nano 33 BLE Sense Lite
* **Software / Libraries**
    * [Arduino IDE](https://www.arduino.cc/en/software/)
    * Python 3.x
    * `tensorflow`
    * `Pillow`
    * `ultralytics` (Optional, only if you wish to experiment with YOLO)

### Installation

1.  **Clone the Repository**
    ```sh
    git clone [https://github.com/your-username/Parking-Lot-Vacancy-Detection-using-TinyML.git](https://github.com/your-username/Parking-Lot-Vacancy-Detection-using-TinyML.git)
    cd Parking-Lot-Vacancy-Detection-using-TinyML
    ```

2.  **Install Python Packages**
    It's recommended to use a virtual environment.
    ```sh
    pip install tensorflow Pillow ultralytics
    ```

3.  **Set up Arduino IDE**
    * Install the `Arduino_OV767X` library via the Arduino IDE Library Manager (`Tools` > `Manage Libraries...`).
    * Manually add the TensorFlow Lite library by copying the [tflite-micro-arduino-examples](https://github.com/tensorflow/tflite-micro-arduino-examples) repository into your Arduino libraries folder (e.g., `~/Arduino/libraries` on Linux/Mac).

---

## 🛠️ Usage

Follow these steps to run the detection on a new image.

1.  **Prepare the Input Image**
    * Run the `generate_injected_image.py` script from the `inference` folder to convert your chosen parking lot image into a C header file. This script handles the resizing and grayscaling.
    * Use an image from the `dataset/valid/images/` directory or your own.

    ```sh
    cd inference
    python generate_injected_image.py path/to/your/image.jpg
    ```
    * This will create a file named `injected_image.h` in the subdirectory `parking-lot-vacancy-detection`.

    | Original Input Image                                                                                              | Processed Image for Arduino                                                                               |
    | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
    | ![Example input image](https://github.com/user-attachments/assets/1e33842b-d5c5-47d7-a51e-243c6aa51c0e) | ![Resized image](https://github.com/user-attachments/assets/e24697b2-45a0-43c9-bf73-3c2e17624974) |

2.  **Deploy to Arduino**
    * Open `inference/parking-lot-vacancy-detection/parking-lot-vacancy-detection.ino` in the Arduino IDE.
    * Connect your Arduino Nano 33 BLE Sense Lite to your PC.
    * Select the correct board (`Arduino Nano 33 BLE`) and port from the `Tools` menu.
    * Click **Upload**. If you encounter a connection error, try double-tapping the reset button on the board.

3.  **View the Results**
    * Open the **Serial Monitor** (`Tools` > `Serial Monitor`) with a baud rate of `9600`.
    * The Arduino will output the classification by printing the sum of detected cars for the injected image in the terminal

    ![Serial Monitor Output](https://github.com/user-attachments/assets/30798d12-e8e7-456a-9681-b47d943344bc)  
    *In this example, the model achieved 85% accuracy, correctly identifying 6 out of 7 cars.*

---

## 🗺️ Project Journey

### Challenges Faced

The primary challenge was the **memory limitation** of the Arduino (1MB flash, 256KB RAM).

Our initial approach was to use a YOLOv11n object detection model. While it performed well in development (detecting both cars and empty spaces), the quantized TensorFlow Lite model was still too large to deploy on the microcontroller.

![YOLO Validation](https://github.com/user-attachments/assets/beb59485-a05e-42f1-bcf9-6a714d027080)

This limitation forced a pivot from a single, complex model to the current, more efficient architecture of classifying individual, pre-defined slots.

### Future Enhancements

* **More Powerful Hardware**: Port the project to a microcontroller with more memory (e.g., Raspberry Pi Pico, ESP32-S3) to enable the use of a full object detection model like YOLO. This would remove the need for pre-defined slot coordinates.
* **Live Video Feed**: Adapt the code to use a live feed from an OV7675 camera instead of a static image.
* **Model Optimization**: Further refine the model's accuracy and performance.
* **Enhance Dataset**: Diversify the dataset by also adding images of e.g. people standing on the parking slot to reduce the number of false positive results.

---


## 🙏 Acknowledgements

* [Roboflow](https://roboflow.com/) for providing the PKLot dataset.
* [TensorFlow](https://www.tensorflow.org/) for the framework and tools.
* [Arduino](https://www.arduino.cc/) for the hardware and development environment.
