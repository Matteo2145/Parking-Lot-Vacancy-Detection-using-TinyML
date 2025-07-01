# Parking-Lot-Vacancy-Detection-using-TinyML

![2012-09-11_16_48_36_jpg rf 4ecc8c87c61680ccc73edc218a2c8d7d](https://github.com/user-attachments/assets/45310e9d-bc77-4c76-ab5b-d59522dbb253)


The aim of this project is to implement a Parking Lot Vacancy Detection using Tiny Machine Learning on a ARDUINO Nano 33 BLE Sense Lite (e.g. included in the Tiny Machine Learning Kit by Arduino).   
The dataset used for training and validation is: [Parking-Lot Dataset](https://public.roboflow.com/object-detection/pklot/2).

![image](https://github.com/user-attachments/assets/d3386dd5-e264-4dba-a1b0-0fe0281f54b1)


## How it works
Our model takes as input an image of a known parking lot, it is fundamental to know the position (bounding box) of each slot. The program resizes the image accordingly to the specifics of our Microcrontroller Unit and performs an inference on every individual parking slot of the image. The detection performed gives output according to the presence, or absence, of a car. 

## Requirements
Some packages/Apps need to be imported:  
-Pillow  
-[Arduino IDE](https://www.arduino.cc/en/software/)  
-[TensorFLowLite Library](https://github.com/tensorflow/tflite-micro-arduino-examples) Just copy this repository in the folder that contains your Arduino IDE libraries (for Linux/ Mac typically ~/Arduino/libraries).   
-Arduino_OV767X. This package can be easily installed directly from the Arduino IDE via the Library Manager.  

## Usage  
### How to connect the board
After connecting the board to your PC using the cable, open Arduino IDE environment. On the top left of the screen you can select the board you are using, select Arduino Nano 33 BLE. Then click Tool->Port and select the port to which the board is connected.   
## Selection of the image
Second step is selecting the photo that we want to analyze using the script *Parking-Lot-Vacancy-Detection-using-TinyML\inference\generate_injected_image.py*. The input argument is the path of the image. You can use an image from *Parking-Lot-Vacancy-Detection-using-TinyML\dataset\valid\images* for example.
The Python script is designed to convert an RGB image into a grayscale and downsized (to the expected size of the Microcontroller) image and then generate a C header file containing the image data as a byte array. 

 ![image](https://github.com/user-attachments/assets/1e33842b-d5c5-47d7-a51e-243c6aa51c0e) 

 Example input image(dataset/valid/images/2013-04-14_09_00_03_jpg.rf.8c861933e4ba9b29326ab2586a521c92.jpg) 



## Detection
Once the C header file is created it's time to start the detection. In the Arduino IDE open the folder *Parking-Lot-Vacancy-Detection-using-TinyML\inference\parking-lot-vacancy-detection*, and upload the file *parking-lot-vacancy-detection.ino*. Check that *injected_image.h* is located in the same folder as *parking-lot-vacancy-detection.ino*.  
If there is an error that says that the device is not connected, try to double click the reset button on the board.   
![Resized](https://github.com/user-attachments/assets/e24697b2-45a0-43c9-bf73-3c2e17624974)  
The original photo after resizing and greyscaling.  
![Results](https://github.com/user-attachments/assets/30798d12-e8e7-456a-9681-b47d943344bc)  
Serial Monitor's output of Arduino IDE (detected 6 out of 7 cars: 85% accurate).  
## Why the use of ARDUINO Nano 33 BLE Sense Lite
Let's start by presenting the specifications of our microcontroller: [Documentation](https://docs.arduino.cc/hardware/nano-33-ble-sense/). It gives the opportunity to develop and deploy embedded ML applications thanks to the 64 MHz Arm® Cortex®-M4F processor, which is suitable to run small models via frameworks like TensorFlow Lite for Microcontrollers. Another important feature of this processor is that it supports quantized models such as int8 quantization.\
A limitation of this microcontroller is memory, having 1MB flash and 256KB of RAM. This limitation forced us to create a model as tiny as possible while maintaining high accuracy. 


## Challenges and Future Features

![yolo_val](https://github.com/user-attachments/assets/beb59485-a05e-42f1-bcf9-6a714d027080)

As stated previously our main challenge was to create a light model. The first idea was to create a **Multiple Objects Detection** starting from a YOLO11n (You Only Look Once) model, the model has been trained in [Google Colab](https://colab.research.google.com/drive/1cFkwcUO_BYdpvcR7aoafKTgsltCv9rN1#scrollTo=o3bnmf2ZgfIC&line=1&uniqifier=1) using the dataset showed in the introduction. We obtained good results in terms of accuracy, the model was able to detect cars as well as empty slots (the image above represents our results using Yolo11n). The drawback of this solution was the size of the model, because it was impossible, even with int8 quantization, to obtain a model that could be deployed on our Arduino.  
This obstacle lead us to completely change the architecture of our model.  
To obtain a lighter model we abandoned the idea of performing a Multiple Objects Detection to focus on a single object. Therefore the model takes as input an image of the whole parking-lot and crops it creating an image for each lot. To make this important step we need every slot's bounding box, this information is already provided by the dataset. Thus the input image **can't** be a random parking-lot.  
The detection performed shows the presence or absence of a vehicle in a single parking slot, in the end by simply summing up the number of cars detected it is possible to decide if the parking lot is completely full or whether there are available slots.  
In the future this project can be optimized for other microcontrollers, the only thing needed is more memory! For example the training of a YOLO based model is easy and doesn't require much fine-tuning. The implementation can be done in many ways, for example exporting a YOLO model into an ONNX file or a TensorFlow library resulting in a size of around  5MB, making it feasible for many micro controllers on the market.
