# Parking-Lot-Vacancy-Detection-using-TinyML

The aim of this project is to implement a Parking Lot Vacancy Detection using Tiny Machine Learning on a ARDUINO Nano 33 BLE Sense Lite given in the Tiny Machine Learning Kit.\ 
The dataset used for training and validation is: [Parking-Lot Dataset](https://public.roboflow.com/object-detection/pklot/2).

## How it works
Our model takes in input an image of a known parking lot, it is fundamental to know the position (bounding box) of each slot. The program resize the image accordingly to the specifics of our Microcrontroller Unit and performs an inference on every image representing a single slot. The detection performed gives in output the presence, or absence, of a car. 

## Deployement

## Why the use of ARDUINO Nano 33 BLE Sense Lite
Let's start by presenting the specifications of our microcontroller: ([Documentation](https://docs.arduino.cc/hardware/nano-33-ble-sense/)). It gives the opportunity to develope and deploy embedded ML applications thanks to the 64 MHz Arm® Cortex®-M4F processor, ehich is suitable to run small models and uses afficiently frameworks like TensorFlow Lite for Microcontrollers. Another important feature of this processor is that its support quantized models such as int8 quantization.\
A limitation of this microcontroller is the limited amount of memory, having 1MB flash and 256KB of RAM. This limitation forced us to create a model as tiny as possible having a good value of accuracy.

## Challenges and Future Features
As stated previously our main challenge was to create a light model. The first idea was to create a **Multiple Objects Detection** starting from a YOLO11n model, the model has been trained in Google Collab using the dataset showed in the introduction. We obtained good results in terms of accuracy, the model was able to detect cars as well as empty slots. The drawback of this solution was the size of the model, because it was impossible, even with int8 quantization, to obtain a model that could be deployed on our Arduino.\ 
This obstacle lead us to completely change the architecture of our model.\ 
To obtain a lighter model we abandoned the idea of performing a Multiple Objects Detection to focus on a single object. Therefore the model takes in input an image of the whole parking-lot and crops it creating an image for each lot. To make this important step we need every slot's bounding box, this information is already provided by the dataset. Thus the input image **can't** be a random parking-lot.\ 









