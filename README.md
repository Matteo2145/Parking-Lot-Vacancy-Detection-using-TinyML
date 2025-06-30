# Parking-Lot-Vacancy-Detection-using-TinyML

The aim of this project is to implement a Parking Lot Vacancy Detection using Tiny Machine Learning on a ARDUINO Nano 33 BLE Same Lite given in the Tiny Machine Learning Kit. 
The dataset used for training and validation is: [Parking-Lot Dataset](https://public.roboflow.com/object-detection/pklot/2).

## How it works
Our model takes in input an image of a known parking lot, it is fundamental to know the position (bounding box) of each slot. The program resize the image accordingly to the specifics of our Microcrontroller Unit and performs an inference on every image representing a single slot. The detection performed gives in output the presence, or absence, of a car. 


