# Documentation

## Four different implementations:

1) Tensorflow TFOD pre-trained frozen models
2) Coco + Plain OpenCV2
3) Darknet and YOLO 
4) Tracker


## Technologies
 - Version	tensorflow_gpu - 2.5.0 
 - Python version -	3.6-3.9	
 - Compiler - 	MSVC 2019	
 - Build tools - Bazel 3.7.2	
 - cuDNN - 	8.1
	- CUDA - 11.2


## Collecting Images & Labelling
- labelling is done using labelImg (https://github.com/tzutalin/labelImg)
- collecting images via webcam
- labelling by drawing boxes in labelImg

#
 <img  width="80%" height="80%" align = "center" src ="https://miro.medium.com/max/1384/1*7R068tzqqK-1edu4hbAVZQ.png">
 (source: https://medium.com/analytics-vidhya/image-classification-with-mobilenet-cc6fbb2cd470)
 
 
## Installation
 - Make sure the GPU, VS version, TF version, cuDNN & CUDA versions are matching\
 as in: https://www.tensorflow.org/install/source_windows
 - python -m pip install --upgrade pip
 - pip install ipykernel (associate the environement with our jupyter notebook)
 - python -m ipykernel install --user --name=tfodj (install our virtual environment into jupyter)


## Documentation TFOD (Tensorflow Object Detection) -- Notes for creators:
 - Done with TFOD API
 - Leveraged Camera (Labelling and identifying the objects)
 - Training (Static & Dynamic-real time- Detection) is done via labelling
 - Data: Image ... Answers: Annotations/Labels ... Trained the ML model with these
 - Freeze (save&load) model
 - Export and Deploy (Reusable models) / Convert into 'tf' format
 - Perform tuning (solving wrongfully detected objects)
 - Training on Google Cloud (saves RAM)
 - TBT: Rasperry PI integration (coming in September)
 - CudaNN (and CUDA) gives GPU acceleration when doing model training (slower on CPU)
 - Create enviroment (isolate libs and dependencies) via 'python -m venv tfod' (create a notebook in Jupyter - run with 'jupyter notebook' cmd)

## Algorithms & Approaches for TFOD:
- MobileNet Convolutional NN model used\
(read more on: https://medium.com/analytics-vidhya/image-classification-with-mobilenet-cc6fbb2cd470)
- Depthwise Separable Convolutional Layering

## Steps
1) Images Collecting and Labelling (with LabelImg library) -> Collecting images with our webcam/ preprocessed videos
2) LabelImg -> can put squares and labels images manually
3) When labelling: Take (~20)pictures from different angles, lighting & rotation and as tight as possible on the object - the more **instances**, the better
   e.g. **side, front, back**, so that the model detects it easier

 
# Documentation for the other approaches
- Coming soon (or not, who knows)
