from ultralytics import YOLO
import cv2
import math


"""

# Load a model
model = YOLO('yolov8n.yaml') # build a new model from YAML
model = YOLO('yolov8n.pt') # load a pretrained model 
model = YOLO('yolov8n.pt').load('yolov8n.pt') # build from YAML and transfer weights

"""
model = YOLO()

# Train the model
model.train(data='coco128.yaml', epochs=1, imgsz=640) # I don't have any GPU so not using 'device' argument, 
                                                      # since using CPU, low epochs, epochs can be increased as needed
                                                      # can be trained using other datasets (instead of coco128)
