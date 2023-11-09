import os
import glob
from IPython.display import Image, display
from IPython import display
import ultralytics

display.clear_output()

#HOME = os.getcwd()

#ultralytics.checks()
model = ultralytics.YOLO("yolov8m.pt") # for segmentation
model.train(batch=10, data='/home/mateus/Desktop/ObjectDetection_Course/SECTION_3/POT_HOLES_DETECTION/data/annotated/data.yaml', epochs=10, imgsz=640)
model.export(format="onnx")
