from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import os
import glob
from IPython.display import Image, display
from IPython import display
display.clear_output()

model = YOLO("/home/mateus/Desktop/ObjectDetection_Course/SECTION_3/PEN_BOOK_DETECTION/data/best.pt")
model.predict(source="0", show=True, conf=0.15)