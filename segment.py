from ultralytics import YOLO

#Initialize YOLO with the Model Name
#model = YOLO("yolov8n.pt") # for detection
model = YOLO("yolov8n-seg.pt") # for segmentation

##Predict Method Takes all the parameters of the Command Line Interface
#model.predict(source='data/image1.jpg', save=True, conf=0.5, save_txt=True)
#model.predict(source='data/demo.mp4', save=True, conf=0.5, save_txt=True)
#model.predict(source='data/image1.jpg', save=True, conf=0.8, save_txt=True)
#model.predict(source='data/image1.jpg', save=True, conf=0.5, save_crop=True)
#model.predict(source='data/image1.jpg', save=True, conf=0.5, hide_labels=True, hide_conf = True)
model.predict(task='segment', source='/home/mateus/Desktop/ObjectDetection_Course/img0000001 (cópia).jpg', save=True, conf=0.5)

model.export(format="onnx")
