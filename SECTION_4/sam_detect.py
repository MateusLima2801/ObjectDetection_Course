import cv2
import torch
import math
import numpy as np
from numpy import random
import supervision as sv
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
from super_gradients.training import models
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = 'vit_h'

def load_sam():
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam

sam = load_sam()
cap=cv2.VideoCapture("/content/carvideo.mp4")
frame_width=int(cap.get(3))
frame_height=int(cap.get(4))

names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella","handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush" ]
out = cv2.VideoWriter('Output2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

def show_mask(mask, ax, random_color=False):
  if random_color:
      color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
  else:
      color = np.array([30/255, 144/255, 255/255, 0.6])
  h, w = mask.shape[-2:]
  mask_image = mask.reshape(h, w, 1)  * color.reshape(1, 1, -1)
  return mask_image

#Initialize Sam automatic mask generator parameters - > parameters can be modified for more optimized performance.
def mask_generator():

    mask_generator_ = SamAutomaticMaskGenerator(
        model=load_sam(),
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.96,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    return mask_generator_

def get_input_boxes(bboxes, class_ids, frame):
  for _, (box, class_id) in enumerate(zip(bboxes, class_ids)):
      label = names[class_id]
      c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
      cv2.rectangle(frame, c1, c2, color=(0, 28, 136), thickness=2, lineType=cv2.LINE_AA)
      if label:
          tl=2
          tf = max(tl - 1, 1)  # font thickness
          t_size = cv2.getTextSize(str(label), 0, fontScale=tl / 3, thickness=tf)[0]
          c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
          cv2.rectangle(frame, c1, c2, color=(0, 28, 136),thickness=-1, lineType=cv2.LINE_AA)  # filled
          cv2.putText(frame, str(label), (c1[0], c1[1] - 2), 0, tl / 3,
                      [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
  input_tensor_box = torch.tensor(bboxes, device=device)
  return input_tensor_box

def segment_object(bbox, frame, class_ids, filter_classes):
    img_rgb = frame.copy()
    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(frame)
    input_boxes = get_input_boxes(bbox,  class_ids, frame)
    transformed_boxes = mask_predictor.transform.apply_boxes_torch(input_boxes, img_rgb.shape[:2])
    masks, scores, logits = mask_predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)
    return masks

def draw_segmented_mask(anns, frame):
  img = frame.copy()
  mask_annotator = sv.MaskAnnotator(color=sv.Color.blue())
  detections = sv.Detections(
    xyxy=sv.mask_to_xyxy(masks=anns),
    mask=anns)
  detections = detections[detections.area == np.max(detections.area)]
  segmented_mask = mask_annotator.annotate(scene=frame, detections=detections)
  return segmented_mask

count = 0
while True:
    ret, frame = cap.read()
    count += 1
    if ret:
        result = list(model.predict(frame, conf=0.5))[0]
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()
        label = [int(labels) for labels in labels]
        print("Bounding Box Coorinates",bbox_xyxys)
        print("Scores", confidences)
        print("Class IDS", label)
        masks = segment_object(bbox_xyxys, frame, label, filter_classes=None)
        for mask in masks:
          segmented_mask = show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
          rearranged_mask = np.transpose(segmented_mask[:,:,:3], (2,0,1))
          frame = draw_segmented_mask(rearranged_mask, frame)
        out.write(frame)
        plt.close()
    else:
        break

out.release()
cap.release()