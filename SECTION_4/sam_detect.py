import cv2
import torch
import math
import numpy as np
from numpy import random
import supervision as sv
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
# from super_gradients.training import models
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = 'vit_h'

def load_sam():
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam
