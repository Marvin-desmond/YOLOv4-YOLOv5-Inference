from torch_utils.yolo_torch_utils import (
    model_init,
    loadImages,
    showImg,
    preprocessAndInference,
    non_max_suppression
)
import torch
import cv2
import time
import os
import glob
import sys
from numpy import random
device = torch.device('cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# https://github.com/ultralytics/yolov5/releases
model = model_init(id_=0)

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
classes = open("../coco.names").read().strip().split('\n')

paths_and_images = loadImages("../inputs/")

for path, image in paths_and_images:
    total_detections = []
    then = time.time()
    predictions = preprocessAndInference(image, model)
    now = time.time()
    print("Inference time: {:.3f} seconds".format(now-then))
    predictions = non_max_suppression(predictions)[0]
    for prediction in predictions:
        total_detections.append([path, prediction.int().numpy()[:4].tolist()])
        cls = prediction[-1]
        box = prediction[:4]
        box = box.int().numpy()
        cv2.rectangle(image,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    color=colors[int(cls)],
                    thickness=2)
        cv2.putText(image,
                    f"{classes[int(cls)]}",
                    (box[:2][0], box[:2][1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
    showImg(image)
