
from datetime import datetime

from ultralytics import YOLO
import os
import shutil
import random
import yaml
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import pandas as pd
import numpy as np



def show_model_predict_one(model, img_path, ax, class_names):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.predict(img_path)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    ax.imshow(img_rgb)
    ax.axis('off')
    ax.set_title(os.path.basename(img_path), fontsize=12, weight='bold')

    colors = plt.cm.get_cmap('tab20', len(class_names))

    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=colors(cl), facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(0, y1 - 5),
            f'{class_names[cl]} {score:.2f}',
            color='white', fontsize=10, weight='bold',
            backgroundcolor='black', alpha=0.6
        )


def show_model_predict(model, test_dir):
    test_imgs = os.listdir(test_dir)
    random_imgs = random.sample(test_imgs, min(4, len(test_imgs)))
    class_names_dict = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
        80: "drone"
    }
    class_names = [ v for k, v in class_names_dict.items() ]

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    for img_path, ax in zip(random_imgs, axs.flatten()):
        full_path = os.path.join(test_dir, img_path)
        show_model_predict_one(model, full_path, ax, class_names)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model = YOLO("yolo26s_drone_202601310945.pt")
    show_model_predict(model, "./data/test/copy/")
    print("predict")