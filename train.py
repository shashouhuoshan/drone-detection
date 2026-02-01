from datetime import datetime

from ultralytics import YOLO
import os
import shutil
import random
import yaml
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_dataset():
    return

def split_dataset(path:str, base_dir:str):
    files = [ f for f in os.listdir(path) if f.endswith(".jpg") ]
    if not len(files):
        print(f"[ERROR]: no images in {path}")
        return
    labels = []
    images = []
    for file in files:
        label = file.replace(".jpg",".txt")
        if os.path.exists(os.path.join(path, label)):
            images.append(file)
            labels.append(label)

    print(f"{len(images)}-{len(labels)} items in data")

    train_images,  test_images, train_labels, test_labels = train_test_split(
        images,labels, train_size=0.8, random_state=42
    )

    val_images, test_images, val_labels, test_labels = train_test_split(
        test_images, test_labels, test_size=0.4, random_state=42
    )

    print(f"train:{len(train_images)}-{len(train_labels)}\n"
          f"valid:{len(val_images)}-{len(val_labels)}\n"
          f"test:{len(test_images)}-{len(test_labels)}\n")

    def copy_data(imgs, lbls, sp):
        for i, l in zip(imgs, lbls):
            shutil.copy(os.path.join(path,i), os.path.join(base_dir,sp,"images",i))
            shutil.copy(os.path.join(path,l), os.path.join(base_dir,sp,"labels",l))

    for split, image, label in zip(["train","val","test"],
                                   (train_images, val_images, test_images),
                                   (train_labels, val_labels, test_labels)):
        os.makedirs(f"{base_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{base_dir}/{split}/labels", exist_ok=True)
        copy_data(image, label, split)

    data = {
        'train': os.path.join(os.path.abspath(base_dir), 'train', 'images'),
        'val': os.path.join(os.path.abspath(base_dir), 'val', 'images'),
        'test': os.path.join(os.path.abspath(base_dir), 'test', 'images'),
        'nc': 1,
        'names':['drone']
    }

    with open(os.path.join(base_dir,'yolo_dataset.yaml'),"w") as file:
        yaml.dump(data, file)

    print(f"yaml config created: {base_dir}/yolo_dataset.yaml")

def augment_images(img_dir, label_dir):
    augmentor = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.3),
        A.Rotate(limit=15, p=0.4, border_mode=cv2.BORDER_REFLECT_101),
        A.Blur(blur_limit=3, p=0.2),
        A.HueSaturationValue(p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    num_aug = 1

    def clip_boxes(bboxes):
        clipped = []
        for box in bboxes:
            x, y, w, h = [max(0, min(1, v)) for v in box]
            clipped.append([x, y, w, h])
        return clipped

    for img_name in os.listdir(img_dir):
        if not img_name.endswith(".jpg"):
            continue

        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

        image = cv2.imread(img_path)
        if image is None or not os.path.exists(label_path):
            continue

        with open(label_path) as f:
            boxes, classes = [], []
            for line in f.readlines():
                vals = line.strip().split()
                if len(vals) != 5:
                    continue
                cls, x, y, bw, bh = map(float, vals)
                boxes.append([x, y, bw, bh])
                classes.append(int(cls))

        for i in range(num_aug):
            try:
                augmented = augmentor(image=image, bboxes=boxes, class_labels=classes)
            except ValueError:
                continue

            aug_img = augmented["image"]
            aug_boxes = clip_boxes(augmented["bboxes"])
            aug_classes = augmented["class_labels"]

            new_name = img_name.replace(".jpg", f"_aug{i}.jpg")
            cv2.imwrite(os.path.join(img_dir, new_name), aug_img)

            with open(os.path.join(label_dir, new_name.replace(".jpg", ".txt")), "w") as f:
                for cls, (x, y, bw, bh) in zip(aug_classes, aug_boxes):
                    f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

    print("✅ Augmentation completed successfully!")


def show_bbox_distribution(dataset):
    widths, heights = [], []
    label_path = os.path.join(dataset, "labels")
    # image_path = os.path.join(dataset, "images")

    for label_file in os.listdir(label_path):
        if not label_file.endswith(".txt"):
            continue
        with open(os.path.join(label_path, label_file)) as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                widths.append(w)
                heights.append(h)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=30, color='skyblue')
    plt.title("Distribution of bbox widths (normalized)")

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=30, color='salmon')
    plt.title("Distribution of bbox heights (normalized)")

    plt.show()

def show_image_with_bbox(dataset):
    img_file = random.choice(os.listdir(os.path.join(dataset,"images")))
    if not img_file.endswith(".jpg"):
        return
    img_path = os.path.join(dataset,"images", img_file)
    label_path = os.path.join(dataset, "labels", img_file.replace('.jpg','.txt'))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(label_path) as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            h_img, w_img, _ = img.shape
            x1 = int((x - w/2) * w_img)
            y1 = int((y - h/2) * h_img)
            x2 = int((x + w/2) * w_img)
            y2 = int((y + h/2) * h_img)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def train_model_yolo(project_name, data_cfg, model):

    # ok
    # model.train(
    #     data=data_cfg,
    #     epochs=70,
    #     batch=16,
    #     imgsz=512,
    #     project='drone_detection',
    #     name='yolo_drone',
    #     exist_ok=True,
    #     save=True,
    #     save_period=5,
    #     patience=8,
    #     optimizer='AdamW',
    #     lr0=0.0005,
    #     lrf=0.1,
    #     momentum=0.9,
    #     weight_decay=0.0005,
    #     warmup_epochs=3,
    #     warmup_momentum=0.8,
    #     warmup_bias_lr=0.1,
    #     cache='ram',
    #     device='mps',
    #     workers=8,
    #     deterministic=True,
    #     val=True,
    #     plots=True,
    #     amp=True,
    #     multi_scale=False,
    #     close_mosaic=10
    # )

    model.train(
        data=data_cfg,
        epochs=70,
        batch=16,
        imgsz=512,
        project='drone_detection',
        name=project_name,
        exist_ok=True,
        save=True,
        save_period=5,
        patience=8,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cache='ram',
        device='mps',
        workers=8,
        deterministic=False,
        val=True,
        plots=True,
        amp=True,
        multi_scale=False,
        close_mosaic=10
    )


    print(f"complete!")
    csv_path = f"./runs/detect/drone_detection/{project_name}/results.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        table = df[['epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                    'metrics/precision(B)', 'metrics/recall(B)',
                    'metrics/mAP50(B)', 'metrics/mAP50-95(B)']]

        table = table.groupby('epoch', as_index=False).mean()
        print(table)
    else:
        print("results.csv not found. Make sure training is finished and path is correct.")


def valid_model_yolo(data_cfg, model):
    test_metrics = model.val(
        data=data_cfg,
        split='test',
        imgsz=512,
        batch=8,
        save=False,
        verbose=False
    )

    precision, recall, map50, map50_95 = test_metrics.mean_results()

    print("\n-------------------------------------\n")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"mAP50: {map50:.3f}")
    print(f"mAP50-95: {map50_95:.3f}")
    print("\n-------------------------------------\n")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    basedir = os.path.abspath("./data")
    dataset = os.path.abspath("./drone-dataset/drone_dataset_yolo/dataset_txt")
    if not os.path.exists(os.path.join(basedir,"train")):
        split_dataset(dataset, basedir)
        augment_images(f"{basedir}/train/images", f"{basedir}/train/labels")
        show_bbox_distribution(os.path.join(basedir,"train"))

    show_image_with_bbox(os.path.join(basedir,"train"))
    show_image_with_bbox(os.path.join(basedir,"test"))
    show_image_with_bbox(os.path.join(basedir,"val"))

    model = YOLO("yolo26s.pt")
    # model = YOLO("yolo26n.pt")
    dataset_cfg = os.path.join(basedir, "yolo_dataset.yaml")
    pr = 'yolo26s_drone_s512_b16_lr0.001'
    train_model_yolo(pr, dataset_cfg, model)
    valid_model_yolo(dataset_cfg, model)

    model.save(f"yolo26s_drone_{datetime.now().strftime('%Y%m%d%H%M')}.pt")
