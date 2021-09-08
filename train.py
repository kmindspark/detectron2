import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode

def get_loop_dicts(img_dir):
    """
    Return a list of dictionaries that contain the information for each image
    in the dataset.
    """
    # Get the list of images in the dataset
    img_list = os.listdir(img_dir)
    img_list = [img_name for img_name in img_list if img_name.endswith(".jpg")]
    img_list.sort()

    dataset_dicts = []

    # Create a dictionary for each image and its corresponding annots file
    for i, img_name in enumerate(img_list):
        record = {}
        objs = []
        
        # read image and corresponding annotation
        img_path = os.path.join(img_dir, 'images', img_name)
        height,width = cv2.imread(img_path).shape[:2]

        record['image_id'] = i
        record['file_name'] = img_name
        record['height'] = height
        record['width'] = width

        # read annotation
        annots_path = os.path.join(img_dir, 'annots', img_name.replace(".jpg", ".xml.txt"))
        with open(annots_path, 'r') as f:
            for line in f:
                # split line by space
                centerx, centery, width, height = list(map(lambda x: float(x), line.split()))
                xmin, xmax = centerx - width/2, centerx + width/2
                ymin, ymax = centery - height/2, centery + height/2

                obj = {
                    'bbox': [xmin, ymin, xmax, ymax],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': 0,
                }
                objs.append(obj)
            record['annotations'] = objs
        dataset_dicts.append(record)

        return dataset_dicts

for d in ['train', 'val']:
    DatasetCatalog.register("loop_" + d, lambda d=d: get_loop_dicts("loop/" + d))
    MetadataCatalog.get("loop_" + d).set(thing_classes=["loop"])

loop_metadata = MetadataCatalog.get("loop_train")

# verify dataset loading
dataset_dicts = get_loop_dicts("loop/train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d['file_name'])
    visualizer = Visualizer(img[:,:,::-1], metadata=loop_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite("loop_train_sample.jpg", out[:,:,::-1])