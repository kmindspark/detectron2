"""
Based heavily on: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
"""

import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
import pycocotools

from train import get_loop_dicts, get_config

def predict(image, thresh=0.90):
    # if image is a torch tensor, convert to cpu and numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # save image into cur_pred_dir/images
    cur_pred_dir = "./cur_pred_dir/images"
    if not os.path.exists(cur_pred_dir):
        os.makedirs(cur_pred_dir)
    image_path = os.path.join(cur_pred_dir, '00000.jpg')
    print("image to predict shape:")
    print(image.shape)
    plt.imsave(image_path, image.squeeze())

    cfg = get_config()

    # evaluate the model
    cfg.MODEL.WEIGHTS = "./output/model_06_20_940_iters.pth"  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh # used to be 0.90   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode

    dataset_dicts = get_loop_dicts("./cur_pred_dir")
    print("dataset_dicts:", dataset_dicts)
    for d in random.sample(dataset_dicts, 1):
        MetadataCatalog.get("./cur_pred_dir/").set(thing_classes=["knot"])
        loop_metadata = MetadataCatalog.get("./cur_pred_dir/")

        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        v = Visualizer(im[:, :, ::-1],
                    metadata=loop_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = out.get_image()[:, :, ::-1]
        print("OUTPUT VIZ SHAPE:")
        print(img.shape)
        return outputs["instances"].to("cpu").pred_boxes.tensor.numpy(), img

if __name__ == '__main__':
    os.mkdir("./cur_pred_dir", exist_ok=True)
    os.mkdir("./cur_pred_dir/images", exist_ok=True)
    test_images = "./datasets/detectron_massive/test/images"



