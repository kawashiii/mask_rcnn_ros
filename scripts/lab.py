"""
Mask R-CNN
Train on the toy lab dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

LAB_DIR = os.path.join(ROOT_DIR, "lab")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class_names = ['BG', 'obj']

############################################################
#  Configurations
############################################################


class LabConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "lab"

    # Number of images to train with on each GPU. 
    # A 12GB GPU can typically handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. 
    # Use the highest number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    # Batch size = IMAGES_PER_GPU * GPU_COUNT
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = len(class_names)
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3

