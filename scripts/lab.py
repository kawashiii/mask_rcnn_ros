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

EPOCH = 50

class_names = [
    'BG', 'caloriemate', 'koiwashi', 'fabrise', 'saratekt',
    'cleanser', 'jerry', 'dishcup', 'bottle'
]

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
    IMAGES_PER_GPU = 2

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    # Batch size = IMAGES_PER_GPU * GPU_COUNT
    GPU_COUNT = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    # ??? Why this doesn't need to match the size of the training set ???
    STEPS_PER_EPOCH = 100

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.8


############################################################
#  Dataset
############################################################

class LabDataset(utils.Dataset):

    def load_lab(self, dataset_dir, subset):
        """Load a subset of the Lab dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        for i, name in enumerate(class_names):
            if name == "BG": continue
            self.add_class("lab", i, name)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())

        # The VIA tool saves images in the JSON even if they don't have any annotations.
        # Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        label = {}
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                names = [r['region_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
                names = [r['region_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read the image.
            # This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "lab",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons, names=names)

            for name in names:
                if label.get(name["name"]):
                    label[name["name"]] += 1
                else:
                    label[name["name"]] = 1

        print("The number of ", subset, "image: ", len(self.image_info))
        print("The number of ", subset, "label: ", sum(label.values()))
        for k, v in sorted(label.items()):
            print(" The number of '", k, "' label: ", v)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a lab dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "lab":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = []
        for i, (p, n) in enumerate(zip(info["polygons"], info["names"])):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            class_id = class_names.index(n["name"])
            class_ids.append(class_id)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "lab":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = LabDataset()
    dataset_train.load_lab(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = LabDataset()
    dataset_val.load_lab(args.dataset, "val")
    dataset_val.prepare()

    # Start training
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=EPOCH,
                layers='heads')

    model_path = os.path.join(LAB_DIR, "mask_rcnn_lab.h5")
    model.keras_model.save_weights(model_path)

def detect(model, image_path):
    print("Write detect code")
    # Detect code

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect lab data.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/lab/dataset/",
                        help='Directory of the lab dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to detect by learned model')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image, "Argment --image is required for detection"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = LabConfig()
    else:
        class InferenceConfig(LabConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect(model, args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
