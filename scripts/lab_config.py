from mrcnn.config import Config
import numpy as np

############################################################
#  ROS Configurations
############################################################
FRAME_ID = "basler_ace_rgb_sensor_calibrated"
CAMERA_INFO_TOPIC = "/pylon_camera_node/camera_info"
IMAGE_TOPIC = "/pylon_camera_node/image_rect"
DEPTH_TOPIC = "/phoxi_camera/aligned_depth_map_rect"

############################################################
#  MaskRCNN Configurations
############################################################
def normalize_texture(texture):
    texture = texture.astype(np.float32)
    texture /= 255.0

    return texture

def normalize_texture2(texture):
    return texture

def normalize_depth(depth):
    min_depth = 0.0
    max_depth = 400.0
    depth = (depth - min_depth)  / (max_depth - min_depth)
    depth[np.where(depth < 0.0)] = 0.0
    depth[np.where(depth > 1.0)] = 0.0

    return depth

def normalize_depth2(depth):
    min_depth = 1000.0
    max_depth = 2000.0
    depth = (depth - min_depth)  / (max_depth - min_depth)
    depth[np.where(depth < 0.0)] = 0.0
    depth[np.where(depth > 1.0)] = 1.0

    return depth

NORMALIZE_TEXTURE = normalize_texture

NORMALIZE_DEPTH = normalize_depth

CLASS_NAMES = ['BG', 'obj']

INPUT_DATA_TYPE = "depth"
TRAINED_MODEL = "mask_rcnn_lab_depth_from_container.h5"
#INPUT_DATA_TYPE = "rgb"
#TRAINED_MODEL = "mask_rcnn_lab_choice.h5"

DEPTH_COORDINATE = "container" # or camera
# change coordinate matrix from camera to container
TRANS = np.asarray([-0.05529858, -0.03283987, 1.58937867])
ROT   = np.asarray([[-0.99975764, -0.01042886, -0.01938806],
                    [-0.01002908,  0.99973742, -0.02060383],
                    [ 0.01959784, -0.02040439, -0.99959971]])

class LabConfig(Config):
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
    NUM_CLASSES = len(CLASS_NAMES)

    IMAGE_CHANNEL_COUNT = int('rgb' in INPUT_DATA_TYPE) * 3 + int('gray' in INPUT_DATA_TYPE) * 1 + int('depth' in INPUT_DATA_TYPE) * 1

    MEAN_PIXEL = np.zeros(IMAGE_CHANNEL_COUNT)
    #MEAN_PIXEL = [123.7, 116.8, 103.9]

    DETECTION_MIN_CONFIDENCE = 0.5

    DETECTION_NMS_THRESHOLD = 0.3

