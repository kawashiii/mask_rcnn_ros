#!/usr/bin/env python3
import os
import sys
import time
import threading
import numpy as np

import rospy
#from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from std_msgs.msg import UInt8MultiArray
from mask_rcnn_ros.msg import Result
#from cv_bridge import CvBridge

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import lab
import cv2
from cv_bridge import CvBridge

from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

ROOT_DIR = os.path.abspath("src/mask_rcnn_ros")
MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_lab.h5")
TEST_IMG = os.path.join(ROOT_DIR, "test.png")

CLASS_NAMES = ['BG', 'caloriemate', 'koiwashi', 'fabrise', 'saratekt', 'cleanser', 'jerry', 'dishcup', 'bottle']

class InferenceConfig(lab.LabConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRCNNNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        self._model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)
        self._model.load_weights(MODEL_PATH, by_name=True)

        self._last_msg = None
        self._msg_lock = threading.Lock()

        # self._class_names = rospy.get_param('~class_names', CLASS_NAMES)
        # self._visualization = rospy.get_param('~visualization', True)
        # self._publish_rate = rospy.get_param('~publish_rate', 100)
        self._class_names = CLASS_NAMES
        self._visualization = True
        self._publish_rate = 100
        self._class_colors = visualize.random_colors(len(CLASS_NAMES))

    def run(self):
        self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
        vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        sub = rospy.Subscriber("/camera/color/image_raw", Image, self._image_callback, queue_size=1)
        # sub = rospy.Subscriber('~input', Image, self._image_callback, queue_size=1)

        rate = rospy.Rate(self._publish_rate)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                np_image = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
                np_image = np_image[:, 420:1500]

                # Run detection
                results = self._model.detect([np_image], verbose=0)
                result = results[0]
                result_msg = self._build_result_msg(msg, result)
                self._result_pub.publish(result_msg)

                # Visualize results
                if self._visualization:
                    vis_image = self._visualize(result, np_image)
                    cv_result = np.zeros(shape=vis_image.shape, dtype=np.uint8)
                    cv2.convertScaleAbs(vis_image, cv_result)
                    image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                    vis_pub.publish(image_msg)

            rate.sleep()

    def run2(self):
        self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
        vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        sub = rospy.Subscriber("/camera/color/image_raw", Image, self._image_callback, queue_size=1)
        
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            np_image = cv2.imread(TEST_IMG)
            np_image = np_image[:, 420:1500]
            
            # Run detection
            t1 = time.time()
            results = self._model.detect([np_image], verbose=0)
            t2 = time.time()
            result = results[0]
            # result_msg = self._build_result_msg(msg, result)
            # self._result_pub.publish(result_msg)
            
            # Print detection time
            detection_time = t2 - t1
            print("Detection time: ", round(detection_time, 2), " s")
            
            # Visualize results
            if self._visualization:
                vis_image = self._visualize(result, np_image)
                cv_result = np.zeros(shape=vis_image.shape, dtype=np.uint8)
                cv2.convertScaleAbs(vis_image, cv_result)
                image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                vis_pub.publish(image_msg)
                print("Published detected image!")
                
            rate.sleep()

    def _build_result_msg(self, msg, result):
        result_msg = Result()
        result_msg.header = msg.header
        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
            box = RegionOfInterest()
            box.x_offset = np.asscalar(x1)
            box.y_offset = np.asscalar(y1)
            box.height = np.asscalar(y2 - y1)
            box.width = np.asscalar(x2 - x1)
            result_msg.boxes.append(box)

            class_id = result['class_ids'][i]
            result_msg.class_ids.append(class_id)

            class_name = self._class_names[class_id]
            result_msg.class_names.append(class_name)

            score = result['scores'][i]
            result_msg.scores.append(score)

            mask = Image()
            mask.header = msg.header
            mask.height = result['masks'].shape[0]
            mask.width = result['masks'].shape[1]
            mask.encoding = "mono8"
            mask.is_bigendian = False
            mask.step = mask.width
            mask.data = (result['masks'][:, :, i] * 255).tobytes()
            result_msg.masks.append(mask)
        return result_msg

    def _visualize(self, result, image):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axes = fig.gca()
        visualize.display_instances(image, result['rois'], result['masks'],
                                    result['class_ids'], CLASS_NAMES,
                                    result['scores'], ax=axes)
        fig.tight_layout()
        canvas.draw()
        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        _, _, w, h = fig.bbox.bounds
        result = result.reshape((int(h), int(w), 3))
        return result

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()

def main():
    rospy.init_node('mask_rcnn')

    node = MaskRCNNNode()
    # node.run()
    node.run2()

if __name__ == '__main__':
    main()
