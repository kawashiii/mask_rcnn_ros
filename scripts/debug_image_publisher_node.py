#!/usr/bin/python3

import os
import sys
import time

import numpy as np
import rospy
import rosparam
import roslib.packages
from sensor_msgs.msg import Image

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from cv_bridge import CvBridge

cv_bridge = CvBridge()
ROOT_DIR = os.path.abspath(roslib.packages.get_pkg_dir('mask_rcnn_ros'))

def main():
    rospy.init_node('debug_mrcnn_image_publisher')

    param, _ = rosparam.load_file(ROOT_DIR + "/config/mask_rcnn.yaml")[0]
    used_camera = param[param["input_camera"]]

    image_publisher = rospy.Publisher("/debug" + used_camera["image_topic"], Image, queue_size=1, latch=True)
    depth_publisher = rospy.Publisher("/debug" + used_camera["depth_topic"], Image, queue_size=1, latch=True)

    image_file = sys.argv[1]
    depth_file = sys.argv[2]

    image = cv2.imread(image_file)
    depth = np.load(depth_file)

    cv_result = np.zeros(shape=image.shape, dtype=np.uint8)
    cv2.convertScaleAbs(image, cv_result)
    image_msg = cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
    depth_msg = cv_bridge.cv2_to_imgmsg(depth, '32FC1')
    
    image_publisher.publish(image_msg)
    depth_publisher.publish(depth_msg)

    r = rospy.Rate(10)
    print("publishing image and depth")
    while not rospy.is_shutdown():

        r.sleep()


if __name__ == "__main__":
    main()
