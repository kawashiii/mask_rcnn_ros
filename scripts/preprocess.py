#!/usr/bin/env python
import os
import sys
import copy
import time
import math
import numpy as np

import rospy
import roslib.packages
from sensor_msgs.msg import Image,CameraInfo
from phoxi_camera.srv import *
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
import tf
import tf_conversions

cv_bridge = CvBridge()
listener = tf.TransformListener()
frame_id = ""

x_min = -0.298
x_max = 0.298
y_min = -0.200
y_max = 0.200
z_min = 0.005
z_max = 0.300

def in_container(point):
    point_camera = PointStamped()
    point_camera.header.frame_id = frame_id
    point_camera.header.stamp = rospy.Time.now()
    point_camera.point = point

    point_container = listener.tranformPoint("container", point_camera)
    return x_min < point_container.point.x < x_max and y_min < point_container.point.y < y_max and z_min < point_container.point.z < z_max

def main():
    rospy.init_node('image_saver')


    aligned_depth_topic_name = "/phoxi_camera/aligned_depth_map"
    image_topic_name = "/pylon_camera_node/image_rect"
    camera_info_topic_name = "/pylon_camera_node/camera_info"

    save_folder = "/home/dfk2/data/MaskRCNN/choice"
    timeout = 10

    camera_info = rospy.wait_for_message(camera_info_topic_name, CameraInfo, timeout)
    cameraMatrix = np.array(camera_info.K).reshape(3, 3)
    distCoeffs = np.array(camera_info.D)
    fx = cameraMatrix[0,0]
    fy = cameraMatrix[1,1]
    cx = cameraMatrix[0,2]
    cy = cameraMatrix[1,2]

    rospy.wait_for_service("phoxi_camera/get_frame");
    try:
        srvGetFrame = rospy.ServiceProxy("/phoxi_camera/get_frame", GetFrame)
        resp = srvGetFrame(-1)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

    aligned_depth_msg = rospy.wait_for_message(aligned_depth_topic_name, Image, timeout)
    image_msg = rospy.wait_for_message(image_topic_name, Image, timeout)
    cv_aligned_depth = cv_bridge.imgmsg_to_cv2(aligned_depth_msg, "32FC1")
    cv_image = cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")

    rectify_map = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, np.eye(3), cameraMatrix, (cv_aligned_depth.shape[1], cv_aligned_depth.shape[0]), cv2.CV_32FC1)
    rectified_depth = cv2.remap(cv_aligned_depth, rectify_map[0], rectify_map[1], cv2.INTER_NEAREST)

    height, width = rectified_depth.shape[:2]

    for v in range(height):
        for u in range(width):
            z = rectified_depth[v,u]/1000
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            point = Point(x, y, z)

            if not in_container(point):
                cv_image[v,u] = np.zeros(3)

    cv2.imshow(cv_image)
    cv2.show()

if __name__ == "__main__":
    main()
