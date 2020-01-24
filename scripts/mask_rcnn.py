#!/usr/bin/env python3
import os
import sys
import copy
import time
import math
import threading
import numpy as np

import rospy
import roslib.packages
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import UInt8MultiArray
from phoxi_camera.srv import *
from mask_rcnn_ros.msg import MaskRCNNMsg
from mask_rcnn_ros.srv import MaskRCNNSrv, MaskRCNNSrvResponse

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import lab
import cv2
from math import atan2, cos, sin, sqrt, pi
from cv_bridge import CvBridge

from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

ROOT_DIR = os.path.abspath(roslib.packages.get_pkg_dir('mask_rcnn_ros'))
MODEL = os.path.join(ROOT_DIR, "mask_rcnn_lab_2019_11_07.h5")
# CAMERA_INTRINSIC = os.path.join(ROOT_DIR, "config/realsense_intrinsic_1.xml")
# CAMERA_INTRINSIC = os.path.join(ROOT_DIR, "config/realsense_intrinsic_2.xml")
CAMERA_INTRINSIC = os.path.join(ROOT_DIR, "config/basler_intrinsic.xml")

#REGION_X_OFFSET = 700
#REGION_Y_OFFSET = 210
#REGION_WIDTH    = 700
#REGION_HEIGHT   = 500

REGION_X_OFFSET = 0
REGION_Y_OFFSET = 0
REGION_WIDTH    = 2048
REGION_HEIGHT   = 1536

# if the depth of (x, y) is 0, it is approximate depth value around specific pixel
DEPTH_APPROXIMATE_RANGE = 10


class InferenceConfig(lab.LabConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRCNNNode(object):
    def __init__(self):
        self.cv_bridge = CvBridge()

        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)
        self.model.load_weights(MODEL, by_name=True)
        self.model.keras_model._make_predict_function()

        # Set calibration matrix
        fs = cv2.FileStorage(CAMERA_INTRINSIC, cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode("camera_matrix").mat()
        self.dist_coeffs = fs.getNode("distortion_coefficients").mat()

        self.class_names = lab.class_names
    
    def run(self):
        # Define publisher
        self.result_pub = rospy.Publisher(rospy.get_name() + '/MaskRCNNMsg', MaskRCNNMsg, queue_size=1)
        self.visualization_pub = rospy.Publisher(rospy.get_name() + '/visualization', Image, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher(rospy.get_name() + '/axes', MarkerArray, queue_size=1)

        # Define service
        self.result_srv = rospy.Service(rospy.get_name() + '/MaskRCNNSrv', MaskRCNNSrv, self.wait_frame)
        rospy.loginfo("Ready to be called service")
        rospy.spin()

    def wait_frame(self, req):
        res = MaskRCNNSrvResponse()

        rospy.loginfo("Waiting frame ...")
        timeout = 10
        image_msg = rospy.wait_for_message("/phoxi_camera/external_camera_texture", Image, timeout)
        # image_msg = rospy.wait_for_message("/camera/color/image_raw", Image, timeout)
        depth_msg = rospy.wait_for_message("/phoxi_camera/aligned_depth_map", Image, timeout)
        # depth_msg = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image, timeout)
        rospy.loginfo("Acquired frame")

        start_build_msg = time.time() 
        result_msg = self.detect_objects(image_msg, depth_msg)
        res.detectedMaskRCNN = result_msg
        end_build_msg = time.time() 
        build_msg_time = end_build_msg - start_build_msg
        rospy.loginfo("%s[s] (Total time)", round(build_msg_time, 3))

        return res

    def detect_objects(self, image_msg, depth_msg):
        np_image = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        np_image = np_image[REGION_Y_OFFSET:REGION_Y_OFFSET + REGION_HEIGHT, REGION_X_OFFSET:REGION_X_OFFSET + REGION_WIDTH]
        np_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '32FC1')

        # MaskRCNN input data is RGB, not BGR
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        start_detection = time.time() 
        rospy.loginfo("Detecting ...")
        results = self.model.detect([np_image], verbose=0)
        end_detection = time.time()
        detection_time = end_detection - start_detection
        rospy.loginfo("%s[s] (Detection time)", round(detection_time, 3))

        # Back to BGR for visualization and publish
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        result = results[0]
        result_msg = self.build_result_msg(image_msg.header, result, np_image, np_depth)

        return result_msg

    def build_result_msg(self, msg_header, result, image, depth):
        rospy.loginfo("Building msg ...")
        result_msg = MaskRCNNMsg()
        result_msg.header = msg_header
        # result_msg.header.frame_id = "realsense_rgb_sensor_calibrated"
        result_msg.header.frame_id = "basler_ace_rgb_sensor_calibrated"
        result_msg.count = 0

        vis_image = np.copy(image)
        axes_msg = MarkerArray()
        delete_marker = self.delete_all_markers()
        axes_msg.markers.append(delete_marker)
        
        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
            rospy.loginfo("'%s' is detected", self.class_names[result['class_ids'][i]])
            mask = result['masks'][:,:,i].astype(np.uint8)
            area, center, x_axis, y_axis = self.estimate_object_attribute(mask, depth, vis_image)
            if (area, center, x_axis, y_axis) == (0, 0, 0, 0):
                continue

            z_axis = Vector3(
                x_axis.y * y_axis.z - x_axis.z * y_axis.y, 
                x_axis.z * y_axis.x - x_axis.x * y_axis.z,
                x_axis.x * y_axis.y - x_axis.y * y_axis.x
            )
            if z_axis.z > 0:
                z_axis = Vector3(z_axis.x * -1, z_axis.y * -1, z_axis.z * -1)
                x_axis = Vector3(x_axis.x * -1, x_axis.y * -1, x_axis.z * -1)

            result_msg.areas.append(area)
            result_msg.centers.append(center)
            result_msg.axes.append(x_axis)
            result_msg.normals.append(z_axis)            

            x_axis_marker = self.build_marker_msg(i, center, x_axis, 1.0, 0.0, 0.0, "x_axis")
            y_axis_marker = self.build_marker_msg(i, center, y_axis, 0.0, 1.0, 0.0, "y_axis")
            z_axis_marker = self.build_marker_msg(i, center, z_axis, 0.0, 0.0, 1.0, "z_axis")
            axes_msg.markers.append(x_axis_marker)
            axes_msg.markers.append(y_axis_marker)
            axes_msg.markers.append(z_axis_marker)

            # result_msg.ids.append(i)
            result_msg.ids.append(result_msg.count)
            result_msg.count += 1

            box = RegionOfInterest()
            box.x_offset = np.asscalar(x1)
            box.y_offset = np.asscalar(y1)
            box.height = np.asscalar(y2 - y1)
            box.width = np.asscalar(x2 - x1)
            result_msg.boxes.append(box)

            class_id = result['class_ids'][i]
            result_msg.class_ids.append(class_id)

            class_name = self.class_names[class_id]
            result_msg.class_names.append(class_name)

            score = result['scores'][i]
            result_msg.scores.append(score)

            caption = "{} {:.3f}".format(class_name, score) if score else class_name
            cv2.putText(vis_image, caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1, cv2.LINE_AA)

        self.result_pub.publish(result_msg)
        self.marker_pub.publish(axes_msg)

        cv_result = np.zeros(shape=vis_image.shape, dtype=np.uint8)
        cv2.convertScaleAbs(vis_image, cv_result)
        image_msg = self.cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
        self.visualization_pub.publish(image_msg)

        rospy.loginfo("Published msg completely")

        return result_msg

    def estimate_object_attribute(self, mask, depth, vis_image):
        ret, thresh = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Because of one mask, the number of contours should be one.
        if len(contours) != 1:
            rospy.logwarn("Skip this object.(Inferenced mask is not clearly.)")
            return (0, 0, 0, 0)

        # Check Contour Area        
        contour = contours[0]        
        area = cv2.contourArea(contour)
        if area < 1e2 or 1e5 < area:
            rospy.logwarn("Skip this object.(The area of contours is too small or big.)")
            return (0, 0, 0, 0)

        # Calculate PCA for x-axis and y-axis of object        
        sz = len(contour)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for k in range(data_pts.shape[0]):
            data_pts[k,0] = contour[k,0,0]
            data_pts[k,1] = contour[k,0,1]
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
        obj_center_pixel = (int(mean[0,0]) + REGION_X_OFFSET, int(mean[0,1]) + REGION_Y_OFFSET)
        unit_x_axis_pixel = np.array([eigenvectors[0,0] * eigenvalues[0,0], eigenvectors[0,1] * eigenvalues[0,0]]) / math.sqrt((eigenvectors[0,0] * eigenvalues[0,0])**2 + (eigenvectors[0,1] * eigenvalues[0,0])**2)
        unit_y_axis_pixel = np.array([eigenvectors[1,0] * eigenvalues[1,0], eigenvectors[1,1] * eigenvalues[1,0]]) / math.sqrt((eigenvectors[1,0] * eigenvalues[1,0])**2 + (eigenvectors[1,1] * eigenvalues[1,0])**2)

        # Calculate center point on image coordinate by camera intrinsic parameter
        np_obj_center_pixel = np.array(obj_center_pixel, dtype=self.camera_matrix.dtype)
        undistorted_center = cv2.undistortPoints(np_obj_center_pixel, self.camera_matrix, self.dist_coeffs)
        undistorted_center = undistorted_center.reshape(2)

        # Check center point of Depth Value
        obj_center_depth, x_axis_depth, y_axis_depth = self.get_depth(depth, obj_center_pixel, unit_x_axis_pixel, unit_y_axis_pixel)
        if obj_center_depth == 0.0 or x_axis_depth == 0.0 or y_axis_depth == 0.0:
            rospy.logwarn("Skip this object.(Depth value around center point is all 0.)")
            return (0, 0, 0, 0)

        print("obj_center_depth: ", obj_center_depth)
        print("x_axis_depth: ", x_axis_depth)

        # obj_center_depth += 0.06

        # Calculate center point on camera coordiante
        obj_center_z = obj_center_depth
        obj_center_x = undistorted_center[0] * obj_center_z
        obj_center_y = undistorted_center[1] * obj_center_z
        obj_center_camera = Point(obj_center_x, obj_center_y, obj_center_z)

        # Calculate x-axis
        np_x_axis_pixel = np.array([obj_center_pixel[0] + unit_x_axis_pixel[0]*20, obj_center_pixel[1] + unit_x_axis_pixel[1]*20], dtype=self.camera_matrix.dtype)
        undistorted_x_axis = cv2.undistortPoints(np_x_axis_pixel, self.camera_matrix, self.dist_coeffs)
        undistorted_x_axis = undistorted_x_axis.reshape(2)
        x_axis_z = x_axis_depth - obj_center_z
        x_axis_x = undistorted_x_axis[0] * x_axis_depth - obj_center_x 
        x_axis_y = undistorted_x_axis[1] * x_axis_depth - obj_center_y
        magnitude_x_axis = sqrt(x_axis_x**2 + x_axis_y**2 + x_axis_z**2)
        x_axis_camera = Vector3(x_axis_x/magnitude_x_axis, x_axis_y/magnitude_x_axis, x_axis_z/magnitude_x_axis)

        np_y_axis_pixel = np.array([obj_center_pixel[0] - unit_y_axis_pixel[0]*20, obj_center_pixel[1] - unit_y_axis_pixel[1]*20], dtype=self.camera_matrix.dtype)
        undistorted_y_axis = cv2.undistortPoints(np_y_axis_pixel, self.camera_matrix, self.dist_coeffs)
        undistorted_y_axis = undistorted_y_axis.reshape(2)
        y_axis_z = y_axis_depth - obj_center_z
        y_axis_x = undistorted_y_axis[0] * y_axis_depth - obj_center_x 
        y_axis_y = undistorted_y_axis[1] * y_axis_depth - obj_center_y
        magnitude_y_axis = sqrt(y_axis_x**2 + y_axis_y**2 + y_axis_z**2)
        y_axis_camera = Vector3(y_axis_x/magnitude_x_axis, y_axis_y/magnitude_x_axis, y_axis_z/magnitude_x_axis)

        # visualize mask, axis
        cv2.drawContours(vis_image, contours, 0, (255, 255, 0), 2)
        cntr = (int(mean[0,0]), int(mean[0,1]))
        cv2.circle(vis_image, cntr, 3, (255, 0, 255), 2)
        p1 = (cntr[0] + unit_x_axis_pixel[0]*20, cntr[1] + unit_x_axis_pixel[1]*20)
        p2 = (cntr[0] - unit_y_axis_pixel[0]*20, cntr[1] - unit_y_axis_pixel[1]*20)
        self.drawAxis(vis_image, cntr, p1, (0, 0, 255), 1)
        self.drawAxis(vis_image, cntr, p2, (0, 255, 0), 1)

        return area, obj_center_camera, x_axis_camera, y_axis_camera

    def get_depth(self, depth, center, x_axis, y_axis):
        center_x = center[0]
        center_y = center[1]
        center_depth = depth[center_y, center_x] / 1000
        x_axis_depth = depth[center_y + int(x_axis[1]*20), center_x + int(x_axis[0]*20)] / 1000
        y_axis_depth = depth[center_y + int(y_axis[1]*20), center_x + int(y_axis[0]*20)] / 1000       
        if not center_depth:
            # Consider the depth value of 10 * 10 pixels around (x, y)
            height, width = depth.shape
            rect_x_min, rect_x_max = center_x - int(DEPTH_APPROXIMATE_RANGE / 2), center_x + int(DEPTH_APPROXIMATE_RANGE / 2)
            rect_y_min, rect_y_max = center_y - int(DEPTH_APPROXIMATE_RANGE / 2), center_y + int(DEPTH_APPROXIMATE_RANGE / 2)
            if rect_x_min >= 0 and rect_x_max <= width and rect_y_min >= 0 and rect_y_max <= height:
                depth_array = []
                for h in range(rect_y_max - rect_y_min):
                    for w in range(rect_x_max - rect_x_min):
                        if depth[rect_y_min + h, rect_x_min + w] == 0:
                            continue
                        depth_array.append(depth[rect_y_min + h, rect_x_min + w])
                if len(depth_array) == 0:
                    center_depth = 0
                rospy.loginfo("This object's depth value is averaged around the center point")
                center_depth = sum(depth_array) / len(depth_array) / 1000

        return center_depth, x_axis_depth, y_axis_depth

    def build_marker_msg(self, id, center, axis, r, g, b, description):
        axis_marker = Marker()

        # axis_marker.header.frame_id = "realsense_rgb_sensor_calibrated"
        axis_marker.header.frame_id = "basler_ace_rgb_sensor_calibrated"
        axis_marker.header.stamp = rospy.Time()
        axis_marker.ns = "mask_rcnn_detected_" + description
        axis_marker.type = Marker.ARROW
        axis_marker.action = Marker.ADD
        axis_marker.frame_locked = 1
        axis_marker.scale.x = 0.01
        axis_marker.scale.y = 0.01
        axis_marker.scale.z = 0.01
        axis_marker.color.a = 1.0
        axis_marker.color.r = r
        axis_marker.color.g = g
        axis_marker.color.b = b
        axis_marker.id = id
        axis_marker.text = str(axis_marker.id)
        start_point = center
        # end_point = x_axis
        end_point = Point(center.x + axis.x * 0.05, center.y + axis.y * 0.05, center.z + axis.z * 0.05)
        axis_marker.points = [start_point, end_point]

        return axis_marker

    def delete_all_markers(self):
        delete_marker = Marker()

        # delete_marker.header.frame_id = "realsense_rgb_sensor_calibrated"
        delete_marker.header.frame_id = "basler_ace_rgb_sensor_calibrated"
        delete_marker.header.stamp = rospy.Time()
        delete_marker.type = Marker.ARROW
        delete_marker.action = Marker.DELETEALL

        return delete_marker

    def drawAxis(self, img, p_, q_, colour, scale):
        p = list(p_)
        q = list(q_)
        
        angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
        hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
        # Here we lengthen the arrow by a factor of scale
        q[0] = p[0] - scale * hypotenuse * cos(angle)
        q[1] = p[1] - scale * hypotenuse * sin(angle)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
        # create the arrow hooks
        p[0] = q[0] + 9 * cos(angle + pi / 4)
        p[1] = q[1] + 9 * sin(angle + pi / 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
        p[0] = q[0] + 9 * cos(angle - pi / 4)
        p[1] = q[1] + 9 * sin(angle - pi / 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

def main():
    rospy.init_node('mask_rcnn')

    node = MaskRCNNNode()
    node.run()

if __name__ == '__main__':
    main()
