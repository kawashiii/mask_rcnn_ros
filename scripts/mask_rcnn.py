#!/usr/bin/env python3
import os
import sys
import copy
import time
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
MODEL = os.path.join(ROOT_DIR, "mask_rcnn_lab.h5")
CAMERA_INTRINSIC = os.path.join(ROOT_DIR, "config/realsense_intrinsic.xml")
TEST_IMG = os.path.join(ROOT_DIR, "test.png")

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
        self.set_coordinate_trasformation()

        self.class_names = lab.class_names
        # self.publish_rate = 100
        # self.class_colors = visualize.random_colors(len(CLASS_NAMES))

    def set_coordinate_trasformation(self):
        # maybe mistake, after check, delete following 3 line.
        self.tvec = np.array([0.259659, 0.221303, 0.854517], dtype=np.float32)
        self.rvec = np.array([
            [0.01391082878792294, 0.9998143798348266, 0.01333021822529676],
            [0.9998941380955942, -0.0138525840224244, -0.004451799408139552],
            [-0.00426631509639492, 0.01339073528237409, -0.9999012385051315]
        ], dtype=np.float32)
        self.marker_origin = np.array([0.24, -0.58, -0.100], dtype=np.float32)

        # Marker position and pose from camera coordinate
        tvec_from_camera = np.array([-0.221303, -0.259659, 0.854517], dtype=np.float32)
        rvec_from_camera = np.array([
            [0.01391082878792294, 0.9998143798348266, 0.01333021822529676],
            [0.9998941380955942, -0.0138525840224244, -0.004451799408139552],
            [-0.00426631509639492, 0.01339073528237409, -0.9999012385051315]
        ], dtype=np.float32)

        # Camera position and pose from marker coordinate
        tvec_from_marker = np.dot(rvec_from_camera, tvec_from_camera) * -1
        rvec_from_marker = rvec_from_camera.T

        # Homogeneous coordinate transformation from camera to marker
        marker_camera = np.array([
            np.append(rvec_from_marker[0], tvec_from_marker[0]),
            np.append(rvec_from_marker[1], tvec_from_marker[1]),
            np.append(rvec_from_marker[2], tvec_from_marker[2]),
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Homogeneous coordinate transformation from marker to robot
        robot_marker = np.array([
            [1.0, 0.0, 0.0, 0.24],
            [0.0, 1.0, 0.0, -0.58],
            [0.0, 0.0, 1.0, -0.1],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.robot_camera = np.dot(robot_marker, marker_camera)


    def run(self):
        # Define publisher
        self.result_pub = rospy.Publisher(rospy.get_name() + '/MaskRCNNMsg', MaskRCNNMsg, queue_size=1)
        self.visualization_pub = rospy.Publisher(rospy.get_name() + '/visualization', Image, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher(rospy.get_name() + '/axes', MarkerArray, queue_size=1)

        # Define service
        self.result_srv = rospy.Service(rospy.get_name() + '/MaskRCNNSrv', MaskRCNNSrv, self.wait_frame)
        print("Ready to detect objects. Please service call", rospy.get_name() +  '/MaskRCNNSrv')
        rospy.spin()

    def wait_frame(self, req):
        res = MaskRCNNSrvResponse()

        print("Waiting frame...")
        rospy.wait_for_service('/phoxi_camera/get_calibrated_frame')
        try:
            srvGetCalibratedFrame = rospy.ServiceProxy('/phoxi_camera/get_calibrated_frame', GetCalibratedFrame)
            resp = srvGetCalibratedFrame(-1, "/camera/color/image_raw")
            print("Servce call for Phoxi Success")
        except rospy.ServiceException:
            print("Service call failed")

        image_msg = rospy.wait_for_message("/phoxi_camera/rgb_texture", Image)
        depth_msg = rospy.wait_for_message("/phoxi_camera/depth_map", Image)
        print("Acquired frame!")

        self.detect_objects(image_msg, depth_msg)
        res.detectedMaskRCNN = self.result_msg

        return res

    def detect_objects(self, image_msg, depth_msg):
        np_image = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        np_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        
        # Run inference
        t1 = time.time()
        results = self.model.detect([np_image], verbose=0)
        t2 = time.time()            
        # Print detection time
        inference_time = t2 - t1
        print("Inference time: ", round(inference_time, 2), " s")

        result = results[0]
        self.result_msg, self.axes_msg = self.build_result_msg(image_msg, result, np_image, np_depth)
        self.result_pub.publish(self.result_msg)
        self.marker_pub.publish(self.axes_msg)
        
        # Visualize results        
        vis_image = self.visualize(result, np_image)
        cv_result = np.zeros(shape=vis_image.shape, dtype=np.uint8)
        cv2.convertScaleAbs(vis_image, cv_result)
        image_msg = self.cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
        self.visualization_pub.publish(image_msg)
        print("Published detected image!")

    def build_result_msg(self, msg, result, image, depth):
        result_msg = MaskRCNNMsg()
        result_msg.header = msg.header
        result_msg.header.frame_id = "base_link"
        result_msg.count = 0

        axes_msg = MarkerArray()
        
        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
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

            m = result['masks'][:,:,i].astype(np.uint8)
            ret, thresh = cv2.threshold(m, 0.5, 1.0, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for j, c in enumerate(contours):
                area = cv2.contourArea(c)
                if area < 1e2 or 1e5 < area: continue
                result_msg.areas.append(area)

                sz = len(c)
                data_pts = np.empty((sz, 2), dtype=np.float64)
                for k in range(data_pts.shape[0]):
                    data_pts[k,0] = c[k,0,0]
                    data_pts[k,1] = c[k,0,1]
                mean = np.empty((0))
                mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
                cntr = (int(mean[0,0]), int(mean[0,1]))

                np_cntr = np.array(cntr, dtype=self.camera_matrix.dtype)
                np_cntr = np_cntr.reshape(-1, 1, 2)
                undistorted_cntr = cv2.undistortPoints(np_cntr, self.camera_matrix, self.dist_coeffs)
                undistorted_cntr = undistorted_cntr.reshape(2)

                z_camera = (depth[cntr[1], cntr[0]])/1000 + 0.02
                x_camera = undistorted_cntr[0] * z_camera
                y_camera = undistorted_cntr[1] * z_camera
                xyz_center_camera = np.array([x_camera, y_camera, z_camera], dtype=np.float32)
                print("The Center of Object (Camera Coordinate):", xyz_center_camera)

                xyz_center_world = self.marker_origin + self.tvec + np.dot(self.rvec, xyz_center_camera)
                xyz_center_camera_tmp = np.array([x_camera, y_camera, z_camera, 1.0], dtype=np.float32)
                tmp = np.dot(self.robot_camera, xyz_center_camera_tmp)
                print("The Center of Object (World Coordinate):", xyz_center_world)
                print("The Center of Object (World Coordinate) tmp:", tmp)

                center = Point()
                center.x = xyz_center_world[0]
                center.y = xyz_center_world[1]
                center.z = xyz_center_world[2]                
                result_msg.centers.append(center)

                np_x = np.array([cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0]], dtype=self.camera_matrix.dtype)
                np_x = np_x.reshape(-1, 1, 2)
                undistorted_x_axis = cv2.undistortPoints(np_x, self.camera_matrix, self.dist_coeffs)
                undistorted_x_axis = undistorted_x_axis.reshape(2)
                xyz_axis_camera = np.array([undistorted_x_axis[0] * z_camera, undistorted_x_axis[1] * z_camera, z_camera], dtype=np.float32)
                xyz_axis_world = self.marker_origin + self.tvec + np.dot(self.rvec, xyz_axis_camera) - xyz_center_world
                
                axe = Vector3()
                axe.x = xyz_axis_world[0]
                axe.y = xyz_axis_world[1]
                axe.z = xyz_axis_world[2]
                result_msg.axes.append(axe)

                axis = Marker()

                axis.header.frame_id = "base_link"
                axis.header.stamp = rospy.Time()
                axis.ns = "mask_rcnn_detected_axis_pp"
                axis.type = Marker.ARROW
                axis.action = Marker.ADD
                axis.frame_locked = 1
                axis.scale.x = 0.01
                axis.scale.y = 0.02
                axis.scale.z = 0.0
                axis.color.a = 1.0
                axis.color.r = 0.0
                axis.color.g = 0.0
                axis.color.b = 1.0
                axis.id = i
                axis.text = str(axis.id)
                start_point = Point()
                start_point.x = xyz_center_world[0]
                start_point.y = xyz_center_world[1]
                start_point.z = xyz_center_world[2]
                end_point = Point()
                end_point.x = xyz_center_world[0] + xyz_axis_world[0]
                end_point.y = xyz_center_world[1] + xyz_axis_world[1]
                end_point.z = xyz_center_world[2] + xyz_axis_world[2]
                axis.points = [start_point, end_point]
                axes_msg.markers.append(axis)

                # np_y = np.array([eigenvectors[1,0] * eigenvalues[1,0], eigenvectors[1,1] * eigenvalues[1,0]], dtype=self.camera_matrix.dtype)
                # np_y = np_y.reshape(-1, 1, 2)
                # undistorted_y_axis = cv2.undistortPoints(np_y, self.camera_matrix, self.dist_coeffs)
                # undistorted_y_axis = undistorted_y_axis.reshape(2)
                # x_axis = Vector3()
                # x_axis.x = undistorted_x_axis[0]
                # x_axis.y = undistorted_x_axis[1]
                # x_axis.z = center.z
                # y_axis = Vector3()
                # y_axis.x = undistorted_y_axis[0]
                # y_axis_y = undistorted_y_axis[1]
                # y_axis_z = center.z
                # result_msg.x_axis.append(x_axis)
                # result_msg.y_axis.append(y_axis)
            
            result_msg.ids.append(i)
            result_msg.count += 1
        return result_msg, axes_msg

    def estimate_object_pose(self, mask):
        print("test")

    def build_marker_msg(self):
        print("test")

    def visualize(self, result, image):
        rois = result['rois']
        masks = result['masks']
        class_ids = result['class_ids']
        scores = result['scores']
        result = np.copy(image)

        for i in range(masks.shape[-1]):
            y1, x1, y2, x2 = rois[i]
            class_id = class_ids[i]
            score = scores[i]
            label = self.class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
            cv2.putText(result, caption, (x1, y1 + 8), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,255,0), 1, 8)
 
            m = masks[:,:,i].astype(np.uint8)
            ret, thresh = cv2.threshold(m, 0.5, 1.0, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                if area < 1e2 or 1e5 < area:
                    continue
                cv2.drawContours(result, contours, i, (0, 0, 255), 2)
                self.getOrientation(c, result)

        return result

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

    def getOrientation(self, pts, img):    
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i,0] = pts[i,0,0]
            data_pts[i,1] = pts[i,0,1]
        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
        # Store the center of the object
        cntr = (int(mean[0,0]), int(mean[0,1]))

        cv2.circle(img, cntr, 3, (255, 0, 255), 2)
        p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
        self.drawAxis(img, cntr, p1, (0, 0, 255), 1)
        self.drawAxis(img, cntr, p2, (0, 255, 0), 5)
        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
        
        return angle

def main():
    rospy.init_node('mask_rcnn')

    node = MaskRCNNNode()
    node.run()

if __name__ == '__main__':
    main()
