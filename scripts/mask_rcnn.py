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

        result_msg = self.detect_objects(image_msg, depth_msg)
        res.detectedMaskRCNN = result_msg

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
        result_msg, axes_msg = self.build_result_msg(image_msg, result, np_image, np_depth)
        self.result_pub.publish(result_msg)
        self.marker_pub.publish(axes_msg)
        
        # Visualize results        
        vis_image = self.visualize(result, np_image)
        cv_result = np.zeros(shape=vis_image.shape, dtype=np.uint8)
        cv2.convertScaleAbs(vis_image, cv_result)
        image_msg = self.cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
        self.visualization_pub.publish(image_msg)

        print("Published result and image msg!")
        return result_msg

    def build_result_msg(self, msg, result, image, depth):
        result_msg = MaskRCNNMsg()
        result_msg.header = msg.header
        result_msg.header.frame_id = "base_link"
        result_msg.count = len(result['class_ids'])

        axes_msg = MarkerArray()
        
        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
            result_msg.ids.append(i)

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

            mask = result['masks'][:,:,i].astype(np.uint8)
            area, center, x_axis = self.estimate_object_attribute(mask, depth)
            if (area, center, x_axis) != (0, 0, 0):
                result_msg.areas.append(area)
                result_msg.centers.append(center)
                result_msg.axes.append(x_axis)            

            x_axis_marker = self.build_marker_msg(i, center, x_axis)
            axes_msg.markers.append(x_axis_marker)

        return result_msg, axes_msg

    def estimate_object_attribute(self, mask, depth):
        ret, thresh = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Because of one mask, the number of contours should be one.
        if len(contours) != 1:
            print("Many contours are detected")
            return (0, 0, 0)

        # Check Contour Area        
        contour = contours[0]        
        area = cv2.contourArea(contour)
        if area < 1e2 or 1e5 < area:
            print("The area of contours is too small or big")
            return (0, 0, 0)
        # result_msg.areas.append(area)

        # Calculate PCA for x-axis and y-axis of object        
        sz = len(contour)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for k in range(data_pts.shape[0]):
            data_pts[k,0] = contour[k,0,0]
            data_pts[k,1] = contour[k,0,1]
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
        cntr = (int(mean[0,0]), int(mean[0,1]))

        # Calculate center point on image coordinate by camera intrinsic parameter
        np_cntr = np.array(cntr, dtype=self.camera_matrix.dtype)
        np_cntr = np_cntr.reshape(-1, 1, 2)
        undistorted_cntr = cv2.undistortPoints(np_cntr, self.camera_matrix, self.dist_coeffs)
        undistorted_cntr = undistorted_cntr.reshape(2)

        # Calculate center point on camera coordiante
        z_camera = (depth[cntr[1], cntr[0]])/1000 + 0.02
        x_camera = undistorted_cntr[0] * z_camera
        y_camera = undistorted_cntr[1] * z_camera
        xyz_center_camera = np.array([x_camera, y_camera, z_camera], dtype=np.float32)
        print("The Center of Object (Camera Coordinate):", xyz_center_camera)

        # Calculate center point on world(robot) coordinate
        xyz_center_camera_homogeneous = np.append(xyz_center_camera, 1.0)
        xyz_center_world = np.dot(self.robot_camera, xyz_center_camera_homogeneous)
        xyz_center_world_tmp = self.marker_origin + self.tvec + np.dot(self.rvec, xyz_center_camera)
        print("The Center of Object (World Coordinate):", xyz_center_world)
        print("The Center of Object (World Coordinate):", xyz_center_world_tmp)

        center = Point(xyz_center_world[0], xyz_center_world[1], xyz_center_world[2])

        # Calculate x-axis
        np_x = np.array([cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0]], dtype=self.camera_matrix.dtype)
        np_x = np_x.reshape(-1, 1, 2)
        undistorted_x_axis = cv2.undistortPoints(np_x, self.camera_matrix, self.dist_coeffs)
        undistorted_x_axis = undistorted_x_axis.reshape(2)
        xyz_axis_camera = np.array([undistorted_x_axis[0] * z_camera, undistorted_x_axis[1] * z_camera, z_camera], dtype=np.float32)
        xyz_axis_camera_homogeneous = np.append(xyz_axis_camera, 1.0)
        xyz_axis_world = np.dot(self.robot_camera, xyz_axis_camera_homogeneous) - xyz_center_world
        # xyz_axis_world = self.marker_origin + self.tvec + np.dot(self.rvec, xyz_axis_camera) - xyz_center_world
        
        x_axis = Vector3(xyz_axis_world[0], xyz_axis_world[1], xyz_axis_world[2])

        return area, center, x_axis

    def build_marker_msg(self, id, center, x_axis):
        x_axis_marker = Marker()

        x_axis_marker.header.frame_id = "base_link"
        x_axis_marker.header.stamp = rospy.Time()
        x_axis_marker.ns = "mask_rcnn_detected_x_axis"
        x_axis_marker.type = Marker.ARROW
        x_axis_marker.action = Marker.ADD
        x_axis_marker.frame_locked = 1
        x_axis_marker.scale.x = 0.01
        x_axis_marker.scale.y = 0.02
        x_axis_marker.scale.z = 0.0
        x_axis_marker.color.a = 1.0
        x_axis_marker.color.r = 0.0
        x_axis_marker.color.g = 0.0
        x_axis_marker.color.b = 1.0
        x_axis_marker.id = id
        x_axis_marker.text = str(x_axis_marker.id)
        start_point = center
        end_point = Point(start_point.x + x_axis.x, start_point.y + x_axis.y, start_point.z + x_axis.z)
        x_axis_marker.points = [start_point, end_point]

        return x_axis_marker

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
