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
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import UInt8MultiArray
from std_msgs.msg import Int32
from mask_rcnn_ros.msg import MaskRCNNMsg
from mask_rcnn_ros.srv import MaskRCNNSrv, MaskRCNNSrvResponse, GetNormal, GetNormalResponse
from icp_registration.msg import Container

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import lab
import cv2
from math import atan2, cos, sin, sqrt, pi
from cv_bridge import CvBridge

from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

ROOT_DIR = os.path.abspath(roslib.packages.get_pkg_dir('mask_rcnn_ros'))
MODEL = os.path.join(ROOT_DIR, "mask_rcnn_lab_caloriemate.h5")
CAMERA_INTRINSIC = os.path.join(ROOT_DIR, "config/basler_intrinsic.xml")
FRAME_ID = "basler_ace_rgb_sensor_calibrated"

REGION_X_OFFSET = 0
REGION_Y_OFFSET = 0
REGION_WIDTH    = 2048
REGION_HEIGHT   = 1536

# if the depth of (x, y) is 0, it is approximate depth value around specific pixel
DEPTH_APPROXIMATE_RANGE = 10

# The edge position of container 
CONTAINER_EDGE_POSITION = np.array([
    [0.300, 0.203, 0.000],
    [0.300, -0.203, 0.000],
    [-0.300, 0.203, 0.000],
    [-0.300, -0.203, 0.000],
    [0.300, 0.203, 0.315],
    [0.300, -0.203, 0.315],
    [-0.300, 0.203, 0.315],
    [-0.300, -0.203, 0.315]
], dtype=float)
                               

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
        self.result_pub = rospy.Publisher(rospy.get_name() + '/MaskRCNNMsg', MaskRCNNMsg, queue_size=1, latch=True)
        self.visualization_pub = rospy.Publisher(rospy.get_name() + '/visualization', Image, queue_size=1, latch=True)
        self.vis_depth_pub = rospy.Publisher(rospy.get_name() + '/depth', Image, queue_size=1, latch=True)
        self.vis_picking_object_pub = rospy.Publisher(rospy.get_name() + '/vis_picking_object', Image, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher(rospy.get_name() + '/axes', MarkerArray, queue_size=1, latch=True)
        self.trigger_pub = rospy.Publisher(rospy.get_name() + '/trigger_id', Int32, queue_size=1, latch=True)
        self.vis_container_edge = rospy.Publisher(rospy.get_name() + "/vis_container_edge", Image, queue_size=1, latch=True)

        # Define subscriber
        # self.result_sub = rospy.Subscriber("/phoxi_camera/external_camera_texture", Image, self.callback_get_image)
        self.container_sub = rospy.Subscriber("icp_registration/container_position", Container, self.callback_get_container_edge)

        # Define service
        self.result_srv = rospy.Service(rospy.get_name() + '/MaskRCNNSrv', MaskRCNNSrv, self.wait_frame)

        rospy.loginfo("Ready to be called service")
        rospy.spin()

    def callback_get_image(self, image_msg):
        depth_msg = rospy.wait_for_message("/phoxi_camera/aligned_depth_map", Image, 10)
        trigger_id = rospy.wait_for_message("/phoxi_camera/trigger_id", Int32, 10)
        rospy.loginfo("TriggerID: %s", trigger_id.data)       
 
        start_build_msg = time.time() 
        result_msg = self.detect_objects(image_msg, depth_msg)
        self.trigger_pub.publish(trigger_id.data)
        end_build_msg = time.time() 
        build_msg_time = end_build_msg - start_build_msg
        rospy.loginfo("%s[s] (Total time)", round(build_msg_time, 3))

    def callback_get_container_edge(self, msg):
        rospy.loginfo("CallBack container position")
        pts = []
        for p in msg.edges:
            edge = np.array([p.point.x, p.point.y, p.point.z], dtype=float)
            projected_point = cv2.projectPoints(edge, (0,0,0), (0,0,0), self.camera_matrix, self.dist_coeffs)
            projected_point = projected_point[0].reshape(-1)
            pts.append((int(projected_point[0]), int(projected_point[1])))

        image_msg = rospy.wait_for_message("/pylon_camera_node/image_raw", Image, 10)
        img = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')

        for pt in pts:
            cv2.circle(img, pt, 8, (0, 0, 255), -1)

        cv2.line(img, pts[0], pts[1], (0, 0, 255), 5)
        cv2.line(img, pts[1], pts[2], (0, 0, 255), 5)
        cv2.line(img, pts[2], pts[3], (0, 0, 255), 5)
        cv2.line(img, pts[3], pts[0], (0, 0, 255), 5)
        cv2.line(img, pts[4], pts[5], (0, 0, 255), 5)
        cv2.line(img, pts[5], pts[6], (0, 0, 255), 5)
        cv2.line(img, pts[6], pts[7], (0, 0, 255), 5)
        cv2.line(img, pts[7], pts[4], (0, 0, 255), 5)
        cv2.line(img, pts[0], pts[4], (0, 0, 255), 5)
        cv2.line(img, pts[1], pts[5], (0, 0, 255), 5)
        cv2.line(img, pts[2], pts[6], (0, 0, 255), 5)
        cv2.line(img, pts[3], pts[7], (0, 0, 255), 5)

        image_msg = self.cv_bridge.cv2_to_imgmsg(img, 'bgr8')
        self.vis_container_edge.publish(image_msg)

    def wait_frame(self, req):
        res = MaskRCNNSrvResponse()

        rospy.loginfo("Waiting frame ...")
        timeout = 10
        image_msg = rospy.wait_for_message("/pylon_camera_node/image_raw", Image, timeout)
        depth_msg = rospy.wait_for_message("/phoxi_camera/aligned_depth_map", Image, timeout)
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

        self.image = np.copy(np_image)

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
        result_msg.header.frame_id = FRAME_ID
        result_msg.count = 0

        vis_image = np.copy(image)
        vis_depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        axes_msg = MarkerArray()
        delete_marker = self.delete_all_markers()
        axes_msg.markers.append(delete_marker)
        
        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
            rospy.loginfo("'%s' is detected", self.class_names[result['class_ids'][i]])
            mask = result['masks'][:,:,i].astype(np.uint8)
            area, center, x_axis, y_axis = self.estimate_object_attribute(mask, depth, vis_image, vis_depth)
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

            z_axis_stamped = Vector3Stamped()
            z_axis_stamped.header.frame_id = FRAME_ID
            z_axis_stamped.header.stamp = rospy.Time.now()
            z_axis_stamped.vector = z_axis

            center_stamped = PointStamped()
            center_stamped.header.frame_id = FRAME_ID
            center_stamped.header.stamp = rospy.Time.now()
            center_stamped.point = center

            x_axis_stamped = Vector3Stamped()
            x_axis_stamped.header.frame_id = FRAME_ID
            x_axis_stamped.header.stamp = rospy.Time.now()
            x_axis_stamped.vector = x_axis

            result_msg.areas.append(area)
            result_msg.centers.append(center_stamped)
            result_msg.axes.append(x_axis_stamped)
            result_msg.normals.append(z_axis_stamped)            

            x_axis_marker = self.build_marker_msg(FRAME_ID, Marker.ARROW, result_msg.count, center, x_axis, 1.0, 0.0, 0.0, "x_axis")
            y_axis_marker = self.build_marker_msg(FRAME_ID, Marker.ARROW, result_msg.count, center, y_axis, 0.0, 1.0, 0.0, "y_axis")
            z_axis_marker = self.build_marker_msg(FRAME_ID, Marker.ARROW, result_msg.count, center, z_axis, 0.0, 0.0, 1.0, "z_axis")
           
            text_marker = self.build_marker_msg(FRAME_ID, Marker.TEXT_VIEW_FACING, result_msg.count, center, z_axis, 1.0, 1.0, 1.0, "id_text")

            #axes_msg.markers.append(x_axis_marker)
            #axes_msg.markers.append(y_axis_marker)
            #axes_msg.markers.append(z_axis_marker)
            axes_msg.markers.append(text_marker)
            
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
            cv2.putText(vis_image, caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 2, cv2.LINE_AA)

        # service call get_normal in mask_rcnn_utils.py
        try:
            get_normal = rospy.ServiceProxy("/mask_rcnn_utils/get_normal", GetNormal)
            res = get_normal(result_msg.centers)
        except rospy.ServiceException as e:
            rospy.logerr("Service calll failed: ",e)

        result_msg.centers = []
        result_msg.normals = []
        for i, (center, normal) in enumerate(zip(res.centers, res.normals)):
            result_msg.normals.append(normal)
            result_msg.centers.append(center)
            normal_marker = self.build_marker_msg(center.header.frame_id, Marker.ARROW, i, center.point, normal.vector, 0.5, 0.0, 0.5, "normal")
            axes_msg.markers.append(normal_marker)
        

        self.result_pub.publish(result_msg)
        self.marker_pub.publish(axes_msg)

        cv_result = np.zeros(shape=vis_image.shape, dtype=np.uint8)
        cv2.convertScaleAbs(vis_image, cv_result)
        image_msg = self.cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
        # self.visualization_pub.publish(image_msg)
        self.vis_picking_object_pub.publish(image_msg)

        vis_image = self.visualize(result, image)
        cv_result = np.zeros(shape=image.shape, dtype=np.uint8)
        cv2.convertScaleAbs(vis_image, cv_result)
        image_msg = self.cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
        self.visualization_pub.publish(image_msg)

        cv_result = np.zeros(shape=vis_depth.shape, dtype=np.uint8)
        cv2.convertScaleAbs(vis_depth, cv_result)
        depth_msg = self.cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
        self.vis_depth_pub.publish(depth_msg)

        rospy.loginfo("Published msg completely")

        return result_msg

    def estimate_object_attribute(self, mask, depth, vis_image, vis_depth):
        ret, thresh = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Because of one mask, the number of contours should be one.
        if len(contours) != 1:
            rospy.logwarn("Skip this object.(Inferenced mask is not clearly.)")
            return (0, 0, 0, 0)

        # Check Contour Area        
        contour = contours[0]        
        area = cv2.contourArea(contour)
        if area < 1e2 or 1e7 < area:
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
        undistorted_x = int(self.camera_matrix[0][0]*undistorted_center[0] + self.camera_matrix[0][2])
        undistorted_y = int(self.camera_matrix[1][1]*undistorted_center[1] + self.camera_matrix[1][2])
        

        # Check center point of Depth Value
        obj_center_depth, x_axis_depth, y_axis_depth = self.get_depth(depth, obj_center_pixel, unit_x_axis_pixel, unit_y_axis_pixel)
        #obj_center_depth, x_axis_depth, y_axis_depth = self.get_depth(depth, (undistorted_x, undistorted_y), unit_x_axis_pixel, unit_y_axis_pixel)
        if obj_center_depth == 0.0 or x_axis_depth == 0.0 or y_axis_depth == 0.0:
            rospy.logwarn("Skip this object.(Depth value around center point is all 0.)")
            return (0, 0, 0, 0)

        # print("obj_center_depth: ", obj_center_depth)
        # print("x_axis_depth: ", x_axis_depth)

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
        cv2.drawContours(vis_image, contours, 0, (255, 255, 0), 6)
        cntr = (int(mean[0,0]), int(mean[0,1]))
        cv2.circle(vis_image, cntr, 10, (255, 0, 255), -1)
        p1 = (cntr[0] + unit_x_axis_pixel[0]*20, cntr[1] + unit_x_axis_pixel[1]*20)
        p2 = (cntr[0] - unit_y_axis_pixel[0]*20, cntr[1] - unit_y_axis_pixel[1]*20)
        self.drawAxis(vis_image, cntr, p1, (0, 0, 255), 3)
        self.drawAxis(vis_image, cntr, p2, (0, 255, 0), 3)

        cv2.circle(vis_depth, obj_center_pixel, 10, (255, 0, 0), 3)
        cv2.circle(vis_depth, (undistorted_x, undistorted_y), 10, (0, 0, 255), 3)

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
                else: 
                    center_depth = sum(depth_array) / len(depth_array) / 1000
                    rospy.loginfo("This object's depth value is averaged around the center point")

        return center_depth, x_axis_depth, y_axis_depth

    def build_marker_msg(self, frame_id, marker_type, marker_id, center, axis, r, g, b, description):
        axis_marker = Marker()

        axis_marker.header.frame_id = frame_id
        axis_marker.header.stamp = rospy.Time()
        axis_marker.ns = "mask_rcnn_detected_" + description
        axis_marker.type = marker_type
        axis_marker.action = Marker.ADD
        axis_marker.frame_locked = 1
        axis_marker.scale.x = 0.01
        axis_marker.scale.y = 0.01
        axis_marker.scale.z = 0.01
        if marker_type == Marker.TEXT_VIEW_FACING:
            axis_marker.scale.x = 0.04
            axis_marker.scale.y = 0.04
            axis_marker.scale.z = 0.04
        axis_marker.color.a = 1.0
        axis_marker.color.r = r
        axis_marker.color.g = g
        axis_marker.color.b = b
        axis_marker.id = marker_id
        axis_marker.text = str(axis_marker.id)

        if marker_type == Marker.TEXT_VIEW_FACING:
            axis_marker.pose.position.x = center.x + axis.x * 0.05
            axis_marker.pose.position.y = center.y + axis.y * 0.05
            axis_marker.pose.position.z = center.z - 0.01 + axis.z * 0.05
            return axis_marker

        start_point = center
        # end_point = x_axis
        end_point = Point(center.x + axis.x * 0.05, center.y + axis.y * 0.05, center.z + axis.z * 0.05)
        axis_marker.points = [start_point, end_point]

        return axis_marker

    def delete_all_markers(self):
        delete_marker = Marker()

        #delete_marker.header.frame_id = FRAME_ID
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
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 3, cv2.LINE_AA)
        # create the arrow hooks
        p[0] = q[0] + 9 * cos(angle + pi / 4)
        p[1] = q[1] + 9 * sin(angle + pi / 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 3, cv2.LINE_AA)
        p[0] = q[0] + 9 * cos(angle - pi / 4)
        p[1] = q[1] + 9 * sin(angle - pi / 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 3, cv2.LINE_AA)

    def visualize(self, result, image):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt

        height, width = image.shape[:2]
        plt.rcParams["figure.subplot.left"] = 0
        plt.rcParams["figure.subplot.bottom"] = 0
        plt.rcParams["figure.subplot.right"] = 1
        plt.rcParams["figure.subplot.top"] = 1

        fig = Figure(figsize=(width/100,height/100))
        # fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axes = fig.gca()
        # axes = fig.add_axes([0.,0.,1.,1.])
        visualize.display_instances(image, result['rois'], result['masks'],
                                    result['class_ids'], self.class_names,
                                    result['scores'], ax=axes)
        #fig.tight_layout()
        canvas.draw()
        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        _, _, w, h = fig.bbox.bounds
        result = result.reshape((int(h), int(w), 3))
        return result

def main():
    rospy.init_node('mask_rcnn')

    node = MaskRCNNNode()
    node.run()

if __name__ == '__main__':
    main()
