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
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from mask_rcnn_ros.msg import MaskRCNNMsg
from mask_rcnn_ros.srv import *
#from region_growing_segmentation.srv import *

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import lab
import cv2
from math import atan2, cos, sin, sqrt, pi
from cv_bridge import CvBridge

from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from keras.backend import tensorflow_backend as backend

ROOT_DIR = os.path.abspath(roslib.packages.get_pkg_dir('mask_rcnn_ros'))
MODEL_DIR = os.path.join(ROOT_DIR, "models/")
CLASS_NAME = "caloriemate"
MODEL = os.path.join(MODEL_DIR, "mask_rcnn_lab_" + CLASS_NAME + ".h5")

CAMERA_INFO_TOPIC = "/pylon_camera_node/camera_info"
IMAGE_TOPIC = "/pylon_camera_node/image_rect"
DEPTH_TOPIC = "/phoxi_camera/aligned_depth_map"
FRAME_ID = "basler_ace_rgb_sensor_calibrated"

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
        self.is_service_called = False

        # Create model object in inference mode.
        config = InferenceConfig()
        #config.display()
        self.model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)
        self.model.load_weights(MODEL, by_name=True)
        self.model.keras_model._make_predict_function()
        self.class_names = ['BG', CLASS_NAME]

        # Set calibration matrix
        rospy.loginfo("Waiting camera info topic")
        camera_info = rospy.wait_for_message(CAMERA_INFO_TOPIC, CameraInfo, 10)
        self.camera_matrix = np.array(camera_info.K).reshape([3, 3])
        self.fx = self.camera_matrix[0][0]
        self.fy = self.camera_matrix[1][1]
        self.cx = self.camera_matrix[0][2]
        self.cy = self.camera_matrix[1][2]
        self.dist_coeffs = np.array(camera_info.D)
        self.map = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, np.eye(3), self.camera_matrix, (camera_info.width, camera_info.height), cv2.CV_32FC1)

        width = camera_info.width
        height = camera_info.height
        size = height, width, 3
        self.dummy_image = np.zeros(size, dtype=np.uint8)
        rospy.loginfo("Acquired camera info")       
    
    def run(self):
        # Define publisher
        self.result_pub = rospy.Publisher(rospy.get_name() + '/MaskRCNNMsg', MaskRCNNMsg, queue_size=1, latch=True)
        self.vis_original_pub = rospy.Publisher(rospy.get_name() + '/original_result', Image, queue_size=1, latch=True)
        self.vis_processed_pub = rospy.Publisher(rospy.get_name() + '/processed_result', Image, queue_size=1, latch=True)
        self.vis_depth_pub = rospy.Publisher(rospy.get_name() + '/aligned_depth_map', Image, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher(rospy.get_name() + '/axes', MarkerArray, queue_size=1, latch=True)

        # Define service
        result_srv = rospy.Service(rospy.get_name() + '/MaskRCNNSrv', MaskRCNNSrv, self.wait_frame)
        set_model_srv = rospy.Service(rospy.get_name() + "/set_model", SetModel, self.set_model)

        self.model.detect([self.dummy_image], verbose=0)
        rospy.loginfo("Ready to be called service")

        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.is_service_called:
                vis_original = self.visualize(self.result, self.image)
                cv_result = np.zeros(shape=self.image.shape, dtype=np.uint8)
                cv2.convertScaleAbs(vis_original, cv_result)
                msg = self.cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                self.vis_original_pub.publish(msg)

                cv_result = np.zeros(shape=self.vis_processed_result.shape, dtype=np.uint8)
                cv2.convertScaleAbs(self.vis_processed_result, cv_result)
                msg = self.cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                self.vis_processed_pub.publish(msg)

                cv_result = np.zeros(shape=self.vis_depth.shape, dtype=np.uint8)
                cv2.convertScaleAbs(self.vis_depth, cv_result)
                msg = self.cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                self.vis_depth_pub.publish(msg)
                self.is_service_called = False           
 
            r.sleep()

        rospy.spin()

    def set_model(self, req):
        res = SetModelResponse()
    
        CLASS_NAME = req.class_name
        MODEL = os.path.join(MODEL_DIR, "mask_rcnn_lab_" + CLASS_NAME + ".h5")
        
        # Create model object in inference mode.
        try:
            backend.clear_session()
            config = InferenceConfig()
            self.model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)
            self.model.load_weights(MODEL, by_name=True)
            self.model.keras_model._make_predict_function()
        except Exception as e:
            res.message = str(e)
            res.success = False
            rospy.logerr(e)
            return res
        
        self.class_names = ['BG', CLASS_NAME]
        self.model.detect([self.dummy_image], verbose=0)
        rospy.loginfo("Finished setting " + CLASS_NAME + " model")

        res.message = "OK"
        res.success = True

        return res

    def wait_frame(self, req):
        res = MaskRCNNSrvResponse()

        rospy.loginfo("Waiting frame ...")
        timeout = 10
        image_msg = rospy.wait_for_message(IMAGE_TOPIC, Image, timeout)
        depth_msg = rospy.wait_for_message(DEPTH_TOPIC, Image, timeout)
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
        np_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        np_depth = cv2.remap(np_depth, self.map[0], self.map[1], cv2.INTER_NEAREST)
        #np_depth = cv2.warpAffine(np_depth, np.float32([[1,0,-10],[0,1,0]]), (np_depth.shape[1],np_depth.shape[0]))

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

        self.result = results[0]
        result_msg = self.build_result_msg(image_msg.header, results[0], np_image, np_depth)

        self.is_service_called = True
        return result_msg

    def build_result_msg(self, msg_header, result, image, depth):
        rospy.loginfo("Building msg ...")
        result_msg = MaskRCNNMsg()
        result_msg.header = msg_header
        result_msg.header.frame_id = FRAME_ID
        result_msg.count = 0

        self.vis_processed_result = np.copy(image)
        self.vis_depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        axes_msg = MarkerArray()
        delete_marker = self.delete_all_markers()
        axes_msg.markers.append(delete_marker)

        mask_msgs=[]
        
        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
            rospy.loginfo("ID:%s '%s' is detected", str(i), self.class_names[result['class_ids'][i]])
            mask = result['masks'][:,:,i].astype(np.uint8)
            area, center, x_axis, y_axis = self.estimate_object_attribute(mask, depth)
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
           

            #axes_msg.markers.append(x_axis_marker)
            #axes_msg.markers.append(y_axis_marker)
            #axes_msg.markers.append(z_axis_marker)
            
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

            m = self.cv_bridge.cv2_to_imgmsg(mask, 'mono8')
            mask_msgs.append(m)
            #m = Image()
            #m.header = msg_header
            #m.height = result['masks'].shape[0]
            #m.width = result['masks'].shape[1]
            #m.encoding = "mono8"
            #m.is_bigendian = False
            #m.step = m.width
            #m.data = (result['masks'][:, :, i] * 255).tobytes()
            #mask_msgs.append(m)

            #caption = "{} {:.3f}".format(class_name, score) if score else class_name
            id_caption = "ID:" + str(i)
            class_caption = class_name + " " +  str(round(score, 3))
            cv2.putText(self.vis_processed_result, id_caption, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 2, cv2.LINE_AA) 
            cv2.putText(self.vis_processed_result, class_caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 2, cv2.LINE_AA)
            cv2.putText(self.vis_depth, id_caption, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 2, cv2.LINE_AA) 
            cv2.putText(self.vis_depth, class_caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 2, cv2.LINE_AA)

        # service call get_normal in mask_rcnn_utils.py
        #try:
        #    get_normal = rospy.ServiceProxy("/mask_rcnn_utils/get_normal", GetNormal)
        #    res = get_normal(result_msg.centers)
        #except rospy.ServiceException as e:
        #    rospy.logerr("Service calll failed: ",e)

        if len(result_msg.class_names) == 0:
            rospy.loginfo("Service Get Normal finished")

            self.result_pub.publish(result_msg)
            self.marker_pub.publish(axes_msg)

            rospy.loginfo("Published msg completely")
            return result_msg

        if result_msg.class_names[0] == "koiwashi":
            for i, (center,normal) in enumerate(zip(result_msg.centers,result_msg.normals)):
                normal.vector.x = 0.0
                normal.vector.y = 0.0
                normal.vector.z = -1.0
                normal_marker = self.build_marker_msg(center.header.frame_id, Marker.ARROW, i, center.point, normal.vector, 0.0, 0.0, 1.0, "normal")
                text_marker = self.build_marker_msg(center.header.frame_id, Marker.TEXT_VIEW_FACING, i, center.point, normal.vector, 1.0, 1.0, 1.0, "id_text")
                axes_msg.markers.append(normal_marker)
                axes_msg.markers.append(text_marker)

        else:
            try:
                get_masked_surface = rospy.ServiceProxy("/mask_region_growing/get_masked_surface", GetMaskedSurface)
                res = get_masked_surface(mask_msgs)
            except rospy.ServiceException as e:
                rospy.logerr("Service calll failed: ",e)

            result_msg.ids = []
            count = 0
            result_msg.count = 0
            result_msg.centers = []
            result_msg.normals = []
            result_msg.axes = []
            for i, (centers_msg, normals_msg) in enumerate(zip(res.centers_list, res.normals_list)):
                
                for j, (center, normal) in enumerate(zip(centers_msg.centers, normals_msg.normals)):
                    result_msg.normals.append(normal)
                    result_msg.centers.append(center)
                    result_msg.axes.append(normal)
                    normal_marker = self.build_marker_msg(center.header.frame_id, Marker.ARROW, count, center.point, normal.vector, 0.0, 0.0, 1.0, "normal")
                    text_marker = self.build_marker_msg(center.header.frame_id, Marker.TEXT_VIEW_FACING, count, center.point, normal.vector, 1.0, 1.0, 1.0, "id_text")
                    axes_msg.markers.append(normal_marker)
                    axes_msg.markers.append(text_marker)
    
                    result_msg.ids.append(count)
                    result_msg.count += 1
                    count+=1
        
        rospy.loginfo("Service Get Normal finished")

        self.result_pub.publish(result_msg)
        self.marker_pub.publish(axes_msg)

        rospy.loginfo("Published msg completely")

        return result_msg

    def estimate_object_attribute(self, mask, depth):
        ret, thresh = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Because of one mask, the number of contours should be one.
        #if len(contours) != 1:
        #    rospy.logwarn("Skip this object.(Inferenced mask is not clearly.)")
        #    return (0, 0, 0, 0)
        if len(contours) > 1:
            contours.sort(key=len, reverse=True)
            rospy.logwarn("This object is used the biggest mask")

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
        obj_center_pixel = (int(mean[0,0]), int(mean[0,1]))
        unit_x_axis_pixel = np.array([eigenvectors[0,0] * eigenvalues[0,0], eigenvectors[0,1] * eigenvalues[0,0]]) / math.sqrt((eigenvectors[0,0] * eigenvalues[0,0])**2 + (eigenvectors[0,1] * eigenvalues[0,0])**2)
        unit_y_axis_pixel = np.array([eigenvectors[1,0] * eigenvalues[1,0], eigenvectors[1,1] * eigenvalues[1,0]]) / math.sqrt((eigenvectors[1,0] * eigenvalues[1,0])**2 + (eigenvectors[1,1] * eigenvalues[1,0])**2)

        cv2.circle(self.vis_depth, obj_center_pixel, 5, (0, 0, 255), 3)
        cv2.drawContours(self.vis_depth, contours, 0, (255, 255, 0), 6)

        # Check center point of Depth Value
        obj_center_depth, x_axis_depth, y_axis_depth = self.get_depth(depth, obj_center_pixel, unit_x_axis_pixel, unit_y_axis_pixel)
        #if obj_center_depth == 0.0 or x_axis_depth == 0.0 or y_axis_depth == 0.0:
        if obj_center_depth == 0.0:
            rospy.logwarn("Skip this object.(Depth value around center point is all 0.)")
            return (0, 0, 0, 0)

        # Calculate center point on camera coordiante
        u = obj_center_pixel[0]
        v = obj_center_pixel[1]
        obj_center_z = obj_center_depth
        obj_center_x = (u - self.cx) * obj_center_z / self.fx
        obj_center_y = (v - self.cy) * obj_center_z / self.fy
        obj_center_camera = Point(obj_center_x, obj_center_y, obj_center_z)

        # Calculate x-axis
        np_x_axis_pixel = np.array([obj_center_pixel[0] + unit_x_axis_pixel[0]*20, obj_center_pixel[1] + unit_x_axis_pixel[1]*20], dtype=self.camera_matrix.dtype)
        u_x_axis = np_x_axis_pixel[0]
        v_x_axis = np_x_axis_pixel[1]
        x_axis_z = x_axis_depth - obj_center_z
        x_axis_x = (u_x_axis - self.cx) * x_axis_depth / self.fx - obj_center_x 
        x_axis_y = (v_x_axis - self.cy) * x_axis_depth / self.fy - obj_center_y
        magnitude_x_axis = sqrt(x_axis_x**2 + x_axis_y**2 + x_axis_z**2)
        x_axis_camera = Vector3(x_axis_x/magnitude_x_axis, x_axis_y/magnitude_x_axis, x_axis_z/magnitude_x_axis)

        np_y_axis_pixel = np.array([obj_center_pixel[0] - unit_y_axis_pixel[0]*20, obj_center_pixel[1] - unit_y_axis_pixel[1]*20], dtype=self.camera_matrix.dtype)
        u_y_axis = np_y_axis_pixel[0]
        v_y_axis = np_y_axis_pixel[1]
        y_axis_z = y_axis_depth - obj_center_z
        y_axis_x = (u_y_axis - self.cx) * y_axis_depth / self.fy - obj_center_x 
        y_axis_y = (v_y_axis - self.cy) * y_axis_depth / self.fy - obj_center_y
        magnitude_y_axis = sqrt(y_axis_x**2 + y_axis_y**2 + y_axis_z**2)
        y_axis_camera = Vector3(y_axis_x/magnitude_x_axis, y_axis_y/magnitude_x_axis, y_axis_z/magnitude_x_axis)

        # visualize mask, axis
        cv2.drawContours(self.vis_processed_result, contours, 0, (255, 255, 0), 6)
        cntr = (int(mean[0,0]), int(mean[0,1]))
        cv2.circle(self.vis_processed_result, cntr, 10, (255, 0, 255), -1)
        p1 = (cntr[0] + unit_x_axis_pixel[0]*20, cntr[1] + unit_x_axis_pixel[1]*20)
        p2 = (cntr[0] - unit_y_axis_pixel[0]*20, cntr[1] - unit_y_axis_pixel[1]*20)
        self.drawAxis(self.vis_processed_result, cntr, p1, (0, 0, 255), 3)
        self.drawAxis(self.vis_processed_result, cntr, p2, (0, 255, 0), 3)

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
                    rospy.logwarn("This object's depth value is averaged around the center point")

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
            axis_marker.pose.position.x = center.x + axis.x * 0.04
            axis_marker.pose.position.y = center.y + axis.y * 0.04
            axis_marker.pose.position.z = center.z - 0.01 + axis.z * 0.04
            return axis_marker

        start_point = center
        # end_point = x_axis
        end_point = Point(center.x + axis.x * 0.04, center.y + axis.y * 0.04, center.z + axis.z * 0.04)
        axis_marker.points = [start_point, end_point]

        return axis_marker

    def delete_all_markers(self):
        delete_marker = Marker()

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
