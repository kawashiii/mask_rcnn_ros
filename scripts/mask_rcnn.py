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
from std_msgs.msg import String
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from mask_rcnn_ros_msgs.msg import MaskRCNNMsg
from mask_rcnn_ros_msgs.srv import *
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
CLASS_NAME = "choice"
MODEL = os.path.join(MODEL_DIR, "mask_rcnn_lab_" + CLASS_NAME + ".h5")

CAMERA_INFO_TOPIC = "/pylon_camera_node/camera_info"
IMAGE_TOPIC = "/pylon_camera_node/image_rect"
IMAGE_TOPIC2 = "/phoxi_camera/external_camera_texture"
DEPTH_TOPIC = "/phoxi_camera/aligned_depth_map"
FRAME_ID = "basler_ace_rgb_sensor_calibrated"

DEBUG_IMAGE_TOPIC = "/debug/image_rect"

# if the depth of (x, y) is 0, it is approximate depth value around specific pixel
DEPTH_APPROXIMATE_RANGE = 10

class InferenceConfig(lab.LabConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.5

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

        if rospy.has_param("/mask_rcnn/debug_mode") and rospy.get_param("/mask_rcnn/debug_mode"):
            rospy.logwarn("'mask_rcnn' node is Debug Mode")
            global IMAGE_TOPIC
            IMAGE_TOPIC = DEBUG_IMAGE_TOPIC       
    
    def run(self):
        # Define Publisher
        self.result_pub = rospy.Publisher(rospy.get_name() + '/MaskRCNNMsg', MaskRCNNMsg, queue_size=1, latch=True)
        self.vis_original_pub = rospy.Publisher(rospy.get_name() + '/original_result', Image, queue_size=1, latch=True)
        self.vis_processed_pub = rospy.Publisher(rospy.get_name() + '/processed_result', Image, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher(rospy.get_name() + '/axes', MarkerArray, queue_size=1, latch=True)
        self.input_image_pub = rospy.Publisher(rospy.get_name() + "/input_image", Image, queue_size=1, latch=True)
        self.trigger_id_pub = rospy.Publisher(rospy.get_name() + "/trigger_id", Int32, queue_size=1, latch=True)

        # Define Subscriber
        rospy.Subscriber("/phoxi_camera/trigger_id", Int32, self.callback_sub_trigger)

        # Define Service
        get_detection_srv = rospy.Service(rospy.get_name() + '/MaskRCNNSrv', MaskRCNNSrv, self.callback_get_detection)
        set_model_srv = rospy.Service(rospy.get_name() + "/set_model", SetModel, self.callback_set_model)

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

                cv_result = np.zeros(shape=self.image.shape, dtype=np.uint8)
                cv2.convertScaleAbs(self.image, cv_result)
                msg = self.cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                self.input_image_pub.publish(msg)
                self.is_service_called = False

            r.sleep()

        rospy.spin()

    def callback_set_model(self, req):
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

    def callback_sub_trigger(self, msg):
        trigger_id = msg.data

        # Get Image
        rospy.loginfo("Waiting frame ...")
        timeout = 10
        image_msg = rospy.wait_for_message(IMAGE_TOPIC2, Image, timeout)
        rospy.loginfo("Acquired frame")

        start_build_msg = time.time()

        # Convert Image
        np_image = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        self.image = np.copy(np_image)
        
        # Run Detection
        input_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        start_detection = time.time() 
        rospy.loginfo("Detecting ...")
        results = self.model.detect([input_image], verbose=0)
        end_detection = time.time()
        detection_time = end_detection - start_detection
        rospy.loginfo("%s[s] (Detection time)", round(detection_time, 3))

        # Build msg
        self.result = results[0]
        res = self.build_result_msg()

        end_build_msg = time.time() 
        build_msg_time = end_build_msg - start_build_msg
        rospy.loginfo("%s[s] (Total time)", round(build_msg_time, 3))

        self.is_service_called = True
        self.trigger_id_pub.publish(trigger_id)

    def callback_get_detection(self, req):
        res = MaskRCNNSrvResponse()

        # Get Image
        rospy.loginfo("Waiting frame ...")
        timeout = 10
        image_msg = rospy.wait_for_message(IMAGE_TOPIC, Image, timeout)
        rospy.loginfo("Acquired frame")

        start_build_msg = time.time()

        # Convert Image
        np_image = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        #height,width = np_image.shape[:2]
        #center = (int(width/2), int(height/2))
        #angle = 30.0
        #scale = 1.0
        #trans = cv2.getRotationMatrix2D(center, angle , scale)
        #np_image = cv2.warpAffine(np_image, trans, (width,height))
        self.image = np.copy(np_image)
        
        # Run Detection
        input_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        start_detection = time.time() 
        rospy.loginfo("Detecting ...")
        results = self.model.detect([input_image], verbose=0)
        end_detection = time.time()
        detection_time = end_detection - start_detection
        rospy.loginfo("%s[s] (Detection time)", round(detection_time, 3))

        # Build msg
        self.result = results[0]
        res.detectedMaskRCNN = self.build_result_msg()

        end_build_msg = time.time() 
        build_msg_time = end_build_msg - start_build_msg
        rospy.loginfo("%s[s] (Total time)", round(build_msg_time, 3))

        self.is_service_called = True
        return res

    def build_result_msg(self):
        rospy.loginfo("Building msg ...")
        result_msg = MaskRCNNMsg()
        result_msg.header.stamp = rospy.Time.now()
        result_msg.header.frame_id = FRAME_ID
        result_msg.count = 0

        self.vis_processed_result = np.copy(self.image)
        axes_marker_msg = MarkerArray()
        delete_marker = self.delete_all_markers()
        axes_marker_msg.markers.append(delete_marker)

        mask_msgs=[]
        is_rigid_object = True
        
        for i, (y1, x1, y2, x2) in enumerate(self.result['rois']):
            rospy.loginfo("ID:%s '%s' is detected", str(i), self.class_names[self.result['class_ids'][i]])
            result_msg.ids.append(result_msg.count)
            result_msg.count += 1

            box = RegionOfInterest()
            box.x_offset = np.asscalar(x1)
            box.y_offset = np.asscalar(y1)
            box.height = np.asscalar(y2 - y1)
            box.width = np.asscalar(x2 - x1)
            result_msg.boxes.append(box)

            class_id = self.result['class_ids'][i]
            result_msg.class_ids.append(class_id)

            class_name = self.class_names[class_id]
            result_msg.class_names.append(class_name)
            if class_name == "koiwashi": is_rigid_object = False

            score = self.result['scores'][i]
            result_msg.scores.append(score)

            mask = self.result['masks'][:,:,i].astype(np.uint8)
            biggest_mask = self.get_biggest_mask(mask)
            mask_msg = self.cv_bridge.cv2_to_imgmsg(biggest_mask, 'mono8')
            mask_msgs.append(mask_msg)
           
            # Draw information to image 
            id_caption = "ID:" + str(i)
            class_caption = class_name + " " +  str(round(score, 3))
            cv2.putText(self.vis_processed_result, id_caption, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 2, cv2.LINE_AA) 
            cv2.putText(self.vis_processed_result, class_caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 2, cv2.LINE_AA)

        if len(result_msg.class_names) == 0:
            self.result_pub.publish(result_msg)
            self.marker_pub.publish(axes_marker_msg)

            rospy.logwarn("No objects could be detected.")
            return result_msg

        try:
            get_masked_surface = rospy.ServiceProxy("/mask_region_growing/get_masked_surface", GetMaskedSurface)
            res = get_masked_surface(is_rigid_object, mask_msgs)
        except rospy.ServiceException as e:
            rospy.logerr("Service calll failed: ",e)


        for i in range(result_msg.count):
            masked_object_attrs = res.moas_list[i]
            result_msg.x_axes.append(masked_object_attrs.x_axes[0])
            result_msg.y_axes.append(masked_object_attrs.y_axes[0])
            result_msg.z_axes.append(masked_object_attrs.z_axes[0])
            result_msg.axes.append(masked_object_attrs.x_axes[0])
            result_msg.polygons.append(masked_object_attrs.corners[0])
            result_msg.areas.append(masked_object_attrs.areas[0])
            result_msg.centers.append(masked_object_attrs.centers[0])
            result_msg.normals.append(masked_object_attrs.normals[0])

            center_msg = masked_object_attrs.centers[0]
            normal_msg = masked_object_attrs.normals[0]
          
            normal_marker = self.build_marker_msg(center_msg.header.frame_id, Marker.ARROW, i, center_msg.point, normal_msg.vector, 0.0, 0.0, 1.0, "normal")
            text_marker = self.build_marker_msg(center_msg.header.frame_id, Marker.TEXT_VIEW_FACING, i, center_msg.point, normal_msg.vector, 1.0, 1.0, 1.0, "id_text")
            axes_marker_msg.markers.append(normal_marker)
            axes_marker_msg.markers.append(text_marker)

        #sort_item = []
        #check_center_list = []
        #for i, (centers_msg, normals_msg, areas_msg) in enumerate(zip(res.centers_list, res.normals_list, res.areas_list)):
        #    for j, (center, normal, area) in enumerate(zip(centers_msg.centers, normals_msg.normals, areas_msg.areas)):
        #        if (center.point.z == 0.0) : continue
        #        if center.point in check_center_list : continue
        #        check_center_list.append(center.point)
        #        sort_item.append([i, center, normal, result_msg.boxes[i], area])

        #highest_item = sorted(sort_item, key=lambda x:x[1].point.z)
        #highest_good_normal_list = []
        #most_same_height_list = []
        #height = -1
        #for item in highest_item:
        #    if len(most_same_height_list) == 0:
        #        height = item[1].point.z
        #        most_same_height_list.append(item)
        #        continue

        #    if item[1].point.z - height < 0.02:
        #        most_same_height_list.append(item)
        #    else:
        #        bad_normal_list = []
        #        for h in most_same_height_list:
        #            if h[2].vector.z < -0.9:
        #                highest_good_normal_list.append(h)
        #            else:
        #                bad_normal_list.append(h)
        #        if len(bad_normal_list) != 0: 
        #            sorted_bad_normal_list = sorted(bad_normal_list, key=lambda x:x[2].vector.z)
        #            highest_good_normal_list += sorted_bad_normal_list
        #        #good_normal_list = sorted(most_same_height_list, key=lambda x:x[2].vector.z)
        #        #highest_good_normal_list += good_normal_list
        #        most_same_height_list = []
        #        most_same_height_list.append(item)
        #        height = item[1].point.z

        #if len(most_same_height_list) != 0:
        #    good_normal_list = sorted(most_same_height_list, key=lambda x:x[2].vector.z)
        #    highest_good_normal_list += good_normal_list
            
        #result_msg.count = 0
        #result_msg.ids = []
        #result_msg.boxes = []
        #for i, item in enumerate(highest_good_normal_list):
        #    result_msg.ids.append(i)
        #    result_msg.centers.append(item[1])
        #    result_msg.normals.append(item[2])
        #    result_msg.axes.append(item[2])
        #    result_msg.boxes.append(item[3])
        #    result_msg.areas.append(item[4])
        #    result_msg.count += 1
        #        
        #    normal_marker = self.build_marker_msg(item[1].header.frame_id, Marker.ARROW, i, item[1].point, item[2].vector, 0.0, 0.0, 1.0, "normal")
        #    text_marker = self.build_marker_msg(item[1].header.frame_id, Marker.TEXT_VIEW_FACING, i, item[1].point, item[2].vector, 1.0, 1.0, 1.0, "id_text")
        #    axes_marker_msg.markers.append(normal_marker)
        #    axes_marker_msg.markers.append(text_marker)

        
        self.result_pub.publish(result_msg)
        self.marker_pub.publish(axes_marker_msg)

        rospy.loginfo("Published msg completely")

        return result_msg

    def get_biggest_mask(self, mask):
        ret, thresh = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 1:
            contours.sort(key=len, reverse=True)
            rospy.logwarn("This object is used the biggest mask")

        contour = contours[0]
        zero_image = np.zeros(mask.shape, dtype=np.uint8)
        ret_mask = cv2.fillPoly(zero_image, [contour], color=(255,255,255))
        self.vis_processed_result = cv2.fillPoly(self.vis_processed_result, [contour], color=(255,200,255))

        return ret_mask

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
