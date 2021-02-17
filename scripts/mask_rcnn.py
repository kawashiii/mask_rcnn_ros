#!/usr/bin/env python3
import os
import sys
import copy
import time
import math
import random
import colorsys
import datetime
import threading
import numpy as np

import rospy
import rosparam
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
from mask_rcnn_ros_msgs.msg import *
from mask_rcnn_ros_msgs.srv import *

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import lab_config
from math import atan2, cos, sin, sqrt, pi
from cv_bridge import CvBridge

from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from keras.backend import tensorflow_backend as backend

ROOT_DIR = os.path.abspath(roslib.packages.get_pkg_dir('mask_rcnn_ros'))
MODEL_DIR = os.path.join(ROOT_DIR, "models/")

class MaskRCNNNode(object):
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.is_service_called = False
        self.target_object = "beads"
        self.class_names = lab_config.CLASS_NAMES
        self.object_attributes, _ = rosparam.load_file(ROOT_DIR + "/config/object_attributes.yaml")[0]

        # Create model object in inference mode.
        config = lab_config.LabConfig()
        model_file = os.path.join(MODEL_DIR, lab_config.TRAINED_MODEL)
        self.model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)
        self.model.load_weights(model_file, by_name=True)
        self.model.keras_model._make_predict_function()

        # Set camera topic
        rospy.loginfo("Waiting camera info topic")
        camera_info = rospy.wait_for_message(lab_config.CAMERA_INFO_TOPIC, CameraInfo, 10)
        camera_matrix = np.array(camera_info.K).reshape([3, 3])
        self.fx = camera_matrix[0][0]
        self.fy = camera_matrix[1][1]
        self.cx = camera_matrix[0][2]
        self.cy = camera_matrix[1][2]
        width = camera_info.width
        height = camera_info.height
        size = height, width, config.IMAGE_CHANNEL_COUNT
        self.dummy_image = np.zeros(size, dtype=np.uint8)
        rospy.loginfo("Acquired camera info")

        # Create log dir
        today = datetime.datetime.now()
        log_dir = os.path.join(ROOT_DIR, "logs/" + str(today.year) + str(today.month).zfill(2) + str(today.day).zfill(2))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        dir_num = len(os.listdir(log_dir))
        self.log_dir = os.path.join(log_dir, str(dir_num))
        os.makedirs(self.log_dir)
        self.save_id = 0

        # for change depth coordinate from camera to container
        self.mesh_width, self.mesh_height = np.meshgrid(np.arange(width), np.arange(height))

        self.debug_mode = 0      
        if rospy.has_param("/mask_rcnn/debug_mode") and rospy.get_param("/mask_rcnn/debug_mode"):
            rospy.logwarn("'mask_rcnn' node is Debug Mode")
            self.debug_mode = 1

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

        self.target_object = req.class_name
    
        # class_name = req.class_name
        # model_file = os.path.join(MODEL_DIR, "mask_rcnn_lab_" + class_name + ".h5")
        
        # # Create model object in inference mode.
        # try:
        #     backend.clear_session()
        #     config = MaskRCNNConfig()
        #     config.IMAGE_CHANNEL_COUNT = self.param[self.param["input_data_type"]+"_config"]["IMAGE_CHANNEL_COUNT"]
        #     config.MEAN_PIXEL = self.param[self.param["input_data_type"]+"_config"]["MEAN_PIXEL"]
        #     self.model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)
        #     self.model.load_weights(model_file, by_name=True)
        #     self.model.keras_model._make_predict_function()
        # except Exception as e:
        #     res.message = str(e)
        #     res.success = False
        #     rospy.logerr(e)
        #     return res
        
        # self.class_names = ['BG', class_name]
        # self.model.detect([self.dummy_image], verbose=0)
        # rospy.loginfo("Finished setting " + class_name + " model")

        res.message = "OK"
        res.success = True

        return res

    def callback_sub_trigger(self, msg):
        start_build_msg = time.time()

        trigger_id = msg.data
        self.detect()
        res = self.build_result_msg()

        end_build_msg = time.time() 
        build_msg_time = end_build_msg - start_build_msg
        rospy.loginfo("%s[s] (Total time)", round(build_msg_time, 3))

        self.is_service_called = True
        self.trigger_id_pub.publish(trigger_id)

    def callback_get_detection(self, req):
        start_build_msg = time.time()

        res = MaskRCNNSrvResponse()
        self.detect()
        res.detectedMaskRCNN = self.build_result_msg()

        end_build_msg = time.time() 
        build_msg_time = end_build_msg - start_build_msg
        rospy.loginfo("%s[s] (Total time)", round(build_msg_time, 3))

        self.is_service_called = True
        return res

    def detect(self):
        # Get Image
        rospy.loginfo("Waiting frame ...")
        timeout = 10
        image_topic = lab_config.IMAGE_TOPIC
        depth_topic = lab_config.DEPTH_TOPIC
        if self.debug_mode:
            image_topic = "/debug" + image_topic
            depth_topic = "/debug" + depth_topic
        image_msg = rospy.wait_for_message(image_topic, Image, timeout)
        depth_msg = rospy.wait_for_message(depth_topic, Image, timeout)
        rospy.loginfo("Acquired frame")

        # Convert Image
        np_image = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        np_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        if not self.debug_mode:
            cv2.imwrite(self.log_dir + "/" + str(self.save_id).zfill(4) + ".png", np_image)
            np.save(self.log_dir + "/" + str(self.save_id).zfill(4) + "_depth", np_depth)
            self.save_id += 1
        
        self.image = np.copy(np_image)
        
        if lab_config.DEPTH_COORDINATE == "container":
            height, width = np_depth.shape
            xyz_image = np.zeros((height, width, 3), np.float32)
            xyz_image[...,2] = np_depth/1000
            xyz_image[...,0] = (self.mesh_width - self.cx) * xyz_image[...,2] / self.fx
            xyz_image[...,1] = (self.mesh_height - self.cy) * xyz_image[...,2] / self.fy
            xyz_image = (np.dot(lab_config.ROT, xyz_image.reshape(-1,3).T).T.reshape(height, width, 3) + lab_config.TRANS) * 1000
            np_depth = (np.copy(xyz_image[...,2])).astype(np.float32)
        self.depth = np.copy(np_depth)
        
        # Prepare Input Data to Detect
        input_data = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        input_data = lab_config.NORMALIZE_TEXTURE(input_data)
        if lab_config.INPUT_DATA_TYPE == "depth":
            input_data = np_depth.reshape([np_depth.shape[0], np_depth.shape[1], 1])
            input_data = lab_config.NORMALIZE_DEPTH(input_data)

        # Run Detection
        start_detection = time.time() 
        rospy.loginfo("Detecting ...")
        results = self.model.detect([input_data], verbose=0)
        end_detection = time.time()
        detection_time = end_detection - start_detection
        rospy.loginfo("%s[s] (Detection time)", round(detection_time, 3))

        # Build msg
        self.result = results[0]
        

    def build_result_msg(self):
        rospy.loginfo("Building msg ...")
        result_msg = MaskRCNNMsg()
        result_msg.header.stamp = rospy.Time.now()
        result_msg.header.frame_id = lab_config.FRAME_ID
        result_msg.count = 0

        self.vis_processed_result = np.copy(self.image)
        self.vis_processed_result = cv2.cvtColor(cv2.cvtColor(self.vis_processed_result, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        axes_marker_msg = MarkerArray()
        delete_marker = self.delete_all_markers()
        axes_marker_msg.markers.append(delete_marker)

        masked_depth_std_msg = []
        mask_msgs=[]
        masks = []
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
            masks.append(biggest_mask)
            mask_msg = self.cv_bridge.cv2_to_imgmsg(biggest_mask, 'mono8')
            mask_msgs.append(mask_msg)

            index_for_std = np.bitwise_and(mask>0.5, self.depth<400.0)
            masked_depth_std = np.std(self.depth[index_for_std])
            print(masked_depth_std)
            masked_depth_std_msg.append(masked_depth_std)

        try:
            get_masked_surface = rospy.ServiceProxy("/mask_region_growing/get_masked_surface", GetMaskedSurface)
            res = get_masked_surface(is_rigid_object, mask_msgs, masked_depth_std_msg)
        except rospy.ServiceException as e:
            rospy.logerr("Service calll failed: ",e)

        final_result_msg = MaskRCNNMsg()
        final_result_msg.header.stamp = rospy.Time.now()
        final_result_msg.header.frame_id = lab_config.FRAME_ID
        final_result_msg.count = 0
        for i in range(result_msg.count):
            masked_object_attrs = res.moas_list[i]
            mask_color = [1.0, 0.0, 0.0]
            if masked_object_attrs.surface_count == 0:
                continue

            final_result_msg.ids.append(final_result_msg.count)
            final_result_msg.count += 1
            final_result_msg.x_axes.append(masked_object_attrs.x_axes[0])
            final_result_msg.y_axes.append(masked_object_attrs.y_axes[0])
            final_result_msg.z_axes.append(masked_object_attrs.z_axes[0])
            final_result_msg.axes.append(masked_object_attrs.x_axes[0])
            final_result_msg.polygons.append(masked_object_attrs.corners[0])
            final_result_msg.areas.append(masked_object_attrs.areas[0])
            final_result_msg.centers.append(masked_object_attrs.centers[0])
            final_result_msg.normals.append(masked_object_attrs.normals[0])
            if not self.check_object_size(masked_object_attrs):
                rospy.logwarn("Object Id %d is out of gt size", i)
                mask_color = [0.0, 0.0, 1.0]
                final_result_msg.scores.append(0.0)
            else:
                final_result_msg.scores.append(result_msg.scores[i])
            
            self.draw_mask(i, masks[i], mask_color)
            
            center_msg = masked_object_attrs.centers[0]
            normal_msg = masked_object_attrs.normals[0]
          
            normal_marker = self.build_marker_msg(center_msg.header.frame_id, Marker.ARROW, i, center_msg.point, normal_msg.vector, 0.0, 0.0, 1.0, "normal")
            text_marker = self.build_marker_msg(center_msg.header.frame_id, Marker.TEXT_VIEW_FACING, i, center_msg.point, normal_msg.vector, 1.0, 1.0, 1.0, "id_text")
            axes_marker_msg.markers.append(normal_marker)
            axes_marker_msg.markers.append(text_marker)

        
        self.result_pub.publish(final_result_msg)
        self.marker_pub.publish(axes_marker_msg)

        rospy.loginfo("Published msg completely")

        return final_result_msg

    def get_biggest_mask(self, mask):
        ret, thresh = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 1:
            contours.sort(key=len, reverse=True)
            rospy.logwarn("This object is used the biggest mask")

        contour = contours[0]
        zero_image = np.zeros(mask.shape, dtype=np.uint8)
        ret_mask = cv2.fillPoly(zero_image, [contour], color=(255,255,255))

        return ret_mask

    def draw_mask(self, index, mask, mask_color):
        alpha = 0.5
        #brightness = 1.0
        #hsv = [(i/10, 1, brightness)  for i in range(10)]
        #colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        #random.shuffle(colors)
        #color = colors[0]
        color = mask_color
        
        for c in range(3):
            self.vis_processed_result[:, :, c] = np.where(mask >= 1, self.vis_processed_result[:, :, c] * (1 - alpha) + alpha * color[c] * 255, self.vis_processed_result[:, :, c])
        #self.vis_processed_result = cv2.fillPoly(self.vis_processed_result, [contour], color=(255,200,255))

        contours,hierarchy = cv2.findContours(mask, 1, 2)
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(self.vis_processed_result,[box],0,(0,255,0),4)

        id_caption = "ID:" + str(index)
        caption_location = np.int0(np.sum(box, axis=0)/4)
        cv2.putText(self.vis_processed_result, id_caption, (caption_location[0]-20, caption_location[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA) 
            
        
    def check_object_size(self, attr):
        long_side = attr.long_sides[0] * 1000
        short_side = attr.short_sides[0] * 1000
        gt_size = self.object_attributes[self.target_object]

        is_long_size = False
        is_short_size = False
        if gt_size[0]*0.8 < long_side < gt_size[0]*1.1 or gt_size[1]*0.8 < long_side < gt_size[1]*1.1:
            is_long_size = True

        if gt_size[1]*0.8 < short_side < gt_size[1]*1.1 or gt_size[2]*0.8 < short_side < gt_size[2]*1.1:
            is_short_size = True
            
        return is_long_size and is_short_size
           
        
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
