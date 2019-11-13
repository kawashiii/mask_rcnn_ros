#!/usr/bin/env python3
import os
import sys
import time
import threading
import numpy as np

import rospy
import roslib.packages
#from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Point
from std_msgs.msg import UInt8MultiArray
from mask_rcnn_ros.msg import Result
from mask_rcnn_ros.srv import Detect, DetectResponse
#from cv_bridge import CvBridge

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import lab
import cv2
from math import atan2, cos, sin, sqrt, pi
from cv_bridge import CvBridge

from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

ROOT_DIR = os.path.abspath(roslib.packages.get_pkg_dir('mask_rcnn_ros'))
print(ROOT_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_lab.h5")
TEST_IMG = os.path.join(ROOT_DIR, "test.png")

CLASS_NAMES = lab.class_names

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
        self._model.keras_model._make_predict_function()

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

    def run3(self):
        self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
        self.vis_pub = rospy.Publisher('~visualization', Image, queue_size=1, latch=True)
        self.camera_matrix, self.dist_coeffs = getCalibrationMatrix()

        self.detect_srv = rospy.Service('mask_rcnn/detect_objects', Detect, self._get_frame)
        print("Ready to detect objects. Please service call /mask_rcnn/detect_objects")
        rospy.spin()

    def _get_frame(self, req):
        res = DetectResponse()
        res.message = "NG"
        res.success = False

        if (req.id == 1):
            print("Waiting frame...")
            image_msg = rospy.wait_for_message("/phoxi_camera/rgb_texture")
            depth_msg = rospy.wait_for_message("/phoxi_camera/depth_map")
            print("Acquired frame!")

            self._detect_objects(image_msg, depth_msg)

            res.message = "OK"
            res.success = True

        return res

    def _detect_objects(self, image_msg, depth_msg):
        np_image = self._cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        np_depth = self._cv_bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        
        # Run detection
        t1 = time.time()
        results = self._model.detect([np_image], verbose=0)
        t2 = time.time()            
        # Print detection time
        detection_time = t2 - t1
        print("Detection time: ", round(detection_time, 2), " s")

        result = results[0]
        result_msg = self._build_result_msg(image_msg, result, np_image, np_depth)
        self._result_pub.publish(result_msg)
        
        # Visualize results
        if self._visualization:
            vis_image = self._visualize(result, np_image)
            cv_result = np.zeros(shape=vis_image.shape, dtype=np.uint8)
            cv2.convertScaleAbs(vis_image, cv_result)
            image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
            self.vis_pub.publish(image_msg)
            print("Published detected image!")

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
        self.drawAxis(img, cntr, p1, (0, 255, 0), 1)
        self.drawAxis(img, cntr, p2, (255, 255, 0), 5)
        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
        
        return angle

    def getCalibrationMatrix(self):
        file_path = os.path.join(ROOT_DIR, "config/realsense_intrinsic.xml")
        fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode("camera_matrix").mat()
        self.dist_coeffs = fs.getNode("distortion_coefficients").mat()
        self.tvec = np.array([-0.221303, -0.259659, 0.854517], dtype=np.float32)
        self.rvec = np.array([[0.01391082878792294, 0.9998143798348266, 0.01333021822529676],[0.9998941380955942, -0.0138525840224244, -0.004451799408139552], [-0.00426631509639492, 0.01339073528237409, -0.9999012385051315]], dtype=np.float32)
        self.marker_origin = np.array([0.25, -0.58, -0.116], dtype=np.float32)

        return camera_matrix, dist_coeffs


    def _build_result_msg(self, msg, result, image, depth):
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

            m = result['masks'][:,:,i].astype(np.uint8)
            ret, thresh = cv2.threshold(m, 0.5, 1.0, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for j, c in enumerate(contours):
                area = cv2.contourArea(c)
                if area < 1e2 or 1e5 < area: continue

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
                center = Point()
                center.z = depth[cntr[1]][cntr[0]]
                center.x = undistorted_cntr[0]
                center.y = undistorted_cntr[1]
                result_msg.centers.append(center)

                np_x = np.array([eigenvectors[0,0] * eigenvalues[0,0], eigenvectors[0,1] * eigenvalues[0,0]], dtype=self.camera_matrix.dtype)
                np_x = np_x.reshape(-1, 1, 2)
                undistorted_x_axis = cv2.undistortPoints(np_x, self.camera_matrix, self.dist_coeffs)
                undistorted_x_axis = undistorted_x_axis.reshape(2)
                np_y = np.array([eigenvectors[1,0] * eigenvalues[1,0], eigenvectors[1,1] * eigenvalues[1,0]], dtype=self.camera_matrix.dtype)
                np_y = np_y.reshape(-1, 1, 2)
                undistorted_y_axis = cv2.undistortPoints(np_y, self.camera_matrix, self.dist_coeffs)
                undistorted_y_axis = undistorted_y_axis.reshape(2)
                x_axis = Vector3()
                x_axis.x = undistorted_x_axis[0]
                x_axis.y = undistorted_x_axis[1]
                x_axis.z = center.z
                y_axis = Vector3()
                y_axis.x = undistorted_y_axis[0]
                y_axis_y = undistorted_y_axis[1]
                y_axis_z = center.z
                result_msg.x_axis.append(x_axis)
                result_msg.y_axis.append(y_axis)

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
        # from matplotlib.backends.backend_agg import FigureCanvasAgg
        # from matplotlib.figure import Figure

        # fig = Figure()
        # canvas = FigureCanvasAgg(fig)
        # axes = fig.gca()
        # visualize.display_instances(image, result['rois'], result['masks'],
        #                             result['class_ids'], CLASS_NAMES,
        #                             result['scores'], ax = axes)
        # fig.tight_layout()
        # canvas.draw()
        # result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        #_, _, w, h = fig.bbox.bounds
        #result = result.reshape((int(h), int(w), 3))

        rois = result['rois']
        masks = result['masks']
        class_ids = result['class_ids']
        scores = result['scores']
        result = np.copy(image)

        for i in range(masks.shape[-1]):
            y1, x1, y2, x2 = rois[i]
            class_id = class_ids[i]
            score = scores[i]
            label = CLASS_NAMES[class_id]
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

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()

def main():
    rospy.init_node('mask_rcnn')

    node = MaskRCNNNode()
    # node.run()
    node.run3()

if __name__ == '__main__':
    main()
