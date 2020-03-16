#!/usr/bin/env python3
import sys
import time
from copy import deepcopy
import rospy
import numpy as np
from scipy.spatial import cKDTree
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Image
from mask_rcnn_ros.msg import MaskRCNNMsg, ORDEREDMsg
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from cv_bridge import CvBridge


class ORDERED_ImageDiff(object):
    def __init__(self):
        self._cv_bridge = CvBridge()
        self._fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        # self._fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        # self._fgbg = cv2.createBackgroundSubtractorMOG()
        self.uids = None
        self.ucenters = None
        self.unormals = None

    def run(self):
        self._res_pub = rospy.Publisher(rospy.get_name()+'/ORDEREDMsg', ORDEREDMsg, queue_size=1, latch=True)
        # rospy.Subscriber("/pylon_camera_node/image_raw", Image, self._call_img)
        # rospy.Subscriber("/phoxi_camera/external_camera_texture", Image, self._call_img)
        rospy.Subscriber("/mask_rcnn/MaskRCNNMsg", MaskRCNNMsg, self._call_get_maskmsg)
        self.marker_pub = rospy.Publisher(rospy.get_name() + '/axes_ordered', MarkerArray, queue_size=1, latch=True)
        rospy.loginfo("Ready to be called service")
        rospy.spin()

#    def _call_img(self, img_msg):
#        img = self._get_img(img_msg)
#        fgm = self._fgbg.apply(img)
#        self.fgmask = cv2.threshold(fgm, 200, 255, cv2.THRESH_BINARY)
#        self.img = img
    
    def _call_get_maskmsg(self, msg):
        # Skip old msg
        now = rospy.get_time()
        if (now - msg.header.stamp.secs > 30):
            self.is_subscriber_called = False
            rospy.logwarn("Skip old msg")
            return
        rospy.loginfo("Subscribed Mask R-CNN message")
        s_proc = time.time()
        img_msg = rospy.wait_for_message("/pylon_camera_node/image_raw", Image, 10)
        img = self._get_img(img_msg)
        fgm = self._fgbg.apply(img)
        self.fgmask = np.where(fgm==0, 0, 1)

        ids = np.array(msg.ids, dtype=np.uint32)
        boxes = [[box.x_offset, box.y_offset, box.width, box.height] for box in msg.boxes]
        b = self._assigned_uid2box(msg, ids, boxes)
        e_proc = time.time()
        proc_t = e_proc - s_proc
        rospy.loginfo("%s[s] (Preprocess time)", round(proc_t, 3))
        self._build_res_msg(msg, b)

    def _get_img(self, img_msg):
        img = self._cv_bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _check_move(self, boxes):
        v = np.array(
            [self.fgmask[b[0]:b[0]+b[2], b[1]:b[1]+b[3]].sum() / (b[2] * b[3]) for b in boxes]
        )
        print([self.fgmask[b[0]:b[0]+b[2], b[1]:b[1]+b[3]].sum() for b in boxes])
        print(v)
        return (v>0.6).any()
 #       val = self.fgmask.sum() / self.fgmask.size
 #       print(self.fgmask.sum())
 #       print(self.fgmask.size)
 #       print(val)
 #       return val > 0.6

    def _assigned_uid2box(self, msg, ids, boxes, b=0):
        ids = np.asarray(ids)
        boxes = np.asarray(boxes)
        self._tree = cKDTree(boxes[:, :2])
        if self.uids is None or self._check_move(boxes):
            # Reset when increasing in number
            print('Reset because of First or a lot of moving')
            self.uids = ids
            self.ucenters = msg.centers
            self.unormals = msg.normals
            self._boxes = boxes
            self.uid_max = -1
            b = 1
        elif len(boxes) > len(self._boxes):
            np_center = np.array([[pt.point.x, pt.point.y] for pt in msg.centers])
            np_ucenter = np.array([[pt.point.x, pt.point.y] for pt in self.ucenters])
            tree = cKDTree(np_ucenter)
            d = tree.query(np_center, p=1)[0]
            n = len(boxes) - len(self._boxes)
            sidx = np.argpartition(d, -n)[-n:]
            self.uid_max = max(self.uids.max(), self.uid_max) + 1
            self.uids = np.r_[self.uids, np.arange(self.uid_max, self.uid_max+len(sidx))]
            for i, (u, c) in enumerate(zip(msg.centers, msg.normals)):
                if i in sidx.tolist():
                    self.ucenters.append(u)
                    self.unormals.append(c)
            self._boxes = boxes
            print('N: ', n)
            print('box: ', len(boxes))
            print('self._boxes:', len(self._boxes))
            print('distance: ', d)
            print('sidx:', sidx)
            print('max value:', self.uid_max)
            print('UID: ', self.uids)
#            # Reset when increasing in number
#            print('Reset when increasing in number')
#            self.uids = ids
#            self.ucenters = msg.centers
#            self.unormals = msg.normals
#            self._boxes = boxes
#            b = True
        else:
            print('query: ', self._boxes[:, :2])
            d, sidx = self._tree.query(self._boxes[:, :2], p=1, distance_upper_bound=50)
            cond = np.isinf(d)
            print('d:', d)
            print('cond:', cond)
            print('sidx:', sidx)
#            if cond.sum() >= 2:
#                print('Judge as moved and reset')
#                self.uids = ids
#                self.ucenters = msg.centers
#                self.unormals = msg.normals
#                self._boxes = boxes
#                b = True
#            else:
            try:
                # maintaining the order even if decreasing in number by picking
                self.uids = self.uids[~cond]
                self._boxes = self._boxes[~cond]
                didx = cond.nonzero()[0]
                ucenters = []
                unormals = []
                for i, (c, n) in enumerate(zip(self.ucenters, self.unormals)):
                    if i in didx: continue
                    ucenters.append(c)
                    unormals.append(n)
                self.ucenters = ucenters
                self.unormals = unormals
            except Exception as e:
                # In case of unexpected error, Reset
                print('Unexpected error:', e.args)
                pass
#                    # print(sys.exc_info())
#                    self.uids = ids
#                    self.ucenters = msg.centers
#                    self.unormals = msg.normals
#                    self._boxes = boxes
#                    b = True
        
        print('boxes: ', boxes[:, :2])
        print('ids: ', ids)
        print('uids: ', self.uids)
        return b

    def _build_res_msg(self, msg, b):
        text_msgs = MarkerArray()
        rospy.loginfo("Building msg ...")
        self.res_msg = ORDEREDMsg()
        self.res_msg.header = msg.header
        self.res_msg.count = msg.count
        self.res_msg.ids = msg.ids
        self.res_msg.areas = msg.areas
        self.res_msg.axes = msg.axes
        self.res_msg.boxes = msg.boxes
        self.res_msg.class_ids = msg.class_ids
        self.res_msg.class_names = msg.class_names
        self.res_msg.scores = msg.scores
        self.res_msg.masks = msg.masks
        self.res_msg.centers = msg.centers
        self.res_msg.normals = msg.normals
        self.res_msg.reset_id = b

        self.res_msg.uids = self.uids
        print('UID:', self.uids)
        print("centers and normals count", len(msg.centers), len(msg.normals))
        print("ucenters and unormals count", len(self.ucenters), len(self.unormals))

        delete_marker = self.delete_all_markers()
        text_msgs.markers.append(delete_marker)
        for i, c, n in zip(self.uids, self.ucenters, self.unormals):
            text_marker = self.build_marker_msg(c.header.frame_id, Marker.TEXT_VIEW_FACING, i, c.point, n.vector, 1.0, 0.0, 0.0, "id_text")
            text_msgs.markers.append(text_marker)

        self.marker_pub.publish(text_msgs)
        self._res_pub.publish(self.res_msg)
    
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
            axis_marker.pose.position.x = center.x + 0.01 + axis.x * 0.05
            axis_marker.pose.position.y = center.y + 0.01 + axis.y * 0.05
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


def main():
    rospy.init_node('ordered')
    node = ORDERED_ImageDiff()
    node.run()


if __name__ == "__main__":
    main()
