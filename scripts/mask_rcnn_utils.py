#!/usr/bin/env python
"""""""""
Improve mask rcnn normal computation #31
use pcl function to compute the normal of the surface
"""""""""
# import open3d as o3d
# from open3d.geometry import OrientedBoundingBox
import time
import numpy as np
import rospy
#import pcl
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Vector3Stamped
import ros_numpy
import tf
from mask_rcnn_ros.srv import GetNormal, GetNormalResponse

class MaskRCNNUtils(object):
    def __init__(self):
        self.listener = tf.TransformListener()
        self.is_subscriber_called = False
        self.subscribed_time = 0

    def run(self):
        rospy.Subscriber("/phoxi_camera/pointcloud", PointCloud2, self.pointcloud_callback)
        self.result_srv = rospy.Service(rospy.get_name()+'/get_normal', GetNormal, self.get_normal)
        rospy.loginfo("Ready to be called service")
        rospy.spin()

    def pointcloud_callback(self, data):
        # Skip old msg
        now = rospy.get_time()
        if (now - data.header.stamp.secs > 60):
            self.is_subscriber_called = False
            rospy.logwarn("Skip old point cloud msg")
            return
        #if (now - self.subscribed_time < 15):
        #    rospy.logwarn("Skip continuous point cloud msg")
        #    return
        
        start_preprocess = time.time()
        rospy.loginfo("Subscribed point cloud")
        pc = ros_numpy.numpify(data)
        rc = pc.shape[0]
        points = np.zeros((rc, 3), dtype=np.float32)
        points[..., 0] = pc['x']
        points[..., 1] = pc['y']
        points[..., 2] = pc['z']
        #normals = np.zeros((rc, 3), dtype=np.float32)
        #normals[..., 0] = pc['normal_x']
        #normals[..., 1] = pc['normal_y']
        #normals[..., 2] = pc['normal_z']
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        #p.normals = o3d.utility.Vector3dVector(normals.reshape(-1, 3))
        self.pc = p        
        #self.pc = p.voxel_down_sample(voxel_size=0.002)
        #o3d.io.write_point_cloud("./test.ply", self.pc)
        #self.outlier_remove()
        #o3d.io.write_point_cloud("./test_outlier_removed.ply", self.pc)
        self.estimate_normals()
        end_preprocess = time.time()
        preprocess_time = end_preprocess - start_preprocess
        rospy.loginfo("%s[s] (Preprocess time)", round(preprocess_time, 3))
        self.subscribed_time = now
  
        self.is_subscriber_called = True
        
        # p = pcl.PointCloud(np.array(points.reshape(-1, 3), dtype=np.float32))
        # sor = p.make_voxel_grid_filter()
        # sor.set_leaf_size(0.01, 0.01, 0.01)
        # self.pc = sor.filter()

    def estimate_normals(self):
        param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=50)
        self.pc.estimate_normals(search_param=param)
        # o3d.estimate_normals(self.pc, search_param=param)
        # feature = self.pc.make_NormalEstimation()
        # feature.set_KSearch(3)
        # normals = feature.compute()

    def outlier_remove(self):
        cl, ind = self.pc.remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)
        self.pc = self.pc.select_down_sample(ind)    

    def get_normal(self, pts_msg):
        start_get_normal = time.time()
        rospy.loginfo("Get Normal Service was called.")
        res = GetNormalResponse()

        #rospy.loginfo("Waiting for preprocessing point cloud")
        #r = rospy.Rate(10)
        #while not rospy.is_shutdown():
        #    if self.is_subscriber_called:
        #        self.is_subscriber_called = False
        #        break
        #    r.sleep()
        
        rospy.loginfo("Extracting normal around the center")
        for pt in pts_msg.centers:
            center_stamped = self.listener.transformPoint("PhoXi3Dscanner_sensor", pt)
            np_center = np.array([center_stamped.point.x, center_stamped.point.y, center_stamped.point.z])
            
            np_center, np_normal = self.get_normal_around_center(np_center)
            normal = Vector3(np_normal[0], np_normal[1], np_normal[2])
            normal_stamped = Vector3Stamped()
            normal_stamped.header.frame_id = "PhoXi3Dscanner_sensor"
            normal_stamped.header.stamp = rospy.Time.now()
            if (normal.z > 0 ): 
                normal.x *= -1
                normal.y *= -1
                normal.z *= -1
            normal_stamped.vector = normal
            center = Point(np_center[0], np_center[1], np_center[2])
            center_stamped.point = center

            res.centers.append(center_stamped)
            res.normals.append(normal_stamped)
            
        end_get_normal = time.time()
        get_normal_time = end_get_normal - start_get_normal
        rospy.loginfo("%s[s] (Service 'get_normal' time)", round(get_normal_time, 3))
        print("")
        return res

    # center: [x, y, z] from (1544, 2064, 3)
    def get_normal_around_center(self, center):
        from scipy.spatial import cKDTree

        pcd = np.asarray(self.pc.points) # (3186816, 3)
        normals = np.asarray(self.pc.normals)
	tree = cKDTree(pcd)
        sidx = tree.query(center, k=1)[1]
        return pcd[sidx], normals[sidx]


    # https://github.com/strawlab/python-pcl/blob/1d83d2d7ce9ce2c22ff5855249459bfc22025000/examples/official/Features/moment_of_inertia.py
    def get_normals2cloud(cloud):
        # cloud = pcl.load('./examples/pcldata/tutorials/lamppost.pcd')
        feature_extractor = cloud.make_MomentOfInertiaEstimation()
        feature_extractor.compute()
        # moment_of_inertia = feature_extractor.get_MomentOfInertia()
        # eccentricity = feature_extractor.get_Eccentricity()
        # [min_point_AABB, max_point_AABB] = feature_extractor.get_AABB()
        # [min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB] = feature_extractor.get_OBB()
        [major_value, middle_value, minor_value] = feature_extractor.get_EigenValues()
        [major_vector, middle_vector, minor_vector] = feature_extractor.get_EigenVectors()
        mass_center = feature_extractor.get_MassCenter()

        center = pcl.PointCloud(mass_center[0], mass_center[1], mass_center[2])
        x_axis = pcl.PointCloud(
		    major_vector[0]+mass_center[0], 
		    major_vector[1]+mass_center[1], 
		    major_vector[2]+mass_center[2]
	    )
        y_axis = pcl.PointCloud(
		    middle_vector[0]+mass_center[0],
            middle_vector[1]+mass_center[1], 
		    middle_vector[2]+mass_center[2]
	    )
        z_axis = pcl.PointCloud(
            minor_vector[0]+mass_center[0], 
		    minor_vector[1]+mass_center[1], 
		    minor_vector[2]+mass_center[2]
	    )
        return center, x_axis, y_axis, z_axis


def main():
    rospy.init_node('mask_rcnn_utils')
    node = MaskRCNNUtils()
    node.run()


if __name__ == "__main__":
    main()
