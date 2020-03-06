#!/usr/bin/env python
"""""""""
Improve mask rcnn normal computation #31
use pcl function to compute the normal of the surface
"""""""""
# import open3d as o3d
# from open3d.geometry import OrientedBoundingBox
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
        self.pc = None

    def run(self):
        rospy.Subscriber("/phoxi_camera/pointcloud", PointCloud2, self.pointcloud_callback)
        self.result_srv = rospy.Service(rospy.get_name()+'/get_normal', GetNormal, self.get_normal)
        rospy.loginfo("Ready to be called service")
        rospy.spin()

    def pointcloud_callback(self, data):
        rospy.loginfo("Subscribed point cloud")
        pc = ros_numpy.numpify(data)
        r, c = pc.shape
        points = np.zeros((r, c, 3), dtype=np.float32)
        points[..., 0] = pc['x']
        points[..., 1] = pc['y']
        points[..., 2] = pc['z']
        p = o3d.geometry.PointCloud()
        #print(points.shape)
        #print(points.reshape(-1, 3).shape)
        p.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        self.pc = p.voxel_down_sample(voxel_size=0.002)
        self.estimate_normals()
        rospy.loginfo("Finished Down Sampling and Estimating Normals")
        #o3d.io.write_point_cloud("./test_o3d.ply", self.pc)
        
        # p = pcl.PointCloud(np.array(points.reshape(-1, 3), dtype=np.float32))
        # sor = p.make_voxel_grid_filter()
        # sor.set_leaf_size(0.01, 0.01, 0.01)
        # self.pc = sor.filter()

    def estimate_normals(self):
        param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.035, max_nn=30)
        self.pc.estimate_normals(search_param=param)
        # o3d.estimate_normals(self.pc, search_param=param)
        # feature = self.pc.make_NormalEstimation()
        # feature.set_KSearch(3)
        # normals = feature.compute()
    
    def get_normal(self, pts_msg):
        rospy.loginfo("Get Normal Service was called.")
        res = GetNormalResponse()

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
