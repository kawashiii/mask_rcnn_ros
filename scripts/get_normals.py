#!/usr/bin/env python2
"""""""""
Improve mask rcnn normal computation #31
use pcl function to compute the normal of the surface
"""""""""
# import open3d as o3d
# from open3d.geometry import OrientedBoundingBox
import numpy as np
import rospy
import pcl
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import tf
from mask_rcnn_ros.msg import MaskRCNNMsg
from mask_rcnn_ros.srv import MaskRCNNSrv, MaskRCNNSrvResponse


class GetNormals(object):
    def __init(self):
        self.pc = None
        self.center = None
        self.result_srv = None

    def callback(self, data):
        pc = ros_numpy.numpify(data)
        r, c = pc.shape
        points = np.zeros((r, c, 3), dtype=np.float32)
        points[..., 0] = pc['x']
        points[..., 1] = pc['y']
        points[..., 2] = pc['z']
        p = o3d.PointCloud()
        print(points.shape)
        print(points.reshape(-1, 3).shape)
        p.points = o3d.Vector3dVector(points.reshape(-1, 3))
        self.pc = o3d.voxel_down_sample(p, voxel_size=0.01)
        # p = pcl.PointCloud(np.array(points.reshape(-1, 3), dtype=np.float32))
        # sor = p.make_voxel_grid_filter()
        # sor.set_leaf_size(0.01, 0.01, 0.01)
        # self.pc = sor.filter()

    def run(self):
        self.callback_get_cloud()
        self.result_srv = rospy.Service(rospy.get_name()+'/MaskRCNNSrv', MaskRCNNSrv, self.get_centers)
        rospy.loginfo("Ready to be called service")
        rospy.spin()

    def get_centers(self):
        res = MaskRCNNSrvResponse()
        return res

    # cloud = rospy.wait_for_message("/phoxi_camera/pointcloud", Image, 10)
    def callback_get_cloud(self):
        rospy.Subscriber("/phoxi_camera/pointcloud", PointCloud2, self.callback)

    def get_normals(self):
        param = o3d.KDTreeSearchParamHybrid(radius=0.035, max_nn=30)
        open3d.estimate_normals(self.pc, search_param=param)
        # feature = self.pc.make_NormalEstimation()
        # feature.set_KSearch(3)
        # normals = feature.compute()
    
    # center: [x, y, z] from (1544, 2064, 3)
    def get_normal_around_center(self, center):
        from scipy.spatial import cKDTree

        normals = np.asarray(self.pc.normals) # (3186816, 3)
        tree = cKDTree(normals)
        sidx = tree.query(center, k=1)[1]
        return normals[sidx]


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
    rospy.init_node('get_normal')
    node = GetNormals()
    node.run()


if __name__ == "__main__":
	main()