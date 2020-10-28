#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <math.h>

//ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/PolygonStamped.h>

//PCL
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/surface/convex_hull.h>

//OpenCv
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//Msg or Srv
#include "mask_rcnn_ros/GetMaskedSurface.h"
//#include "mask_rcnn_ros/Centers.h"
//#include "mask_rcnn_ros/Normals.h"
//#include "mask_rcnn_ros/Areas.h"
#include "mask_rcnn_ros/MaskedObjectAttributes.h"

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979
#endif
#define deg_to_rad(deg) (((deg)/360)*2*M_PI)
#define rad_to_deg(rad) (((rad)/2/M_PI)*360)
sensor_msgs::CameraInfoConstPtr camera_info;
cv::Mat cameraMatrix;
cv::Mat distCoeffs;
cv::Mat depth;
vector<sensor_msgs::Image> mask_msgs;
double fx;
double fy;
double cx;
double cy;

typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointColorT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointColorT> PointCloudColorT;
PointCloudT::Ptr scene(new PointCloudT);
PointCloudT::Ptr input_scene(new PointCloudT);
pcl::RegionGrowing<PointT, pcl::Normal> scene_reg;
pcl::PointCloud <pcl::Normal>::Ptr scene_normals (new pcl::PointCloud <pcl::Normal>);
vector<PointCloudColorT::Ptr> scene_surface_list;
vector<PointCloudColorT::Ptr> masked_surface_list;

ros::Publisher vis_axes_marker;
ros::Publisher masked_surface_pointcloud_pub;
ros::Publisher scene_surface_pointcloud_pub;
ros::Publisher masked_depth_map_pub;
ros::Publisher input_pointcloud_pub;
std::string camera_info_topic = "/pylon_camera_node/camera_info";
std::string depth_topic = "/phoxi_camera/aligned_depth_map";
std::string debug_depth_topic = "/debug/depth_rect";
std::string frame_id = "basler_ace_rgb_sensor_calibrated";
//vector<mask_rcnn_ros::Centers> center_msg_list;
//vector<mask_rcnn_ros::Normals> normal_msg_list;
//vector<mask_rcnn_ros::Areas> area_msg_list;
//vector<geometry_msgs::PointStamped> corner_msg_list;
vector<mask_rcnn_ros::MaskedObjectAttributes> moas_msg_list;
visualization_msgs::MarkerArray marker_axes_list;
geometry_msgs::Vector3 arrow_scale;
std_msgs::ColorRGBA x_axis_color;
std_msgs::ColorRGBA y_axis_color;
std_msgs::ColorRGBA z_axis_color;

float is_service_called = false;
int debug_mode = 0;

//struct CenterNormalMsg{
//    geometry_msgs::PointStamped center;
//    geometry_msgs::Vector3Stamped normal;
//    float area;
//};

struct MomentOfInertia{
   vector<Eigen::Vector3f> vectors;
   PointT min_point_OBB;
   PointT max_point_OBB;      
};

bool compare_indices_size(const pcl::PointIndices& lp, const pcl::PointIndices& rp)
{
    return lp.indices.size() > rp.indices.size();
}

visualization_msgs::Marker build_marker_msg(std_msgs::Header header, string ns, int id, int type, int action, geometry_msgs::Vector3 scale, std_msgs::ColorRGBA color, geometry_msgs::Point point, geometry_msgs::Vector3 vector, string text)
{
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = id;
    marker.type = type;
    marker.action = action;
    marker.scale = scale;
    marker.color = color;
    if (type == visualization_msgs::Marker::TEXT_VIEW_FACING)
    {
        marker.pose.position.x = point.x + vector.x * 0.04;
        marker.pose.position.y = point.y + vector.y * 0.04;
        marker.pose.position.z = point.z + vector.z * 0.04 + 0.01;
        marker.text = text;

        return marker;
    }
    if (type == visualization_msgs::Marker::SPHERE)
    {
	marker.pose.position.x = point.x;
	marker.pose.position.y = point.y;
	marker.pose.position.z = point.z;
        marker.text = text;

	return marker;
    }
    marker.points.resize(2);
    marker.points[0] = point;
    marker.points[1].x = point.x + vector.x * 0.04;
    marker.points[1].y = point.y + vector.y * 0.04;
    marker.points[1].z = point.z + vector.z * 0.04;
    marker.text = text;

    return marker;
}

float getArea(PointCloudT::Ptr cloud_in)
{
    PointCloudT::Ptr cloud_out(new PointCloudT);

    pcl::ConvexHull<PointT> chull;
    chull.setInputCloud(cloud_in);
    chull.setComputeAreaVolume(true);
    chull.reconstruct(*cloud_out);

    return chull.getTotalArea();
}

MomentOfInertia getMomentOfInertia(PointCloudT::Ptr cloud_in)
{
    PointT min_point_AABB;
    PointT max_point_AABB;
    PointT min_point_OBB;
    PointT max_point_OBB;
    PointT position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    Eigen::Vector3f major_vector, middle_vector, minor_vector;

    pcl::MomentOfInertiaEstimation <PointT> feature_extractor;
    feature_extractor.setInputCloud (cloud_in);
    feature_extractor.compute ();
    feature_extractor.getAABB (min_point_AABB, max_point_AABB);
    feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);

    MomentOfInertia moi;
    vector<Eigen::Vector3f> ret_vectors{major_vector, middle_vector, minor_vector};
    moi.vectors = ret_vectors;
    moi.min_point_OBB = min_point_OBB;
    moi.max_point_OBB = max_point_OBB;

    return moi;

}

void sceneRegionGrowingFromPoint(vector<PointT> center_list, vector<PointCloudT::Ptr> cloud_list)
{
    //mask_rcnn_ros::Centers centers_msg;
    //mask_rcnn_ros::Normals normals_msg;
    //mask_rcnn_ros::Areas areas_msg;
    mask_rcnn_ros::MaskedObjectAttributes moas_msg;
    //for (PointT center : center_list)
    for (int i = 0; i < center_list.size(); i++)
    {
        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud(scene);
        int K = 1;
        vector<int> indices_(K);
        vector<float> distance(K);
        kdtree.nearestKSearch(center_list[i], K, indices_, distance);

        pcl::PointIndices cluster;
        scene_reg.getSegmentFromPoint(indices_[0], cluster);

        ROS_INFO("Global Surface point cloud size: %d", (int)cluster.indices.size()); 

        pcl::CentroidPoint<PointT> centroid;
        PointT c1;
        for (auto point = cluster.indices.begin(); point != cluster.indices.end(); point++)
        {
            centroid.add(scene->points[*point]);
        }
        centroid.get(c1);
        if (c1.z == 0.0) {
            ROS_WARN("Center was zero");
            //continue;
        }

        kdtree.setInputCloud(scene);
        kdtree.nearestKSearch(c1, K, indices_, distance);

        geometry_msgs::PointStamped center;
        geometry_msgs::Vector3Stamped normal;
        center.header.frame_id = frame_id;
        center.header.stamp = ros::Time::now();
        center.point.x = scene->points[indices_[0]].x;
        center.point.y = scene->points[indices_[0]].y;
        center.point.z = scene->points[indices_[0]].z;
        normal.header.frame_id = frame_id;
        normal.header.stamp = ros::Time::now();
        normal.vector.x = scene_normals->points[indices_[0]].normal_x;
        normal.vector.y = scene_normals->points[indices_[0]].normal_y;
        normal.vector.z = scene_normals->points[indices_[0]].normal_z;
        if (normal.vector.z > 0) {
            normal.vector.x *= -1;
            normal.vector.y *= -1;
            normal.vector.z *= -1;
        }

        float area = getArea(cloud_list[i]);

        MomentOfInertia moi = getMomentOfInertia(cloud_list[i]);
	vector<Eigen::Vector3f> axes = moi.vectors;
        geometry_msgs::Vector3Stamped x_axis;
        geometry_msgs::Vector3Stamped y_axis;
        geometry_msgs::Vector3Stamped z_axis;
        x_axis.header.frame_id = frame_id;
        x_axis.header.stamp = ros::Time::now();
        x_axis.vector.x = axes[0](0);
        x_axis.vector.y = axes[0](1);
        x_axis.vector.z = axes[0](2);
        y_axis.header.frame_id = frame_id;
        y_axis.header.stamp = ros::Time::now();
        y_axis.vector.x = axes[1](0);
        y_axis.vector.y = axes[1](1);
        y_axis.vector.z = axes[1](2);
        z_axis.header.frame_id = frame_id;
        z_axis.header.stamp = ros::Time::now();
        z_axis.vector.x = axes[2](0);
        z_axis.vector.y = axes[2](1);
        z_axis.vector.z = axes[2](2);
        if (z_axis.vector.z > 0) {
            z_axis.vector.x *= -1;
            z_axis.vector.y *= -1;
            z_axis.vector.z *= -1;
        }
        geometry_msgs::PolygonStamped corner;
        corner.header.frame_id = frame_id;
        corner.header.stamp = ros::Time::now();
        vector<geometry_msgs::Point32> corner_points;
        geometry_msgs::Point32 tmp;
        tmp.x = center.point.x + x_axis.vector.x * moi.min_point_OBB.x + y_axis.vector.x * moi.min_point_OBB.y;
        tmp.y = center.point.y + x_axis.vector.y * moi.min_point_OBB.x + y_axis.vector.y * moi.min_point_OBB.y;
        tmp.z = center.point.z + x_axis.vector.z * moi.min_point_OBB.x + y_axis.vector.z * moi.min_point_OBB.y;
        corner.polygon.points.push_back(tmp);
        tmp.x = center.point.x + x_axis.vector.x * moi.max_point_OBB.x + y_axis.vector.x * moi.min_point_OBB.y;
        tmp.y = center.point.y + x_axis.vector.y * moi.max_point_OBB.x + y_axis.vector.y * moi.min_point_OBB.y;
        tmp.z = center.point.z + x_axis.vector.z * moi.max_point_OBB.x + y_axis.vector.z * moi.min_point_OBB.y;
        corner.polygon.points.push_back(tmp);
        tmp.x = center.point.x + x_axis.vector.x * moi.min_point_OBB.x + y_axis.vector.x * moi.max_point_OBB.y;
        tmp.y = center.point.y + x_axis.vector.y * moi.min_point_OBB.x + y_axis.vector.y * moi.max_point_OBB.y;
        tmp.z = center.point.z + x_axis.vector.z * moi.min_point_OBB.x + y_axis.vector.z * moi.max_point_OBB.y;
        corner.polygon.points.push_back(tmp);
        tmp.x = center.point.x + x_axis.vector.x * moi.max_point_OBB.x + y_axis.vector.x * moi.max_point_OBB.y;
        tmp.y = center.point.y + x_axis.vector.y * moi.max_point_OBB.x + y_axis.vector.y * moi.max_point_OBB.y;
        tmp.z = center.point.z + x_axis.vector.z * moi.max_point_OBB.x + y_axis.vector.z * moi.max_point_OBB.y;
        corner.polygon.points.push_back(tmp);
	
        moas_msg.centers.push_back(center);
	moas_msg.normals.push_back(normal);
	moas_msg.areas.push_back(area);
        moas_msg.corners.push_back(corner);
	moas_msg.x_axes.push_back(x_axis);
	moas_msg.y_axes.push_back(y_axis);
	moas_msg.z_axes.push_back(z_axis);

        //CenterNormalMsg msg;
        //msg.center.header.frame_id = frame_id;
        //msg.center.header.stamp = ros::Time::now();
        //msg.center.point.x = scene->points[indices_[0]].x;
        //msg.center.point.y = scene->points[indices_[0]].y;
        //msg.center.point.z = scene->points[indices_[0]].z;
        //msg.normal.header.frame_id = frame_id;
        //msg.normal.header.stamp = ros::Time::now();
        //msg.normal.vector.x = scene_normals->points[indices_[0]].normal_x;
        //msg.normal.vector.y = scene_normals->points[indices_[0]].normal_y;
        //msg.normal.vector.z = scene_normals->points[indices_[0]].normal_z;
        //if (msg.normal.vector.z > 0) {
            //msg.normal.vector.x *= -1;
            //msg.normal.vector.y *= -1;
            //msg.normal.vector.z *= -1;
        //}
        //msg.area = area;

        //centers_msg.centers.push_back(msg.center);
        //normals_msg.normals.push_back(msg.normal);
        //areas_msg.areas.push_back(msg.area);
    }

    if (moas_msg.centers.size() == 1)
    {
	moas_msg_list.push_back(moas_msg);
    }
    else
    {
	float angle = 180.0;
	mask_rcnn_ros::MaskedObjectAttributes best_moas_msg;
	for (int i = 0; i < moas_msg.centers.size(); i++)
	{
	    geometry_msgs::PointStamped center = moas_msg.centers[i];
	    geometry_msgs::Vector3Stamped normal = moas_msg.normals[i];
	    // (a1b1 + a2b2 + a3b3)/sqrt(a1^2 + a2^2 + a3^2) * sqrt(b1^2 + b2^2 + b3^2)
	    // a = (0, 0, 1), b = normal
	    float cos_theta = -1 * normal.vector.z/sqrt(pow(normal.vector.x, 2.0) + pow(normal.vector.y, 2.0) + pow(normal.vector.z, 2.0));
	    float theta = rad_to_deg(acos(cos_theta));
	    //std::cout << theta << std::endl;

	    if (fabsf(theta) < angle) {
		best_moas_msg = mask_rcnn_ros::MaskedObjectAttributes();
	        best_moas_msg.centers.push_back(center);
	        best_moas_msg.normals.push_back(normal);
	        best_moas_msg.areas.push_back(moas_msg.areas[i]);
	        best_moas_msg.corners.push_back(moas_msg.corners[i]);
	        best_moas_msg.x_axes.push_back(moas_msg.x_axes[i]);
	        best_moas_msg.y_axes.push_back(moas_msg.y_axes[i]);
	        best_moas_msg.z_axes.push_back(moas_msg.z_axes[i]);

		angle = fabsf(theta);
	    }
	}
	moas_msg_list.push_back(best_moas_msg);
    }

    //if (centers_msg.centers.size() == 0) {
    //    ROS_WARN("Error region growing");
    //    return;
    //}
    //else if (centers_msg.centers.size() == 1)
    //if (centers_msg.centers.size() == 1)
    //{    
        //center_msg_list.push_back(centers_msg);
        //normal_msg_list.push_back(normals_msg);
        //area_msg_list.push_back(areas_msg);
    //}
    //else 
    //{
        //float angle = 180.0;
        //mask_rcnn_ros::Centers best_centers_msg;
        //mask_rcnn_ros::Normals best_normals_msg;
        //mask_rcnn_ros::Areas best_areas_msg;
        //for (int i = 0; i < centers_msg.centers.size(); i++) {
            //geometry_msgs::PointStamped center = centers_msg.centers[i];
            //geometry_msgs::Vector3Stamped normal = normals_msg.normals[i];
            //// (a1b1 + a2b2 + a3b3)/sqrt(a1^2 + a2^2 + a3^2) * sqrt(b1^2 + b2^2 + b3^2)
            //// a = (0, 0, 1), b = normal
            //float cos_theta = -1 * normal.vector.z/sqrt(pow(normal.vector.x, 2.0) + pow(normal.vector.y, 2.0) + pow(normal.vector.z, 2.0));
            //float theta = rad_to_deg(acos(cos_theta));
            ////std::cout << theta << std::endl;

            //if (fabsf(theta) < angle) {
                //best_centers_msg.centers = {};
                //best_normals_msg.normals = {};
                //best_centers_msg.centers.push_back(center);
                //best_normals_msg.normals.push_back(normal);
                //best_areas_msg.areas.push_back(areas_msg.areas[i]);
                //angle = fabsf(theta);
            //}
        //}
        //center_msg_list.push_back(best_centers_msg);
        //normal_msg_list.push_back(best_normals_msg);
        //area_msg_list.push_back(best_areas_msg);
    //}
}

void maskedRegionGrowing(cv::Mat mask)
{
    cv::Mat mask_index;
    cv::findNonZero(mask, mask_index);

    PointCloudT::Ptr cloud (new PointCloudT);
    vector<float> z_values;
    for (int i = 0; i < mask_index.total(); i++)
    {
        int u = mask_index.at<cv::Point>(i).x;
	int v = mask_index.at<cv::Point>(i).y;
	PointT p;
	p.z = depth.at<float>(v, u) / 1000;
	p.x = (u - cx) * p.z / fx;
	p.y = (v - cy) * p.z / fy;
	z_values.push_back(p.z);
	cloud->points.push_back(p);
    }

    //compute rotated corner of mask
    //std::sort(z_values.begin(), z_values.end());
    //int vector_size = (int)(z_values.size() / 2);
    //vector<vector<cv::Point> > contours;
    //vector<cv::Vec4i> hierarchy;
    //cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    //cv::RotatedRect rect = cv::minAreaRect(contours[0]);
    //cv::Point2f vertices[4];
    //rect.points(vertices);

    //downsampling
    pcl::VoxelGrid<PointT> voxelSampler;
    voxelSampler.setInputCloud(cloud);
    voxelSampler.setLeafSize(0.002, 0.002, 0.002);
    voxelSampler.filter(*cloud);

    //outlier remover
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud);

    //regiongrowing
    pcl::search::Search<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::PointCloud <pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
    pcl::RegionGrowing<PointT, pcl::Normal> reg;
    vector<pcl::PointIndices> indices;
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (cloud);
    normal_estimator.setKSearch (30);
    //normal_estimator.setRadiusSearch(0.02);
    normal_estimator.compute (*cloud_normals);
    reg.setMinClusterSize (100);
    reg.setMaxClusterSize (7000);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours (30);
    reg.setInputCloud (cloud);
    reg.setInputNormals (cloud_normals);
    reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold (1.0);
    reg.extract(indices);

    if (indices.size() == 0) {
        ROS_WARN("Couldn't find any surface");
        mask_rcnn_ros::MaskedObjectAttributes moas_tmp_msg;
        geometry_msgs::PointStamped center_tmp;
        geometry_msgs::Vector3Stamped normal_tmp;
        geometry_msgs::Vector3Stamped x_axis_tmp;
        geometry_msgs::Vector3Stamped y_axis_tmp;
        geometry_msgs::Vector3Stamped z_axis_tmp;
        geometry_msgs::PolygonStamped corner_tmp;
        float area_tmp = 0;

	moas_tmp_msg.centers.push_back(center_tmp);
	moas_tmp_msg.normals.push_back(normal_tmp);
	moas_tmp_msg.areas.push_back(area_tmp);
        moas_tmp_msg.corners.push_back(corner_tmp);
	moas_tmp_msg.x_axes.push_back(x_axis_tmp);
	moas_tmp_msg.y_axes.push_back(y_axis_tmp);
	moas_tmp_msg.z_axes.push_back(z_axis_tmp);

        moas_msg_list.push_back(moas_tmp_msg);
        return;
    }

    ROS_INFO("%d surfaces found", (int)indices.size());
    masked_surface_list.push_back(reg.getColoredCloud());
    vector<PointT> center_list;
    vector<PointCloudT::Ptr> cloud_list;
    for (auto i = indices.begin(); i != indices.end(); i++)
    {
        pcl::CentroidPoint<PointT> centroid;
        PointCloudT::Ptr tmp(new PointCloudT);
        PointT center;
        ROS_INFO("Local Surface point cloud size: %d", (int)i->indices.size()); 
        for (auto point = i->indices.begin(); point != i->indices.end(); point++)
        {
            centroid.add(cloud->points[*point]);
            tmp->points.push_back(cloud->points[*point]);
        }
        centroid.get(center);
        center_list.push_back(center);
        cloud_list.push_back(tmp);
    }

    //mask_rcnn_ros::Centers centers_msg;
    //mask_rcnn_ros::Normals normals_msg;
    //mask_rcnn_ros::Areas areas_msg;
    mask_rcnn_ros::MaskedObjectAttributes moas_msg;
    if (center_list.size() > 1) {
        sceneRegionGrowingFromPoint(center_list, cloud_list);
    } else {
        PointT c1 = center_list[0];
        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud(cloud);
        int K = 1;
        vector<int> indices_(K);
        vector<float> distance(K);
        kdtree.nearestKSearch(c1, K, indices_, distance);

        //CenterNormalMsg msg;
        //msg.center.header.frame_id = frame_id;
        //msg.center.header.stamp = ros::Time::now();
        //msg.center.point.x = cloud->points[indices_[0]].x;
        //msg.center.point.y = cloud->points[indices_[0]].y;
        //msg.center.point.z = cloud->points[indices_[0]].z;
        //msg.normal.header.frame_id = frame_id;
        //msg.normal.header.stamp = ros::Time::now();
        //msg.normal.vector.x = cloud_normals->points[indices_[0]].normal_x;
        //msg.normal.vector.y = cloud_normals->points[indices_[0]].normal_y;
        //msg.normal.vector.z = cloud_normals->points[indices_[0]].normal_z;
        //if (msg.normal.vector.z > 0) {
            //msg.normal.vector.x *= -1;
            //msg.normal.vector.y *= -1;
            //msg.normal.vector.z *= -1;
        //}

        geometry_msgs::PointStamped center;
        geometry_msgs::Vector3Stamped normal;
        center.header.frame_id = frame_id;
        center.header.stamp = ros::Time::now();
        center.point.x = cloud->points[indices_[0]].x;
        center.point.y = cloud->points[indices_[0]].y;
        center.point.z = cloud->points[indices_[0]].z;
        normal.header.frame_id = frame_id;
        normal.header.stamp = ros::Time::now();
        normal.vector.x = cloud_normals->points[indices_[0]].normal_x;
        normal.vector.y = cloud_normals->points[indices_[0]].normal_y;
        normal.vector.z = cloud_normals->points[indices_[0]].normal_z;
        if (normal.vector.z > 0) {
            normal.vector.x *= -1;
            normal.vector.y *= -1;
            normal.vector.z *= -1;
        }

        float area = getArea(cloud);

        MomentOfInertia moi = getMomentOfInertia(cloud);
	vector<Eigen::Vector3f> axes = moi.vectors;
        geometry_msgs::Vector3Stamped x_axis;
        geometry_msgs::Vector3Stamped y_axis;
        geometry_msgs::Vector3Stamped z_axis;
        x_axis.header.frame_id = frame_id;
        x_axis.header.stamp = ros::Time::now();
        x_axis.vector.x = axes[0](0);
        x_axis.vector.y = axes[0](1);
        x_axis.vector.z = axes[0](2);
        y_axis.header.frame_id = frame_id;
        y_axis.header.stamp = ros::Time::now();
        y_axis.vector.x = axes[1](0);
        y_axis.vector.y = axes[1](1);
        y_axis.vector.z = axes[1](2);
        z_axis.header.frame_id = frame_id;
        z_axis.header.stamp = ros::Time::now();
        z_axis.vector.x = axes[2](0);
        z_axis.vector.y = axes[2](1);
        z_axis.vector.z = axes[2](2);
        if (z_axis.vector.z > 0) {
            z_axis.vector.x *= -1;
            z_axis.vector.y *= -1;
            z_axis.vector.z *= -1;
        }
        geometry_msgs::PolygonStamped corner;
        corner.header.frame_id = frame_id;
        corner.header.stamp = ros::Time::now();
        vector<geometry_msgs::Point32> corner_points;
        geometry_msgs::Point32 tmp;
        tmp.x = center.point.x + x_axis.vector.x * moi.min_point_OBB.x + y_axis.vector.x * moi.min_point_OBB.y;
        tmp.y = center.point.y + x_axis.vector.y * moi.min_point_OBB.x + y_axis.vector.y * moi.min_point_OBB.y;
        tmp.z = center.point.z + x_axis.vector.z * moi.min_point_OBB.x + y_axis.vector.z * moi.min_point_OBB.y;
        corner.polygon.points.push_back(tmp);
        tmp.x = center.point.x + x_axis.vector.x * moi.max_point_OBB.x + y_axis.vector.x * moi.min_point_OBB.y;
        tmp.y = center.point.y + x_axis.vector.y * moi.max_point_OBB.x + y_axis.vector.y * moi.min_point_OBB.y;
        tmp.z = center.point.z + x_axis.vector.z * moi.max_point_OBB.x + y_axis.vector.z * moi.min_point_OBB.y;
        corner.polygon.points.push_back(tmp);
        tmp.x = center.point.x + x_axis.vector.x * moi.min_point_OBB.x + y_axis.vector.x * moi.max_point_OBB.y;
        tmp.y = center.point.y + x_axis.vector.y * moi.min_point_OBB.x + y_axis.vector.y * moi.max_point_OBB.y;
        tmp.z = center.point.z + x_axis.vector.z * moi.min_point_OBB.x + y_axis.vector.z * moi.max_point_OBB.y;
        corner.polygon.points.push_back(tmp);
        tmp.x = center.point.x + x_axis.vector.x * moi.max_point_OBB.x + y_axis.vector.x * moi.max_point_OBB.y;
        tmp.y = center.point.y + x_axis.vector.y * moi.max_point_OBB.x + y_axis.vector.y * moi.max_point_OBB.y;
        tmp.z = center.point.z + x_axis.vector.z * moi.max_point_OBB.x + y_axis.vector.z * moi.max_point_OBB.y;
        corner.polygon.points.push_back(tmp);

        //geometry_msgs::PolygonStamped corner;
        //corner.header.frame_id = frame_id;
        //corner.header.stamp = ros::Time::now();
        //for (int i = 0; i < 4; i++)
        //{
        //    float z = z_values[vector_size];
        //    float x = (vertices[i].x - cx) * z / fx;
        //    float y = (vertices[i].y - cy) * z / fy;

        //    geometry_msgs::Point32 p32;
        //    p32.x = x;
        //    p32.y = y;
        //    p32.z = z;
        //    corner.polygon.points.push_back(p32);
        //}

        //centers_msg.centers.push_back(msg.center);
        //normals_msg.normals.push_back(msg.normal);
        //areas_msg.areas.push_back(area);

	moas_msg.centers.push_back(center);
	moas_msg.normals.push_back(normal);
	moas_msg.areas.push_back(area);
        moas_msg.corners.push_back(corner);
	moas_msg.x_axes.push_back(x_axis);
	moas_msg.y_axes.push_back(y_axis);
	moas_msg.z_axes.push_back(z_axis);

        //center_msg_list.push_back(centers_msg);
        //normal_msg_list.push_back(normals_msg);
        //area_msg_list.push_back(areas_msg);
	
	moas_msg_list.push_back(moas_msg);
    }
}

void computeCenter(cv::Mat mask)
{
    cv::Mat mask_index;
    cv::findNonZero(mask, mask_index);

    PointCloudT::Ptr cloud (new PointCloudT);
    pcl::CentroidPoint<PointT> centroid;
    PointT c1;
    std::vector<float> z_list;
    for (int i = 0; i < mask_index.total(); i++)
    {
        int u = mask_index.at<cv::Point>(i).x;
	int v = mask_index.at<cv::Point>(i).y;
	PointT p;
	p.z = depth.at<float>(v, u) / 1000;
	p.x = (u - cx) * p.z / fx;
	p.y = (v - cy) * p.z / fy;
	cloud->points.push_back(p);
        z_list.push_back(p.z);
        centroid.add(p);
    }

    centroid.get(c1);

    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);
    int K = 1;
    vector<int> indices_(K);
    vector<float> distance(K);
    kdtree.setInputCloud(cloud);
    kdtree.nearestKSearch(c1, K, indices_, distance);

    std::sort(z_list.begin(), z_list.end());
    size_t n = z_list.size() / 2;

    mask_rcnn_ros::MaskedObjectAttributes moas_msg;
    geometry_msgs::PointStamped center;
    geometry_msgs::Vector3Stamped normal;
    center.header.frame_id = frame_id;
    center.header.stamp = ros::Time::now();
    center.point.x = cloud->points[indices_[0]].x;
    center.point.y = cloud->points[indices_[0]].y;
    center.point.z = z_list[n];
    normal.header.frame_id = "world";
    normal.header.stamp = ros::Time::now();
    normal.vector.x = 0.0;
    normal.vector.y = 0.0;
    normal.vector.z = 1.0;

    moas_msg.centers.push_back(center);
    moas_msg.normals.push_back(normal);

    moas_msg_list.push_back(moas_msg);

    //CenterNormalMsg msg;
    //msg.center.header.frame_id = frame_id;
    //msg.center.header.stamp = ros::Time::now();
    //msg.center.point.x = cloud->points[indices_[0]].x;
    //msg.center.point.y = cloud->points[indices_[0]].y;
    //msg.center.point.z = z_list[n];
    //msg.normal.header.frame_id = "world";
    //msg.normal.header.stamp = ros::Time::now();
    //msg.normal.vector.x = 0.0;
    //msg.normal.vector.y = 0.0;
    //msg.normal.vector.z = 1.0;

    //mask_rcnn_ros::Centers centers_msg;
    //mask_rcnn_ros::Normals normals_msg;
    //
    //centers_msg.centers.push_back(msg.center);
    //normals_msg.normals.push_back(msg.normal);

    //center_msg_list.push_back(centers_msg);
    //normal_msg_list.push_back(normals_msg);
}

void markerInitialization()
{
    marker_axes_list = {};
    arrow_scale.x = 0.01, arrow_scale.y = 0.01, arrow_scale.z = 0.01;
    x_axis_color.r = 1.0, x_axis_color.g = 0.0, x_axis_color.b = 0.0, x_axis_color.a = 1.0;
    y_axis_color.r = 0.0, y_axis_color.g = 1.0, y_axis_color.b = 0.0, y_axis_color.a = 1.0;
    z_axis_color.r = 0.0, z_axis_color.g = 0.0, z_axis_color.b = 1.0, z_axis_color.a = 1.0;
    visualization_msgs::Marker del_msg = build_marker_msg(std_msgs::Header(), "", 0, 0, visualization_msgs::Marker::DELETEALL, geometry_msgs::Vector3(), std_msgs::ColorRGBA(), geometry_msgs::Point(), geometry_msgs::Vector3(), "");
    marker_axes_list.markers.push_back(del_msg);
}

void sceneRegionGrowing()
{
    pcl::search::Search<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
    vector<pcl::PointIndices> indices;
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (scene);
    normal_estimator.setKSearch (30);
    //normal_estimator.setRadiusSearch(0.02);
    normal_estimator.compute (*scene_normals);
    scene_reg.setMinClusterSize (50);
    scene_reg.setMaxClusterSize (7000);
    scene_reg.setSearchMethod (tree);
    scene_reg.setNumberOfNeighbours (50);
    scene_reg.setInputCloud (scene);
    scene_reg.setInputNormals (scene_normals);
    scene_reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
    scene_reg.setCurvatureThreshold (1.0);
    scene_reg.extract(indices);
    scene_surface_list.push_back(scene_reg.getColoredCloud());
    ROS_INFO("Scene RegionGrowing finished");
}

void publishDrawedDepthMap()
{
    cv::Mat vis_depth;
    cv::Mat tmp;
    depth.convertTo(tmp, CV_8UC1);
    cv::cvtColor(tmp, vis_depth, CV_GRAY2BGR);
    for (sensor_msgs::Image mask_msg : mask_msgs)
    {
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(mask_msg, "8UC1");
        cv::Mat mask = cv_ptr->image.clone();

        cv::threshold(mask, mask, 125, 255, cv::THRESH_BINARY);
        vector<vector<cv::Point> > contours;
        vector<cv::Vec4i> hierarchy;
        cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        cv::drawContours(vis_depth, contours, -1, cv::Scalar(0, 0, 255), 4);
    }

    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = frame_id;
    sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(header, "bgr8", vis_depth).toImageMsg();
    
    masked_depth_map_pub.publish(image_msg);
}

bool callbackGetMaskedSurface(mask_rcnn_ros::GetMaskedSurface::Request &req, mask_rcnn_ros::GetMaskedSurface::Response &res)
{
    ROS_INFO("Service called");
    ros::WallTime start_process_time = ros::WallTime::now();

    scene_surface_list = {};
    masked_surface_list = {};
    //center_msg_list = {};
    //normal_msg_list = {};
    //area_msg_list = {};
    //corner_msg_list = {};
    moas_msg_list = {};
    markerInitialization();

    sceneRegionGrowing();
   
    int count = 0;
    mask_msgs = req.masks;
    bool is_rigid_object = req.is_rigid_object;
    for (sensor_msgs::Image mask_msg : mask_msgs)
    {
        cv_bridge::CvImagePtr cv_ptr;
	cv_ptr = cv_bridge::toCvCopy(mask_msg, "8UC1");
	cv::Mat mask = cv_ptr->image.clone();

        //offset
        //cv::Mat offset = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 5.0, 0.0, 1.0, 0.0);
        //cv::warpAffine(mask, mask, offset, mask.size());

	ROS_INFO("ID:%d", count);
	if (is_rigid_object)
            maskedRegionGrowing(mask);
        else
            computeCenter(mask);

	count++;
    }

    //res.centers_list = center_msg_list;
    //res.normals_list = normal_msg_list;
    //res.areas_list = area_msg_list;
    res.moas_list = moas_msg_list;
    //for (int i = 0; i < center_msg_list.size(); i++)
    //{
    //    CenterNormalMsg msg; 
    //    msg.center = center_msg_list[i];
    //    msg.normal = normal_msg_list[i];
    //    visualization_msgs::Marker z_axis_marker = build_marker_msg(msg.center.header, "mask_region_growing_normal", i, visualization_msgs::Marker::ARROW, visualization_msgs::Marker::ADD, arrow_scale, z_axis_color, msg.center.point, msg.normal.vector, "normal_" + to_string(i));
    //    marker_axes_list.markers.push_back(z_axis_marker);
    //    res.centers.push_back(msg.center);
    //    res.normals.push_back(msg.normal);
    //}
    count = 0;
    for (int i = 0; i < moas_msg_list.size(); i++)
    {
	geometry_msgs::PointStamped center = moas_msg_list[i].centers[0];
        geometry_msgs::Vector3Stamped x_axis = moas_msg_list[i].x_axes[0];
        geometry_msgs::Vector3Stamped y_axis = moas_msg_list[i].y_axes[0];
        geometry_msgs::Vector3Stamped z_axis = moas_msg_list[i].z_axes[0];
        geometry_msgs::PolygonStamped corner = moas_msg_list[i].corners[0];

	visualization_msgs::Marker x_axis_marker = build_marker_msg(center.header, "mask_region_growing_x_axis", i, visualization_msgs::Marker::ARROW, visualization_msgs::Marker::ADD, arrow_scale, x_axis_color, center.point, x_axis.vector, "x_axis_" + to_string(i));
	visualization_msgs::Marker y_axis_marker = build_marker_msg(center.header, "mask_region_growing_y_axis", i, visualization_msgs::Marker::ARROW, visualization_msgs::Marker::ADD, arrow_scale, y_axis_color, center.point, y_axis.vector, "y_axis_" + to_string(i));
	visualization_msgs::Marker z_axis_marker = build_marker_msg(center.header, "mask_region_growing_z_axis", i, visualization_msgs::Marker::ARROW, visualization_msgs::Marker::ADD, arrow_scale, z_axis_color, center.point, z_axis.vector, "z_axis_" + to_string(i));

        for (int j = 0; j < corner.polygon.points.size(); j++)
        {
            geometry_msgs::PointStamped c;
            c.header = corner.header;
            c.point.x = corner.polygon.points[j].x;
            c.point.y = corner.polygon.points[j].y;
            c.point.z = corner.polygon.points[j].z;

	    visualization_msgs::Marker sphere_marker = build_marker_msg(corner.header, "mask_region_growing_corner", count, visualization_msgs::Marker::SPHERE, visualization_msgs::Marker::ADD, arrow_scale, z_axis_color, c.point, z_axis.vector, "corner_" + to_string(count));
            count++;
            marker_axes_list.markers.push_back(sphere_marker);
        }

	marker_axes_list.markers.push_back(x_axis_marker);
	marker_axes_list.markers.push_back(y_axis_marker);
	marker_axes_list.markers.push_back(z_axis_marker);
    }

    //for (int i = 0; i < corner_msg_list.size(); i++)
    //{
    //    geometry_msgs::Vector3Stamped tmp = geometry_msgs::Vector3Stamped();;
    //    geometry_msgs::PointStamped corner = corner_msg_list[i];
    //    visualization_msgs::Marker sphere_marker = build_marker_msg(corner.header, "mask_region_growing_corner", i, visualization_msgs::Marker::SPHERE, visualization_msgs::Marker::ADD, arrow_scale, z_axis_color, corner.point, tmp.vector, "corner_" + to_string(i));
    //    marker_axes_list.markers.push_back(sphere_marker);        
    //}

    vis_axes_marker.publish(marker_axes_list);
   
    ros::WallTime end_process_time = ros::WallTime::now();
    double execution_process_time = (end_process_time - start_process_time).toNSec() * 1e-9;
    ROS_INFO("Total Service time(s): %.3f", execution_process_time);

    is_service_called = true;

    return true;    
}

void callbackDepth(const sensor_msgs::ImageConstPtr& depth_msg)
{
    ROS_INFO("Subscribed Depth Image");
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(depth_msg, "32FC1");
    depth = cv_ptr->image.clone();

    //if (debug_mode) {
        //depth = cv_ptr->image.clone();
    //} else {
        //cv::Mat tmp = cv_ptr->image.clone();
        //cv::Mat mapx, mapy;
        //cv::Size imageSize(tmp.cols, tmp.rows);
        //cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, imageSize, CV_32FC1, mapx, mapy);
        //cv::remap(tmp, depth, mapx, mapy, CV_INTER_NN);
    //}

    int width = depth.cols;
    int height = depth.rows;
    scene->width = width;
    scene->height = height;
    scene->resize(width*height);
    for (int v = 0; v < height; v++) {
        float *src = depth.ptr<float>(v);
        for (int u = 0; u < width; u++) {
            PointT p;
            p.z = src[u] / 1000;
            p.x = (u - cx) * p.z / fx;
            p.y = (v - cy) * p.z / fy;
            scene->at(u, v) = p;
        }
    }

    pcl::copyPointCloud(*scene, *input_scene);

    //downsampling
    pcl::VoxelGrid<PointT> voxelSampler;
    voxelSampler.setInputCloud(scene);
    voxelSampler.setLeafSize(0.002, 0.002, 0.002);
    voxelSampler.filter(*scene);

    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud (scene);
    sor.setMeanK (50);
    sor.setStddevMulThresh (0.6);
    sor.filter (*scene);

    //sensor_msgs::PointCloud2 output_cloud;
    //pcl::toROSMsg(*scene, output_cloud);
    //output_cloud.header.frame_id = frame_id;
    //output_cloud.header.stamp = ros::Time::now();

    //scene_surface_pointcloud_pub.publish(output_cloud);

    ROS_INFO("Preprocessed Finished");
}

void setCameraParameters()
{
    cv::Mat K(3, 3, CV_64FC1, (void *)camera_info->K.data());
    cv::Mat D(1, 5, CV_64FC1, (void *)camera_info->D.data());

    cameraMatrix = K;
    distCoeffs = D;
    fx = K.at<double>(0, 0);
    fy = K.at<double>(1, 1);
    cx = K.at<double>(0, 2);
    cy = K.at<double>(1, 2);

}

int main(int argc, char *argv[])
{
    //Initialize ROS
    ros::init (argc, argv, "mask_region_growing");
    ros::NodeHandle nh;

    if (nh.hasParam("/mask_rcnn/debug_mode"))
    {
        nh.getParam("/mask_rcnn/debug_mode", debug_mode);
        if (debug_mode) {
            ROS_WARN("'mask_region_growing node' is Debug Mode");
            depth_topic = debug_depth_topic;
        }
    }

    ROS_INFO("Waiting camera info from %s", camera_info_topic.c_str());
    camera_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(camera_info_topic);
    ROS_INFO("Received Camera Info");
    setCameraParameters();

    ros::Subscriber subDepth = nh.subscribe<sensor_msgs::Image>(depth_topic, 1, callbackDepth);

    ros::ServiceServer get_local_surface = nh.advertiseService(ros::this_node::getName() + "/get_masked_surface", callbackGetMaskedSurface);

    vis_axes_marker = nh.advertise<visualization_msgs::MarkerArray>(ros::this_node::getName() + "/axes", 0, true);

    scene_surface_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/scene_surface_pointcloud", 0, true);
    masked_surface_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/masked_surface_pointcloud", 0, true);
    masked_surface_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/masked_surface_pointcloud", 0, true);
    input_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/input_pointcloud", 0, true);
    masked_depth_map_pub = nh.advertise<sensor_msgs::Image>(ros::this_node::getName() + "/masked_depth_map", 0, true);

    ros::Rate loop_rate(10);
    while(ros::ok())
    {
        if (is_service_called) {
            PointCloudColorT::Ptr scene_surfaces(new PointCloudColorT);
            PointCloudColorT::Ptr masked_surfaces(new PointCloudColorT);
            for (PointCloudColorT::Ptr scene_surface : scene_surface_list) {
                *scene_surfaces += *scene_surface;
            }
            for (PointCloudColorT::Ptr masked_surface : masked_surface_list) {
                *masked_surfaces += *masked_surface;
            }
            // publish scence surface
            sensor_msgs::PointCloud2 scene_surface_msg;
            pcl::toROSMsg(*scene_surfaces, scene_surface_msg);
            scene_surface_msg.header.frame_id = frame_id;
            scene_surface_msg.header.stamp = ros::Time::now();
            scene_surface_pointcloud_pub.publish(scene_surface_msg);
            // publish masked surface
            sensor_msgs::PointCloud2 masked_surface_msg;
            pcl::toROSMsg(*masked_surfaces, masked_surface_msg);
            masked_surface_msg.header.frame_id = frame_id;
            masked_surface_msg.header.stamp = ros::Time::now();
            masked_surface_pointcloud_pub.publish(masked_surface_msg);
            // publish input pointcloud
            sensor_msgs::PointCloud2 input_pointcloud_msg;
            pcl::toROSMsg(*input_scene, input_pointcloud_msg);
            input_pointcloud_msg.header.frame_id = frame_id;
            input_pointcloud_msg.header.stamp = ros::Time::now();
            input_pointcloud_pub.publish(input_pointcloud_msg);
            // publish aligned_depth_map drawed some information
            publishDrawedDepthMap();

            is_service_called = false;
            
        }
        ros::spinOnce();
        loop_rate.sleep();
    }

    ros::spin();
}
